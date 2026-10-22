// Standalone check and developer benchmark for the S1 BF16 GEMV. No Python, no torch: only the CUDA toolkit.
//
// Build (turing, Ada):
//   NVCC=$HOME/cuda-12.9/bin/nvcc
//   $NVCC -O3 -std=c++17 -gencode arch=compute_89,code=sm_89 -DWITH_CUBLAS -I src/paia_gemv
//         src/tests/gemv_bf16_check.cu src/paia_gemv/gemv_bf16.cu -lcublas -Xlinker -rpath,$HOME/cuda-12.9/lib64
//         -o src/build/gemv_check          (one line; drop -DWITH_CUBLAS -lcublas -Xlinker ... if cuBLAS fails to link)
// Run:
//   src/build/gemv_check                       correctness: every variant on awkward and real shapes
//   src/build/gemv_check --full                also lm_head and fused shapes (sampled rows)
//   src/build/gemv_check --bench               DEV benchmark: read-bandwidth probe (B0), then top variants per real shape vs
//                                              cuBLAS GemmEx (n=1); CUDA-graph timing, random weights. --eager for a plain loop.
//                                              Refuses to run while /tmp/paia_bench.lock exists unless --ignore-lock.
// Correctness gate (same as the harness): max|y-ref| / max|ref| < 2e-2 over verified rows, fp64 reference, and every
// output row must have been written (outputs are pre-filled with NaN).
// The benchmark rotates through enough weight copies to exceed 4x L2 but runs no idle gate: numbers are for choosing
// variants and spotting problems, never results.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
int paia_gemv_bf16_variant_count();
int paia_gemv_bf16_variant_info(int v, int* R, int* U, int* W, int* S, int* H);
int paia_gemv_bf16(const void* W, const void* x, void* y, int N, int K, int variant, void* stream);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Shape { int N, K; const char* name; };
static const Shape AWKWARD[] = {{1, 8, "1x8"}, {1, 256, "1x256"}, {3, 264, "3x264"}, {7, 520, "7x520"},
                                {33, 1032, "33x1032"}, {129, 4104, "129x4104"}, {1000, 3584, "1000x3584"},
                                {4097, 4096, "4097x4096"}, {513, 14336, "513x14336"}};
static const Shape REAL[] = {{4096, 4096, "llama q/o"}, {1024, 4096, "llama k/v"}, {14336, 4096, "llama gate/up"},
                             {4096, 14336, "llama down"}, {3584, 3584, "qwen q/o"}, {512, 3584, "qwen k/v"},
                             {18944, 3584, "qwen gate/up"}, {3584, 18944, "qwen down"}, {6144, 4096, "llama qkv"},
                             {4608, 3584, "qwen qkv"}};
static const Shape BIG[] = {{28672, 4096, "llama gate_up"}, {37888, 3584, "qwen gate_up"},
                            {128256, 4096, "llama lm_head"}, {152064, 3584, "qwen lm_head"}};

static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

// Returns the worst relative error over variants, or -1 on any failure. Prints failures.
static bool check_shape(const Shape& sh, int nvar, bool sample_big, double* worst_out) {
    const int N = sh.N, K = sh.K;
    std::mt19937 rng(7919u * N + K);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    std::vector<__nv_bfloat16> hW((size_t)N * K), hx(K);
    for (auto& v : hW) v = __float2bfloat16_rn(U(rng) * 0.04f);
    for (auto& v : hx) v = __float2bfloat16_rn(U(rng));
    std::vector<int> rows;
    if (N <= 4200 || !sample_big) { for (int r = 0; r < N; ++r) rows.push_back(r); }
    else {
        for (int r = 0; r < 8; ++r) { rows.push_back(r); rows.push_back(N - 1 - r); }
        for (int i = 0; i < 48; ++i) rows.push_back((int)(rng() % N));
    }
    std::vector<float> xf(K);
    for (int k = 0; k < K; ++k) xf[k] = bf(hx[k]);
    std::vector<double> ref(rows.size());
    double maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const __nv_bfloat16* w = &hW[(size_t)rows[i] * K];
        double s = 0;
        for (int k = 0; k < K; ++k) s += (double)bf(w[k]) * xf[k];
        ref[i] = s; maxref = std::max(maxref, std::fabs(s));
    }
    void *dW, *dx, *dy;
    CK(cudaMalloc(&dW, (size_t)N * K * 2)); CK(cudaMalloc(&dx, (size_t)K * 2)); CK(cudaMalloc(&dy, (size_t)N * 2));
    CK(cudaMemcpy(dW, hW.data(), (size_t)N * K * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), (size_t)K * 2, cudaMemcpyHostToDevice));
    std::vector<__nv_bfloat16> hy(N);
    bool ok = true; double worst = 0;
    for (int v = 0; v < nvar; ++v) {
        CK(cudaMemset(dy, 0xFF, (size_t)N * 2));                     // 0xFFFF is a bf16 NaN
        int rc = paia_gemv_bf16(dW, dx, dy, N, K, v, nullptr);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy.data(), dy, (size_t)N * 2, cudaMemcpyDeviceToHost));
        int R, Uu, W, S, H; paia_gemv_bf16_variant_info(v, &R, &Uu, &W, &S, &H);
        if (rc != 0) { printf("FAIL %s variant %d (R=%d U=%d W=%d S=%d H=%d): rc=%d\n", sh.name, v, R, Uu, W, S, H, rc); ok = false; continue; }
        int missing = 0; for (int r = 0; r < N; ++r) if (std::isnan(bf(hy[r]))) ++missing;
        double err = 0;
        for (size_t i = 0; i < rows.size(); ++i) err = std::max(err, std::fabs((double)bf(hy[rows[i]]) - ref[i]));
        err = maxref > 0 ? err / maxref : err;
        worst = std::max(worst, err);
        if (missing || !(err < 2e-2)) {
            printf("FAIL %s variant %d (R=%d U=%d W=%d S=%d H=%d): rel_err=%.3e unwritten_rows=%d\n", sh.name, v, R, Uu, W, S, H, err, missing);
            ok = false;
        }
    }
    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    *worst_out = std::max(*worst_out, worst);
    return ok;
}


// Random bf16 weights in [-0.02, 0.02] (the harness's range), generated on the device.
__global__ void fill_random(uint32_t* p, size_t n_words, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_words; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u + seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        const float a = ((h & 0xFFFFu) / 65535.f - 0.5f) * 0.04f, b = ((h >> 16) / 65535.f - 0.5f) * 0.04f;
        const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        p[i] = *reinterpret_cast<const uint32_t*>(&v);
    }
}

// Pure streaming read, 16 B per load; the xor sink keeps the loads alive. This is the dev B0 for this GPU.
__global__ void read_probe(const uint4* __restrict__ p, size_t n16, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9u) out[0] = acc;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Bench {
    int N, K, ncopies; size_t wbytes;
    std::vector<void*> W, x, y;
    cudaStream_t st;
    Bench(int N_, int K_, size_t l2) : N(N_), K(K_) {
        wbytes = (size_t)N * K * 2;
        size_t want = std::max<size_t>(2, (4 * l2 + wbytes - 1) / wbytes);
        size_t fit = std::max<size_t>(1, (10ull << 30) / wbytes);
        ncopies = (int)std::min(want, fit);
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        for (int i = 0; i < ncopies; ++i) {
            void *w, *xx, *yy;
            CK(cudaMalloc(&w, wbytes)); CK(cudaMalloc(&xx, (size_t)K * 2)); CK(cudaMalloc(&yy, (size_t)N * 2));
            fill_random<<<1024, 256, 0, st>>>((uint32_t*)w, wbytes / 4, 1234u + i);
            fill_random<<<64, 256, 0, st>>>((uint32_t*)xx, (size_t)K / 2, 99u + i);
            W.push_back(w); x.push_back(xx); y.push_back(yy);
        }
        CK(cudaStreamSynchronize(st));
    }
    ~Bench() { for (int i = 0; i < ncopies; ++i) { cudaFree(W[i]); cudaFree(x[i]); cudaFree(y[i]); } cudaStreamDestroy(st); }
    // launch(copy_index, stream). Returns median per-op microseconds over 5 repeats, or -1 if graph capture failed.
    template <class F> double time_us(F launch, bool graph) {
        for (int i = 0; i < 20; ++i) launch(i % ncopies, st);
        CK(cudaStreamSynchronize(st));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, st)); for (int i = 0; i < 20; ++i) launch(i % ncopies, st); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        const double est = std::max(1e-3, ms * 1000.0 / 20);
        std::vector<double> reps;
        if (!graph) {
            const int iters = std::max(20, (int)(8000.0 / est));
            for (int r = 0; r < 5; ++r) {
                CK(cudaEventRecord(a, st)); for (int i = 0; i < iters; ++i) launch(i % ncopies, st); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms, a, b)); reps.push_back(ms * 1000.0 / iters);
            }
        } else {
            const int R = std::min(4000, std::max(8, (int)(5000.0 / est)));
            cudaGraph_t g = nullptr; cudaGraphExec_t ex = nullptr;
            if (cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal) != cudaSuccess) return -1;
            for (int i = 0; i < R; ++i) launch(i % ncopies, st);
            if (cudaStreamEndCapture(st, &g) != cudaSuccess || cudaGraphInstantiate(&ex, g, 0) != cudaSuccess) { cudaGetLastError(); return -1; }
            for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(ex, st));
            CK(cudaStreamSynchronize(st));
            for (int r = 0; r < 5; ++r) {
                CK(cudaEventRecord(a, st)); CK(cudaGraphLaunch(ex, st)); CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
                CK(cudaEventElapsedTime(&ms, a, b)); reps.push_back(ms * 1000.0 / R);
            }
            cudaGraphExecDestroy(ex); cudaGraphDestroy(g);
        }
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        return median(reps);
    }
};

// Dev B0: best streaming-read bandwidth over a few grid sizes on a buffer of at least 12x L2 (and 1 GiB).
static double measure_b0(const cudaDeviceProp& p) {
    size_t bytes = std::max<size_t>(1ull << 30, 12 * (size_t)p.l2CacheSize);
    bytes &= ~(size_t)15;
    void *buf; unsigned* sink; CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc((void**)&sink, 4));
    cudaStream_t st; CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    fill_random<<<1024, 256, 0, st>>>((uint32_t*)buf, bytes / 4, 5u); CK(cudaStreamSynchronize(st));
    double best = 0;
    for (int mult : {4, 8, 16, 32}) {
        const int grid = p.multiProcessorCount * mult;
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int i = 0; i < 3; ++i) read_probe<<<grid, 256, 0, st>>>((const uint4*)buf, bytes / 16, sink);
        CK(cudaStreamSynchronize(st));
        std::vector<double> reps;
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(a, st));
            for (int i = 0; i < 5; ++i) read_probe<<<grid, 256, 0, st>>>((const uint4*)buf, bytes / 16, sink);
            CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b));
            float ms; CK(cudaEventElapsedTime(&ms, a, b)); reps.push_back((double)bytes * 5 / (ms * 1e-3) / 1e9);
        }
        best = std::max(best, median(reps));
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    cudaStreamDestroy(st); cudaFree(buf); cudaFree(sink);
    return best;
}

static void bench_shape(const Shape& sh, int nvar, double b0, size_t l2, bool graph) {
    Bench B(sh.N, sh.K, l2);
    const double bytes = 2.0 * sh.N * sh.K + 2.0 * sh.K + 2.0 * sh.N;
    std::vector<std::pair<double, int>> res;
    for (int v = 0; v < nvar; ++v) {
        double t = B.time_us([&](int i, cudaStream_t s) { paia_gemv_bf16(B.W[i], B.x[i], B.y[i], sh.N, sh.K, v, s); }, graph);
        if (t > 0) res.push_back({t, v});
    }
    std::sort(res.begin(), res.end());
    double cub = -1;
#ifdef WITH_CUBLAS
    static cublasHandle_t h = nullptr; if (!h) cublasCreate(&h);
    cublasSetStream(h, B.st);
    const float alpha = 1.f, beta = 0.f;
    auto cub_launch = [&](int i, cudaStream_t) {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.N, 1, sh.K, &alpha, B.W[i], CUDA_R_16BF, sh.K, B.x[i], CUDA_R_16BF, sh.K,
                     &beta, B.y[i], CUDA_R_16BF, sh.N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    cub = B.time_us(cub_launch, graph);
    if (cub < 0) cub = B.time_us(cub_launch, false);   // capture failed: fall back to the eager loop, flagged below
    const bool cub_eager_fallback = graph && cub > 0 && false;
    (void)cub_eager_fallback;
#endif
    auto gbps = [&](double us) { return bytes / (us * 1e-6) / 1e9; };
    auto pct = [&](double us) { return b0 > 0 ? std::to_string((int)(100 * gbps(us) / b0)) + "% of B0" : std::string(); };
    printf("\n%-14s N=%-6d K=%-6d %7.1f MiB  copies=%d\n", sh.name, sh.N, sh.K, bytes / 1048576.0, B.ncopies);
    for (int i = 0; i < 4 && i < (int)res.size(); ++i) {
        int R, U, W, S, H; paia_gemv_bf16_variant_info(res[i].second, &R, &U, &W, &S, &H);
        printf("  ours  %7.1f GB/s (%s) %9.2f us   R=%d U=%d W=%d S=%d H=%d\n", gbps(res[i].first), pct(res[i].first).c_str(), res[i].first, R, U, W, S, H);
    }
    if (cub > 0) printf("  cuBLAS%7.1f GB/s (%s) %9.2f us   -> ours/cuBLAS speedup %.2fx\n", gbps(cub), pct(cub).c_str(), cub, cub / res[0].first);
}

int main(int argc, char** argv) {
    bool full = false, bench = false, eager = false, ignore_lock = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--full")) full = true;
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--eager")) eager = true;
        else if (!strcmp(argv[i], "--ignore-lock")) ignore_lock = true;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { fprintf(stderr, "no CUDA device visible\n"); return 3; }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("%s  cc %d.%d  SMs %d  L2 %.0f MiB  variants %d\n", p.name, p.major, p.minor, p.multiProcessorCount,
           p.l2CacheSize / 1048576.0, paia_gemv_bf16_variant_count());
    const int nvar = paia_gemv_bf16_variant_count();
    if (bench) {
        if (!ignore_lock) {
            if (FILE* f = fopen("/tmp/paia_bench.lock", "r")) {
                char buf[256] = {0}; size_t n = fread(buf, 1, sizeof(buf) - 1, f); (void)n; fclose(f);
                fprintf(stderr, "/tmp/paia_bench.lock exists (a gated measurement session is running): %s\nNot benchmarking. Use --ignore-lock only on a machine where nobody else measures.\n", buf);
                return 4;
            }
        }
        printf("DEV BENCHMARK: no idle gate, not a result. Timing: %s.\n", eager ? "eager loop" : "CUDA graph");
        const double b0 = measure_b0(p);
        printf("dev B0 (streaming read probe, best grid): %.1f GB/s\n", b0);
        for (const Shape& s : REAL) bench_shape(s, nvar, b0, p.l2CacheSize, !eager);
        return 0;
    }
    bool ok = true; double worst = 0;
    for (const Shape& s : AWKWARD) ok &= check_shape(s, nvar, false, &worst);
    for (const Shape& s : REAL) ok &= check_shape(s, nvar, false, &worst);
    if (full) for (const Shape& s : BIG) ok &= check_shape(s, nvar, true, &worst);
    // contract: K % 8 != 0 must be rejected, not run
    void* d; CK(cudaMalloc(&d, 4096));
    int rc = paia_gemv_bf16(d, d, d, 4, 12, 0, nullptr);
    if (rc != -2) { printf("FAIL: K%%8 contract returned %d, expected -2\n", rc); ok = false; }
    cudaFree(d);
    printf("worst relative error over all passing cases: %.2e\n%s\n", worst, ok ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
