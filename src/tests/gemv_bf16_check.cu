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
//   src/build/gemv_check --quant               correctness of FP8 W8A16 and INT4 W4A16 g128 (all variants, naive, unfused)
//   src/build/gemv_check --bench-quant         DEV benchmark: per real shape, bf16 best/cuBLAS vs fp8, int4 fused/naive/unfused
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
#include "quant_host.h"

extern "C" {
int paia_gemv_bf16_variant_count();
int paia_gemv_bf16_variant_info(int v, int* R, int* U, int* W, int* S, int* H);
int paia_gemv_bf16(const void* W, const void* x, void* y, int N, int K, int variant, void* stream);
int paia_gemv_fp8_variant_count();
int paia_gemv_fp8_variant_info(int v, int* R, int* U, int* W, int* H);
int paia_gemv_fp8(const void* Wq, const void* scale, const void* x, void* y, int N, int K, int variant, void* stream);
int paia_gemv_int4_variant_count();
int paia_gemv_int4_variant_info(int v, int* R, int* U, int* W, int* H);
int paia_gemv_int4(const void* Wq, const void* scales, const void* x, void* y, int N, int K, int variant, void* stream);
int paia_gemv_int4_naive(const void* Wq, const void* scales, const void* x, void* y, int N, int K, void* stream);
int paia_gemv_int4_unfused(const void* Wq, const void* scales, const void* x, void* y, void* scratch_bf16, int N, int K, void* stream);
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

struct Timer {
    cudaStream_t st = nullptr;
    int ncopies = 1;
    // launch(copy_index, stream). Median per-op microseconds over 5 repeats, or -1 if graph capture failed.
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

struct Bench : Timer {
    int N, K; size_t wbytes;
    std::vector<void*> W, x, y;
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
};

// Quantized operands: Wq (nbytes_w per copy), S (nbytes_s per copy), x, y. Random bytes are fine for timing: the decode
// has no data-dependent branches. Footprint rotation as in Bench.
struct QBench : Timer {
    int N, K; size_t wb, sb;
    std::vector<void*> W, S, x, y;
    void* scratch = nullptr;
    QBench(int N_, int K_, size_t wb_, size_t sb_, size_t l2, bool want_scratch) : N(N_), K(K_), wb(wb_), sb(sb_) {
        const size_t per = wb + sb + (size_t)K * 2 + (size_t)N * 2;
        size_t want = std::max<size_t>(2, (4 * l2 + per - 1) / per);
        size_t fit = std::max<size_t>(1, (8ull << 30) / per);
        ncopies = (int)std::min(want, fit);
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        for (int i = 0; i < ncopies; ++i) {
            void *w, *s_, *xx, *yy;
            CK(cudaMalloc(&w, wb)); CK(cudaMalloc(&s_, sb)); CK(cudaMalloc(&xx, (size_t)K * 2)); CK(cudaMalloc(&yy, (size_t)N * 2));
            fill_random<<<1024, 256, 0, st>>>((uint32_t*)w, wb / 4, 4321u + i);
            fill_random<<<256, 256, 0, st>>>((uint32_t*)s_, sb / 4, 777u + i);
            fill_random<<<64, 256, 0, st>>>((uint32_t*)xx, (size_t)K / 2, 99u + i);
            W.push_back(w); S.push_back(s_); x.push_back(xx); y.push_back(yy);
        }
        if (want_scratch) CK(cudaMalloc(&scratch, (size_t)N * K * 2));
        CK(cudaStreamSynchronize(st));
    }
    ~QBench() { for (int i = 0; i < ncopies; ++i) { cudaFree(W[i]); cudaFree(S[i]); cudaFree(x[i]); cudaFree(y[i]); } if (scratch) cudaFree(scratch); cudaStreamDestroy(st); }
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


// ------------------------------------------------------------------------------------------------------------------
// Quantized kernels: FP8 W8A16 and INT4 W4A16 g128 (+ naive and unfused INT4 controls)
// ------------------------------------------------------------------------------------------------------------------
static const Shape QAWK_FP8[] = {{1, 16, "1x16"}, {3, 528, "3x528"}, {7, 1040, "7x1040"}, {33, 1552, "33x1552"},
                                 {129, 4112, "129x4112"}, {1000, 3584, "1000x3584"}, {4097, 4096, "4097x4096"}, {513, 14336, "513x14336"}};
static const Shape QAWK_I4[] = {{1, 128, "1x128"}, {3, 384, "3x384"}, {7, 1152, "7x1152"}, {33, 2176, "33x2176"},
                                {129, 4224, "129x4224"}, {1000, 3584, "1000x3584"}, {4097, 4096, "4097x4096"}, {513, 14336, "513x14336"}};

struct QData {
    int N, K; std::vector<float> W; std::vector<__nv_bfloat16> hx; std::vector<float> xf;
};
static QData make_q(const Shape& sh) {
    QData d; d.N = sh.N; d.K = sh.K;
    std::mt19937 rng(31337u * sh.N + sh.K);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    d.W.resize((size_t)sh.N * sh.K); d.hx.resize(sh.K); d.xf.resize(sh.K);
    for (auto& v : d.W) v = U(rng) * 0.04f;
    for (int k = 0; k < sh.K; ++k) { d.hx[k] = __float2bfloat16_rn(U(rng)); d.xf[k] = bf(d.hx[k]); }
    return d;
}
static std::vector<int> pick_rows(int N, std::mt19937& rng) {
    std::vector<int> rows;
    if (N <= 4200) { for (int r = 0; r < N; ++r) rows.push_back(r); }
    else { for (int r = 0; r < 8; ++r) { rows.push_back(r); rows.push_back(N - 1 - r); } for (int i = 0; i < 64; ++i) rows.push_back((int)(rng() % N)); }
    return rows;
}
// Compares y against ref (rows) and checks every row was written. Returns the relative error, or 1e9 on a missing row.
static double compare_y(const std::vector<__nv_bfloat16>& hy, const std::vector<int>& rows, const std::vector<double>& ref, double maxref, int* missing) {
    *missing = 0; for (size_t r = 0; r < hy.size(); ++r) if (std::isnan(bf(hy[r]))) ++(*missing);
    double err = 0; for (size_t i = 0; i < rows.size(); ++i) err = std::max(err, std::fabs((double)bf(hy[rows[i]]) - ref[i]));
    return maxref > 0 ? err / maxref : err;
}

static bool check_fp8_shape(const Shape& sh, double* worst) {
    QData d = make_q(sh); const int N = sh.N, K = sh.K;
    std::vector<uint8_t> q((size_t)N * K); std::vector<float> scale(N);
    paia_host::quantize_fp8_rowwise(d.W.data(), N, K, q.data(), scale.data());
    std::mt19937 rng(5); std::vector<int> rows = pick_rows(N, rng);
    std::vector<double> ref(rows.size()); double maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i) { double s = 0; for (int k = 0; k < K; ++k) s += (double)paia_host::dequant_fp8(q.data(), scale.data(), K, rows[i], k) * d.xf[k]; ref[i] = s; maxref = std::max(maxref, std::fabs(s)); }
    void *dW, *dS, *dx, *dy; CK(cudaMalloc(&dW, q.size())); CK(cudaMalloc(&dS, N * 4)); CK(cudaMalloc(&dx, (size_t)K * 2)); CK(cudaMalloc(&dy, (size_t)N * 2));
    CK(cudaMemcpy(dW, q.data(), q.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dS, scale.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, d.hx.data(), (size_t)K * 2, cudaMemcpyHostToDevice));
    bool ok = true; std::vector<__nv_bfloat16> hy(N);
    for (int v = 0; v < paia_gemv_fp8_variant_count(); ++v) {
        CK(cudaMemset(dy, 0xFF, (size_t)N * 2));
        const int rc = paia_gemv_fp8(dW, dS, dx, dy, N, K, v, nullptr); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hy.data(), dy, (size_t)N * 2, cudaMemcpyDeviceToHost));
        int R, U, W, H; paia_gemv_fp8_variant_info(v, &R, &U, &W, &H); int miss;
        const double err = rc ? 1e9 : compare_y(hy, rows, ref, maxref, &miss); if (rc) miss = -1;
        *worst = std::max(*worst, rc ? 0.0 : err);
        if (rc || miss || !(err < 2e-2)) { printf("FAIL fp8 %s variant %d (R=%d U=%d W=%d H=%d): rc=%d rel_err=%.3e unwritten=%d\n", sh.name, v, R, U, W, H, rc, err, miss); ok = false; }
    }
    cudaFree(dW); cudaFree(dS); cudaFree(dx); cudaFree(dy); return ok;
}

static bool check_int4_shape(const Shape& sh, double* worst) {
    QData d = make_q(sh); const int N = sh.N, K = sh.K;
    std::vector<uint8_t> pk((size_t)N * K / 2); std::vector<uint16_t> sc((size_t)N * K / 128);
    paia_host::quantize_int4_g128(d.W.data(), N, K, pk.data(), sc.data());
    std::mt19937 rng(6); std::vector<int> rows = pick_rows(N, rng);
    std::vector<double> ref(rows.size()); double maxref = 0;
    for (size_t i = 0; i < rows.size(); ++i) { double s = 0; for (int k = 0; k < K; ++k) s += (double)paia_host::dequant_int4(pk.data(), sc.data(), K, rows[i], k) * d.xf[k]; ref[i] = s; maxref = std::max(maxref, std::fabs(s)); }
    void *dW, *dS, *dx, *dy, *dsc; CK(cudaMalloc(&dW, pk.size())); CK(cudaMalloc(&dS, sc.size() * 2)); CK(cudaMalloc(&dx, (size_t)K * 2)); CK(cudaMalloc(&dy, (size_t)N * 2)); CK(cudaMalloc(&dsc, (size_t)N * K * 2));
    CK(cudaMemcpy(dW, pk.data(), pk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dS, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, d.hx.data(), (size_t)K * 2, cudaMemcpyHostToDevice));
    bool ok = true; std::vector<__nv_bfloat16> hy(N);
    auto verify = [&](const char* what, int v, int rc) {
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy.data(), dy, (size_t)N * 2, cudaMemcpyDeviceToHost));
        int miss = -1; const double err = rc ? 1e9 : compare_y(hy, rows, ref, maxref, &miss); *worst = std::max(*worst, rc ? 0.0 : err);
        if (rc || miss || !(err < 2e-2)) { printf("FAIL int4 %s %s variant %d: rc=%d rel_err=%.3e unwritten=%d\n", what, sh.name, v, rc, err, miss); ok = false; }
    };
    for (int v = 0; v < paia_gemv_int4_variant_count(); ++v) { CK(cudaMemset(dy, 0xFF, (size_t)N * 2)); verify("fused", v, paia_gemv_int4(dW, dS, dx, dy, N, K, v, nullptr)); }
    CK(cudaMemset(dy, 0xFF, (size_t)N * 2)); verify("naive", -1, paia_gemv_int4_naive(dW, dS, dx, dy, N, K, nullptr));
    CK(cudaMemset(dy, 0xFF, (size_t)N * 2)); verify("unfused", -1, paia_gemv_int4_unfused(dW, dS, dx, dy, dsc, N, K, nullptr));
    cudaFree(dW); cudaFree(dS); cudaFree(dx); cudaFree(dy); cudaFree(dsc); return ok;
}

static bool check_quant(bool* any) {
    bool ok = true; double w8 = 0, w4 = 0;
    for (const Shape& s : QAWK_FP8) ok &= check_fp8_shape(s, &w8);
    for (const Shape& s : REAL) if (s.K % 16 == 0) ok &= check_fp8_shape(s, &w8);
    for (const Shape& s : QAWK_I4) ok &= check_int4_shape(s, &w4);
    for (const Shape& s : REAL) if (s.K % 128 == 0) ok &= check_int4_shape(s, &w4);
    // contract checks
    void* d; CK(cudaMalloc(&d, 1 << 16));
    if (paia_gemv_fp8(d, d, d, d, 4, 24, 0, nullptr) != -2) { printf("FAIL: fp8 K%%16 contract\n"); ok = false; }
    if (paia_gemv_int4(d, d, d, d, 4, 64, 0, nullptr) != -2) { printf("FAIL: int4 K%%128 contract\n"); ok = false; }
    cudaFree(d); *any = true;
    printf("quantized worst relative error: fp8 %.2e  int4 %.2e\n%s\n", w8, w4, ok ? "QUANT CHECKS PASSED" : "QUANT CHECKS FAILED");
    return ok;
}

static void bench_quant_shape(const Shape& sh, double b0, size_t l2, bool graph) {
    const int N = sh.N, K = sh.K;
    const double by16 = 2.0 * N * K + 2.0 * K + 2.0 * N, by8 = 1.0 * N * K + 4.0 * N + 2.0 * K + 2.0 * N,
                 by4 = 0.5 * N * K + 2.0 * N * (K / 128) + 2.0 * K + 2.0 * N;
    auto gb = [&](double bytes, double us) { return bytes / (us * 1e-6) / 1e9; };
    auto pc = [&](double g) { return b0 > 0 ? std::to_string((int)(100 * g / b0)) : std::string("?"); };
    // bf16 reference: best of a fixed set of 18 S1 variants, and cuBLAS
    double t16 = 1e30, tcub = -1; int bv = 0;
    {
        Bench B(N, K, l2);
        for (int v = 0; v < paia_gemv_bf16_variant_count(); ++v) {
            int R, U, W, S, H; paia_gemv_bf16_variant_info(v, &R, &U, &W, &S, &H);
            if (!(W == 4 && S == 1 && (R == 1 || R == 2 || R == 4) && (U == 1 || U == 4 || U == 8))) continue;
            const double t = B.time_us([&](int i, cudaStream_t s) { paia_gemv_bf16(B.W[i], B.x[i], B.y[i], N, K, v, s); }, graph);
            if (t > 0 && t < t16) { t16 = t; bv = v; }
        }
#ifdef WITH_CUBLAS
        static cublasHandle_t h = nullptr; if (!h) cublasCreate(&h); cublasSetStream(h, B.st);
        const float alpha = 1.f, beta = 0.f;
        auto cl = [&](int i, cudaStream_t) { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, B.W[i], CUDA_R_16BF, K, B.x[i], CUDA_R_16BF, K, &beta, B.y[i], CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
        tcub = B.time_us(cl, graph); if (tcub < 0) tcub = B.time_us(cl, false);
#endif
    }
    (void)bv;
    printf("\n%-14s N=%-6d K=%-6d  bf16 %.1f MiB | fp8 %.1f MiB | int4 %.1f MiB\n", sh.name, N, K, by16 / 1048576.0, by8 / 1048576.0, by4 / 1048576.0);
    printf("  bf16 best-of-18   %7.1f GB/s (%s%% of B0) %9.2f us   [reference for every speedup below]\n", gb(by16, t16), pc(gb(by16, t16)).c_str(), t16);
    if (tcub > 0) printf("  bf16 cuBLAS       %7.1f GB/s (%s%% of B0) %9.2f us\n", gb(by16, tcub), pc(gb(by16, tcub)).c_str(), tcub);
    {   // FP8
        QBench B(N, K, (size_t)N * K, (size_t)N * 4, l2, false);
        double best = 1e30; int bvv = 0;
        for (int v = 0; v < paia_gemv_fp8_variant_count(); ++v) {
            const double t = B.time_us([&](int i, cudaStream_t s) { paia_gemv_fp8(B.W[i], B.S[i], B.x[i], B.y[i], N, K, v, s); }, graph);
            if (t > 0 && t < best) { best = t; bvv = v; }
        }
        int R, U, W, H; paia_gemv_fp8_variant_info(bvv, &R, &U, &W, &H);
        printf("  fp8  best         %7.1f GB/s (%s%% of B0) %9.2f us   R=%d U=%d W=%d H=%d   speedup vs bf16 %.2fx of ideal %.2fx (%d%%)\n",
               gb(by8, best), pc(gb(by8, best)).c_str(), best, R, U, W, H, t16 / best, by16 / by8, (int)(100 * (t16 / best) / (by16 / by8)));
    }
    {   // INT4
        QBench B(N, K, (size_t)N * K / 2, (size_t)N * (K / 128) * 2, l2, true);
        double best = 1e30; int bvv = 0;
        for (int v = 0; v < paia_gemv_int4_variant_count(); ++v) {
            const double t = B.time_us([&](int i, cudaStream_t s) { paia_gemv_int4(B.W[i], B.S[i], B.x[i], B.y[i], N, K, v, s); }, graph);
            if (t > 0 && t < best) { best = t; bvv = v; }
        }
        int R, U, W, H; paia_gemv_int4_variant_info(bvv, &R, &U, &W, &H);
        printf("  int4 fused best   %7.1f GB/s (%s%% of B0) %9.2f us   R=%d U=%d W=%d H=%d   speedup vs bf16 %.2fx of ideal %.2fx (%d%%)\n",
               gb(by4, best), pc(gb(by4, best)).c_str(), best, R, U, W, H, t16 / best, by16 / by4, (int)(100 * (t16 / best) / (by16 / by4)));
        const double tn = B.time_us([&](int i, cudaStream_t s) { paia_gemv_int4_naive(B.W[i], B.S[i], B.x[i], B.y[i], N, K, s); }, graph);
        if (tn > 0) printf("  int4 naive (K1)   %7.1f GB/s (%s%% of B0) %9.2f us   speedup vs bf16 %.2fx\n", gb(by4, tn), pc(gb(by4, tn)).c_str(), tn, t16 / tn);
        const double tu = B.time_us([&](int i, cudaStream_t s) { paia_gemv_int4_unfused(B.W[i], B.S[i], B.x[i], B.y[i], B.scratch, N, K, s); }, graph);
        if (tu > 0) printf("  int4 unfused      %7.1f GB/s (%s%% of B0) %9.2f us   speedup vs bf16 %.2fx   (sanity control: writes and re-reads a bf16 copy)\n", gb(by4, tu), pc(gb(by4, tu)).c_str(), tu, t16 / tu);
    }
}

int main(int argc, char** argv) {
    bool full = false, bench = false, eager = false, ignore_lock = false, quant = false, bench_quant = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--full")) full = true;
        else if (!strcmp(argv[i], "--bench")) bench = true;
        else if (!strcmp(argv[i], "--eager")) eager = true;
        else if (!strcmp(argv[i], "--quant")) quant = true;
        else if (!strcmp(argv[i], "--bench-quant")) bench_quant = true;
        else if (!strcmp(argv[i], "--ignore-lock")) ignore_lock = true;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { fprintf(stderr, "no CUDA device visible\n"); return 3; }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("%s  cc %d.%d  SMs %d  L2 %.0f MiB  variants %d\n", p.name, p.major, p.minor, p.multiProcessorCount,
           p.l2CacheSize / 1048576.0, paia_gemv_bf16_variant_count());
    const int nvar = paia_gemv_bf16_variant_count();
    if (quant) { bool any = false; return check_quant(&any) ? 0 : 1; }
    if (bench || bench_quant) {
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
        if (bench_quant) { for (const Shape& s : REAL) if (s.K % 128 == 0) bench_quant_shape(s, b0, p.l2CacheSize, !eager); return 0; }
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
