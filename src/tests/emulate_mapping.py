"""CPU check of the kernel's index logic (no GPU needed): replays the exact control flow of gemv_bf16_kernel
(row ownership, chunk batches, power-of-two remainder, partial chunk, row clamp/mask) in Python and asserts that every
(row, column) of W is read exactly once and every row is stored exactly once, for many (N, K, R, U, W).
It does not test CUDA semantics (that is test_gemv_bf16.py on a GPU); it catches coverage/indexing mistakes."""
import itertools
import random
from collections import Counter


def replay(N, K, R, U, W):
    reads, stores = Counter(), Counter()
    nblocks = (N + R * W - 1) // (R * W)
    for b in range(nblocks):
        for warp in range(W):
            row0 = (b * W + warp) * R
            if row0 >= N:
                continue
            rows = [min(row0 + r, N - 1) for r in range(R)]
            for lane in range(32):
                def batch(c0, UC):
                    for u in range(UC):
                        col = ((c0 + u) << 8) + lane * 8
                        assert col + 8 <= K, "full-chunk load out of bounds"
                        for r in range(R):
                            if row0 + r < N:                       # clamped duplicate rows are not counted twice
                                reads.update((rows[r], col + j) for j in range(8))
                nfull = K >> 8
                c0 = 0
                while c0 + U <= nfull:
                    batch(c0, U); c0 += U
                if U > 4 and nfull - c0 >= 4: batch(c0, 4); c0 += 4
                if U > 2 and nfull - c0 >= 2: batch(c0, 2); c0 += 2
                if U > 1 and nfull - c0 >= 1: batch(c0, 1); c0 += 1
                assert c0 == nfull, (N, K, R, U, W, c0, nfull)
                if (nfull << 8) < K:
                    col = (nfull << 8) + lane * 8
                    if col < K:
                        assert col + 8 <= K
                        for r in range(R):
                            if row0 + r < N:
                                reads.update((rows[r], col + j) for j in range(8))
                if lane == 0:
                    for r in range(R):
                        if row0 + r < N:
                            stores[row0 + r] += 1
    return reads, stores


def main():
    rnd = random.Random(0)
    cases = []
    for _ in range(60):
        cases.append((rnd.randint(1, 70), rnd.choice([8, 16, 248, 256, 264, 512, 520, 1024, 1032, 2048 + 8, 3584, 4096 - 8]),))
    n_checked = 0
    for (N, K), R, U, W in itertools.product(cases[:40], (1, 2, 4), (1, 2, 4, 8), (4, 8)):
        reads, stores = replay(N, K, R, U, W)
        assert len(reads) == N * K and all(v == 1 for v in reads.values()), (N, K, R, U, W, "coverage")
        assert len(stores) == N and all(v == 1 for v in stores.values()), (N, K, R, U, W, "stores")
        n_checked += 1
    # the real shapes' K values, small N so it stays fast
    for K in (4096, 14336, 3584, 18944):
        for R, U, W in itertools.product((1, 2, 4), (1, 2, 4, 8), (4, 8)):
            reads, stores = replay(37, K, R, U, W)
            assert len(reads) == 37 * K and all(v == 1 for v in reads.values()), (K, R, U, W)
            n_checked += 1
    print(f"mapping OK: {n_checked} (shape, R, U, W) combinations covered each element exactly once")


if __name__ == "__main__":
    main()
