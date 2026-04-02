import cupy as cp
import numpy as np
import time

# =================================================================
# FUSED CUDA KERNEL: Single kernel per cascade level
# Each thread processes one cell: reads 3 inputs, computes full
# 28-op SHA-256 subfunction pipeline, writes 6 outputs (2 children)
# =================================================================
fused_kernel = cp.RawKernel(r'''
extern "C" __global__ void cascade_level(
    const unsigned int* __restrict__ xs_in,
    const unsigned int* __restrict__ ys_in,
    const unsigned int* __restrict__ zs_in,
    unsigned int* __restrict__ xs_out,
    unsigned int* __restrict__ ys_out,
    unsigned int* __restrict__ zs_out,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    unsigned int x = xs_in[tid];
    unsigned int y = ys_in[tid];
    unsigned int z = zs_in[tid];
    
    // Layer 1: All SHA-256 primitives with full 3-input mixing
    unsigned int ch_a1 = x & y;
    unsigned int not_x = ~x;
    unsigned int maj_a1 = x & y;
    unsigned int maj_a2 = x & z;
    unsigned int maj_a3 = y & z;
    
    // Sigma0(x) — working variable 'a'
    unsigned int s0_r2  = (x >> 2)  | (x << 30);
    unsigned int s0_r13 = (x >> 13) | (x << 19);
    unsigned int s0_r22 = (x >> 22) | (x << 10);
    // Sigma1(y) — working variable 'e' [CLOSED LOOP]
    unsigned int s1_r6  = (y >> 6)  | (y << 26);
    unsigned int s1_r11 = (y >> 11) | (y << 21);
    unsigned int s1_r25 = (y >> 25) | (y << 7);
    // sigma0(z) — message schedule mixing
    unsigned int g0_r7  = (z >> 7)  | (z << 25);
    unsigned int g0_r18 = (z >> 18) | (z << 14);
    unsigned int g0_s3  = z >> 3;
    // sigma1(x^y) — cross-entangled message schedule
    unsigned int xy = x ^ y;
    unsigned int g1_r17 = (xy >> 17) | (xy << 15);
    unsigned int g1_r19 = (xy >> 19) | (xy << 13);
    unsigned int g1_s10 = xy >> 10;
    
    // Layer 2: XOR aggregation
    unsigned int ch_a2 = not_x & z;
    unsigned int maj_x1 = maj_a1 ^ maj_a2;
    unsigned int s0_x1 = s0_r2 ^ s0_r13;
    unsigned int s1_x1 = s1_r6 ^ s1_r11;
    unsigned int g0_x1 = g0_r7 ^ g0_r18;
    unsigned int g1_x1 = g1_r17 ^ g1_r19;
    
    // Layer 3: Output collapse
    unsigned int ch    = ch_a1 ^ ch_a2;
    unsigned int maj   = maj_x1 ^ maj_a3;
    unsigned int sig0  = s0_x1 ^ s0_r22;
    unsigned int sig1  = s1_x1 ^ s1_r25;
    unsigned int lsig0 = g0_x1 ^ g0_s3;
    unsigned int lsig1 = g1_x1 ^ g1_s10;
    
    // Fan-out with cross-entanglement
    unsigned int left  = tid * 2;
    unsigned int right = tid * 2 + 1;
    
    xs_out[left]  = ch ^ lsig1;    xs_out[right] = sig1;
    ys_out[left]  = maj;            ys_out[right] = lsig0;
    zs_out[left]  = sig0;           zs_out[right] = ch ^ maj;
}
''', 'cascade_level')


def cascade_fused(x0, y0, z0, depth):
    """GPU-fused BFS cascade: one kernel launch per level."""
    xs = cp.array([x0], dtype=cp.uint32)
    ys = cp.array([y0], dtype=cp.uint32)
    zs = cp.array([z0], dtype=cp.uint32)
    
    total = 0
    for level in range(depth + 1):
        n = len(xs)
        total += n
        if level == depth:
            break
        xs_out = cp.empty(n * 2, dtype=cp.uint32)
        ys_out = cp.empty(n * 2, dtype=cp.uint32)
        zs_out = cp.empty(n * 2, dtype=cp.uint32)
        
        threads = 256
        blocks = (n + threads - 1) // threads
        fused_kernel((blocks,), (threads,), (xs, ys, zs, xs_out, ys_out, zs_out, np.uint32(n)))
        xs, ys, zs = xs_out, ys_out, zs_out
    
    cp.cuda.Stream.null.synchronize()
    return total


if __name__ == "__main__":
    print("==========================================================")
    print(" FUSED CUDA CASCADE BENCHMARK")
    print("==========================================================")

    # Warmup (triggers NVRTC compilation)
    print("\n[Compiling CUDA kernel...]")
    cascade_fused(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372), 10)
    print("[Ready]\n")

    for depth in [10, 16, 20, 24, 25]:
        expected = (2 ** (depth + 1)) - 1
        t0 = time.perf_counter()
        got = cascade_fused(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372), depth)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        cells_s = got / elapsed
        ops_s = got * 28 / elapsed
        print(f"  Depth {depth:>2}: {got:>12,} cells | {elapsed:.4f}s | {cells_s:>14,.0f} cells/s | {ops_s:>16,.0f} ops/s")
