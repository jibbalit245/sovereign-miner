"""
Sovereign Cascade Engine — Optimized GPU Implementation
========================================================
Fused CUDA kernel executing the full 28-op SHA-256 subfunction
pipeline per cell, BFS level-by-level across the binary tree.

Full 3-input mixing (closed loop):
  - Sigma0 operates on x (working var 'a')
  - Sigma1 operates on y (working var 'e')  
  - sigma0 operates on z (message schedule)
  - sigma1 operates on x^y (cross-entangled)
  
Fan-out cross-entangles all 6 outputs into children.
"""
import cupy as cp
import numpy as np
import time

# =================================================================
# CUDA KERNEL
# =================================================================
_CASCADE_KERNEL = cp.RawKernel(r'''
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
    
    // Layer 1: SHA-256 primitives with full 3-input distribution
    unsigned int ch_a1 = x & y;
    unsigned int not_x = ~x;
    unsigned int maj_a2 = x & z;
    unsigned int maj_a3 = y & z;
    
    unsigned int s0_r2  = (x >> 2)  | (x << 30);
    unsigned int s0_r13 = (x >> 13) | (x << 19);
    unsigned int s0_r22 = (x >> 22) | (x << 10);
    unsigned int s1_r6  = (y >> 6)  | (y << 26);
    unsigned int s1_r11 = (y >> 11) | (y << 21);
    unsigned int s1_r25 = (y >> 25) | (y << 7);
    unsigned int g0_r7  = (z >> 7)  | (z << 25);
    unsigned int g0_r18 = (z >> 18) | (z << 14);
    unsigned int g0_s3  = z >> 3;
    unsigned int xy = x ^ y;
    unsigned int g1_r17 = (xy >> 17) | (xy << 15);
    unsigned int g1_r19 = (xy >> 19) | (xy << 13);
    unsigned int g1_s10 = xy >> 10;
    
    // Layer 2: XOR aggregation
    unsigned int ch_a2 = not_x & z;
    unsigned int maj_x1 = ch_a1 ^ maj_a2;  // reuse ch_a1 == x&y == maj_a1
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

TPB = 64  # Optimal threads-per-block from tuning sweep

# =================================================================
# ENGINE
# =================================================================
def sovereign_cascade(x0, y0, z0, depth):
    """Execute the full sovereign cascade on GPU. Returns (total_cells, leaf_xs, leaf_ys, leaf_zs)."""
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
        
        blocks = (n + TPB - 1) // TPB
        _CASCADE_KERNEL((blocks,), (TPB,), (xs, ys, zs, xs_out, ys_out, zs_out, np.uint32(n)))
        xs, ys, zs = xs_out, ys_out, zs_out
    
    cp.cuda.Stream.null.synchronize()
    return total, xs, ys, zs


# =================================================================
# BENCHMARK
# =================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("  SOVEREIGN CASCADE ENGINE — GPU BENCHMARK")
    print("==========================================================")
    
    x0 = np.uint32(0x6a09e667)
    y0 = np.uint32(0xbb67ae85)
    z0 = np.uint32(0x3c6ef372)
    
    # Warmup / compile
    print("\n[Compiling CUDA kernel...]")
    sovereign_cascade(x0, y0, z0, 10)
    print("[Ready]\n")
    
    print(f"{'Depth':>6} | {'Cells':>14} | {'Time':>8} | {'Cells/sec':>16} | {'Ops/sec (28/cell)':>20}")
    print("-" * 80)
    
    for depth in [10, 16, 20, 24, 26, 27, 28]:
        times = []
        got = 0
        for _ in range(3):
            t0 = time.perf_counter()
            got, lx, ly, lz = sovereign_cascade(x0, y0, z0, depth)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        best = min(times)
        cells_s = got / best
        ops_s = cells_s * 28
        leaf_mem_gb = (2**depth * 4 * 3) / (1024**3)
        print(f"{depth:>6} | {got:>14,} | {best:>7.4f}s | {cells_s:>16,.0f} | {ops_s:>20,.0f}")
    
    print("\n[Leaf sample from depth 28]")
    _, lx, ly, lz = sovereign_cascade(x0, y0, z0, 28)
    print(f"  Leaf[0]:        x=0x{int(lx[0]):08X}  y=0x{int(ly[0]):08X}  z=0x{int(lz[0]):08X}")
    print(f"  Leaf[last]:     x=0x{int(lx[-1]):08X}  y=0x{int(ly[-1]):08X}  z=0x{int(lz[-1]):08X}")
    print(f"  Unique x vals:  {int(cp.unique(lx).size):,} / {int(lx.size):,}")
