import cupy as cp
import numpy as np
import time

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
    
    unsigned int ch_a1 = x & y;
    unsigned int not_x = ~x;
    unsigned int maj_a1 = x & y;
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
    
    unsigned int ch_a2 = not_x & z;
    unsigned int maj_x1 = maj_a1 ^ maj_a2;
    unsigned int s0_x1 = s0_r2 ^ s0_r13;
    unsigned int s1_x1 = s1_r6 ^ s1_r11;
    unsigned int g0_x1 = g0_r7 ^ g0_r18;
    unsigned int g1_x1 = g1_r17 ^ g1_r19;
    
    unsigned int ch    = ch_a1 ^ ch_a2;
    unsigned int maj   = maj_x1 ^ maj_a3;
    unsigned int sig0  = s0_x1 ^ s0_r22;
    unsigned int sig1  = s1_x1 ^ s1_r25;
    unsigned int lsig0 = g0_x1 ^ g0_s3;
    unsigned int lsig1 = g1_x1 ^ g1_s10;
    
    unsigned int left  = tid * 2;
    unsigned int right = tid * 2 + 1;
    
    xs_out[left]  = ch ^ lsig1;    xs_out[right] = sig1;
    ys_out[left]  = maj;            ys_out[right] = lsig0;
    zs_out[left]  = sig0;           zs_out[right] = ch ^ maj;
}
''', 'cascade_level')


def cascade_safe(x0, y0, z0, depth, tpb=256):
    """Memory-safe cascade: pre-allocates max buffers, ping-pongs, no intermediate allocs."""
    max_level_size = 2 ** depth
    
    # Check VRAM budget — 12 bytes per cell, need 2 sets (ping-pong)
    vram_needed = max_level_size * 12 * 2
    free_mem = cp.cuda.Device().mem_info[0]
    
    if vram_needed > free_mem * 0.7:  # Stay under 70% of free VRAM
        max_safe_depth = int(np.log2(free_mem * 0.7 / 24))
        raise MemoryError(
            f"Depth {depth} needs ~{vram_needed/1e9:.1f} GB but only "
            f"~{free_mem/1e9:.1f} GB free. Max safe depth: {max_safe_depth}"
        )
    
    # Pre-allocate ping-pong buffers once
    buf_a = (
        cp.empty(max_level_size, dtype=cp.uint32),
        cp.empty(max_level_size, dtype=cp.uint32),
        cp.empty(max_level_size, dtype=cp.uint32),
    )
    buf_b = (
        cp.empty(max_level_size, dtype=cp.uint32),
        cp.empty(max_level_size, dtype=cp.uint32),
        cp.empty(max_level_size, dtype=cp.uint32),
    )
    
    # Seed
    buf_a[0][0] = x0
    buf_a[1][0] = y0
    buf_a[2][0] = z0
    
    src, dst = buf_a, buf_b
    total = 0
    n = 1
    
    for level in range(depth):
        total += n
        blocks = (n + tpb - 1) // tpb
        fused_kernel(
            (blocks,), (tpb,),
            (src[0], src[1], src[2], dst[0], dst[1], dst[2], np.uint32(n))
        )
        n *= 2
        src, dst = dst, src
    
    total += n  # leaf level
    cp.cuda.Stream.null.synchronize()
    return total


if __name__ == "__main__":
    print("=" * 60)
    print(" SOVEREIGN CASCADE GEOMETRY — GPU OPTIMIZATION SUITE")
    print("=" * 60)
    
    free_mem = cp.cuda.Device().mem_info[0]
    total_mem = cp.cuda.Device().mem_info[1]
    dev_id = cp.cuda.Device().id
    print(f"GPU Device {dev_id} | VRAM Free: {free_mem/1e9:.1f} GB / {total_mem/1e9:.1f} GB")
    max_safe = int(np.log2(free_mem * 0.7 / 24))
    print(f"Max safe depth: {max_safe} ({2**(max_safe+1)-1:,} cells)\n")
    
    # Warmup
    cascade_safe(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372), 10)
    
    # --- Phase 1: Find optimal thread block size at safe medium depth ---
    test_depth = min(20, max_safe)
    print(f"Phase 1: Thread block tuning (depth {test_depth}, {2**(test_depth+1)-1:,} cells)")
    print("-" * 60)
    
    best_rate = 0
    best_tpb = 256
    for tpb in [64, 128, 256, 512, 1024]:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            got = cascade_safe(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372), test_depth, tpb)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        best_t = min(times)
        rate = got / best_t
        marker = ""
        if rate > best_rate:
            best_rate = rate
            best_tpb = tpb
            marker = " <-- BEST"
        print(f"  TPB={tpb:>4}: {best_t:.4f}s | {rate:>14,.0f} cells/s | {rate*28:>16,.0f} ops/s{marker}")
    
    print(f"\n  Winner: TPB={best_tpb}\n")
    
    # --- Phase 2: Scale depth with the winner TPB ---
    print(f"Phase 2: Depth scaling (TPB={best_tpb})")
    print("-" * 60)
    
    for d in range(10, max_safe + 1, 2):
        total_cells = 2**(d+1) - 1
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            got = cascade_safe(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372), d, best_tpb)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        best_t = min(times)
        rate = got / best_t
        print(f"  D={d:>2} | {total_cells:>14,} cells | {best_t:.4f}s | {rate:>14,.0f} cells/s | {rate*28:>16,.0f} ops/s")
    
    print("\nDone.")
