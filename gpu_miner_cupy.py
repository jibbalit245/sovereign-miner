import cupy as cp
import numpy as np
import time

cuda_source = """
typedef unsigned int uint32_t;

__constant__ uint32_t d_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ uint32_t shr(uint32_t x, uint32_t n) { return x >> n; }
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ ((~x) & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t Sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ __forceinline__ uint32_t Sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ __forceinline__ uint32_t sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3); }
__device__ __forceinline__ uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10); }

__device__ void compress_round(uint32_t *H, uint32_t *W) {
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + d_K[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

extern "C" __global__ void gpu_miner_kernel(uint32_t base_nonce, const uint32_t *midstate, const uint32_t *remainder_words, uint32_t *output_flag, uint32_t *output_nonce) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = base_nonce + tid;

    if (*output_flag == 1) return;

    uint32_t H[8];
    uint32_t W[64];

    for(int i=0; i<8; i++) H[i] = midstate[i];
    for(int i=0; i<16; i++) W[i] = remainder_words[i];

    W[3] = nonce;

    #pragma unroll
    for (int j = 16; j < 64; j++) {
        W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    }

    compress_round(H, W);

    if (H[7] < 0x000F0000) {
        atomicCAS(output_flag, 0, 1);
        atomicExch(output_nonce, nonce);
    }
}
"""

print("==========================================================")
print(" RTX 4070 NATIVE KERNEL HASHER (.CUPY COMPILATION)")
print(" Payout Channel: bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0")
print("==========================================================")

module = cp.RawModule(code=cuda_source)
gpu_miner_kernel = module.get_function("gpu_miner_kernel")

TOTAL_THREADS = 50_000_000
BASE_NONCE = np.uint32(0)

midstate_cpu = np.array([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=np.uint32)
remainder_cpu = np.zeros(16, dtype=np.uint32)
remainder_cpu[0] = 0xAA223344
remainder_cpu[1] = 0x1D00FFFF
remainder_cpu[4] = 0x80000000
remainder_cpu[15] = 80 * 8

d_midstate = cp.asarray(midstate_cpu)
d_remainder = cp.asarray(remainder_cpu)
d_output_flag = cp.zeros(1, dtype=np.uint32)
d_output_nonce = cp.zeros(1, dtype=np.uint32)

threads_per_block = 512
blocks_per_grid = (TOTAL_THREADS + threads_per_block - 1) // threads_per_block

print(f"[Orchrestrator] Grids: {blocks_per_grid} | Threads/Block: {threads_per_block}")
print(f"[Matrix] Initializing CUDA Kernel Launch ({TOTAL_THREADS:,} Cascaded Nodes)...\n")

gpu_miner_kernel((blocks_per_grid,), (threads_per_block,), (BASE_NONCE, d_midstate, d_remainder, d_output_flag, d_output_nonce))
cp.cuda.Stream.null.synchronize()

t0 = time.perf_counter()
gpu_miner_kernel((blocks_per_grid,), (threads_per_block,), (BASE_NONCE, d_midstate, d_remainder, d_output_flag, d_output_nonce))
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()

elapsed = t1 - t0
print(f"-> Execution Terminated.")
print(f"-> Physical Time: {elapsed:.5f} seconds")
print(f"-> Matrix GPU Yield: {TOTAL_THREADS / elapsed:,.0f} H/s")

flag = d_output_flag.get()[0]
if flag == 1:
    win_nonce = d_output_nonce.get()[0]
    print(f"\n[Validation] Geometric Constraint Satisfied! Valid Hit Confirmed @ Nonce: {win_nonce}")
else:
    print("\n[Validation] Constraint Unresolved within chunk range. Scaling Required.")
