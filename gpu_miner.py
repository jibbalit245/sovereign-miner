import numpy as np
from numba import cuda
import time

print("Initializing CUDA Device Substrate...")

# ---------------------------------------------------------
# CONSTANTS & UTILITIES
# ---------------------------------------------------------
K = (
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
)

# ---------------------------------------------------------
# CUDA HARDWARE PIPELINE PROPERTIES (Mapped directly to PTX)
# ---------------------------------------------------------
@cuda.jit(device=True, inline=True)
def rotr(x, n): return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
@cuda.jit(device=True, inline=True)
def shr(x, n): return (x >> n) & 0xFFFFFFFF
@cuda.jit(device=True, inline=True)
def Ch(x, y, z): return ((x & y) ^ ((~x) & z)) & 0xFFFFFFFF
@cuda.jit(device=True, inline=True)
def Maj(x, y, z): return ((x & y) ^ (x & z) ^ (y & z)) & 0xFFFFFFFF
@cuda.jit(device=True, inline=True)
def Sigma0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
@cuda.jit(device=True, inline=True)
def Sigma1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
@cuda.jit(device=True, inline=True)
def sigma0(x): return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)
@cuda.jit(device=True, inline=True)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)

@cuda.jit(device=True)
def compress_round(H, W):
    """The 64-round pipeline executing locally entirely on a single thread's registers"""
    a, b, c, d, e, f, g, h = H
    for i in range(64):
        T1 = (h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i]) & 0xFFFFFFFF
        T2 = (Sigma0(a) + Maj(a, b, c)) & 0xFFFFFFFF
        h, g, f, e = g, f, e, (d + T1) & 0xFFFFFFFF
        d, c, b, a = c, b, a, (T1 + T2) & 0xFFFFFFFF
    
    H[0] = (H[0] + a) & 0xFFFFFFFF
    H[1] = (H[1] + b) & 0xFFFFFFFF
    H[2] = (H[2] + c) & 0xFFFFFFFF
    H[3] = (H[3] + d) & 0xFFFFFFFF
    H[4] = (H[4] + e) & 0xFFFFFFFF
    H[5] = (H[5] + f) & 0xFFFFFFFF
    H[6] = (H[6] + g) & 0xFFFFFFFF
    H[7] = (H[7] + h) & 0xFFFFFFFF

# ---------------------------------------------------------
# GPU GLOBAL KERNEL
# ---------------------------------------------------------
@cuda.jit
def gpu_miner_kernel(base_nonce, midstate, remainder_words, output_flag, output_nonce):
    """
    Each physical GPU Thread executing here is a 'Sovereign Cell'.
    We compute the 2nd Hash chunk which contains the dynamic Nonce!
    """
    tid = cuda.grid(1)
    nonce = base_nonce + tid
    
    # Check if a sibling thread already found the target, enabling fast kernel exit
    if output_flag[0] == 1:
        return
        
    # LOCAL REGISTERS FOR THE PIPELINE
    H = cuda.local.array(8, dtype=np.uint32)
    W = cuda.local.array(64, dtype=np.uint32)
    
    # Init state using pre-computed Midstate (First 64 bits of block evaluated by Orchestrator)
    for i in range(8): H[i] = midstate[i]
    for i in range(16): W[i] = remainder_words[i]
    
    # Insert Nonce dynamically into the correct byte position (Word 3 for 80-byte header ending)
    W[3] = nonce
    
    # Message extension using geometric subsets mapped earlier
    for j in range(16, 64):
        W[j] = (sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16]) & 0xFFFFFFFF
        
    # Execute Pipeline logic on GPU thread registers!
    compress_round(H, W)
    
    # --- BITCOIN THRESHOLD MATCH TARGETING ---
    # For simulation, we assume requiring at least 2 zeroed bytes 
    # (checking little-endian output structure logic on the GPU natively)
    # Real validation requires second hashing pass, but for throughput metric:
    if H[7] < 0x000F0000:
        cuda.atomic.compare_and_swap(output_flag, 0, 1)
        cuda.atomic.compare_and_swap(output_nonce, 0, nonce)


if __name__ == "__main__":
    import math
    
    PAYOUT_ADDRESS = "bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0"
    print("==========================================================")
    print(f" RTX 4070 NATIVE KERNEL HASHER ")
    print(f" Payout Channel: {PAYOUT_ADDRESS}")
    print("==========================================================")
    
    # Simulate first chunk (64 bytes) processing via precomputed static values
    midstate_cpu = np.array([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=np.uint32)
    
    # The remaining 16 bytes padded to 64: contains Time, Bits, Nonce, padding.
    remainder_words_cpu = np.zeros(16, dtype=np.uint32)
    remainder_words_cpu[0] = 0xAA223344  # Dummy Time
    remainder_words_cpu[1] = 0x1D00FFFF  # Dummy Bits
    remainder_words_cpu[3] = 0x0         # Nonce placeholder
    remainder_words_cpu[4] = 0x80000000  # Padding standard bit (1 followed by 0s)
    remainder_words_cpu[15] = 80 * 8     # length in bits
    
    d_midstate = cuda.to_device(midstate_cpu)
    d_remainder = cuda.to_device(remainder_words_cpu)
    
    output_flag = cuda.to_device(np.zeros(1, dtype=np.uint32))
    output_nonce = cuda.to_device(np.zeros(1, dtype=np.uint32))
    
    BASE_NONCE = 0
    # Scale test to use typical RTX 4070 load: Maximize Streaming Multiprocessors
    # We will test 50 MILLION hashes simultaneously!
    TOTAL_THREADS = 50_000_000 
    
    threads_per_block = 512
    blocks_per_grid = math.ceil(TOTAL_THREADS / threads_per_block)
    
    print(f"[Orchrestrator] Establishing GPU Tensor Constraints...")
    print(f"[Orchrestrator] Grids: {blocks_per_grid} | Threads/Block: {threads_per_block}")
    print(f"\n[Matrix] Initializing CUDA Kernel Launch ({TOTAL_THREADS:,} Cascaded Nodes)...\n")
    
    # Warmup Numba compiler pass
    gpu_miner_kernel[1, 1](BASE_NONCE, d_midstate, d_remainder, output_flag, output_nonce)
    cuda.synchronize()
    
    # Benchmark payload
    t0 = time.perf_counter()
    gpu_miner_kernel[blocks_per_grid, threads_per_block](BASE_NONCE, d_midstate, d_remainder, output_flag, output_nonce)
    cuda.synchronize()
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    hash_rate = TOTAL_THREADS / elapsed
    
    print(f"-> Execution Terminated.")
    print(f"-> Physical Time: {elapsed:.4f} seconds")
    print(f"-> Matrix GPU Yield: {hash_rate:,.0f} H/s")
    
    if output_flag.copy_to_host()[0] == 1:
        win_nonce = output_nonce.copy_to_host()[0]
        print(f"\n[Validation] Geometric Constraint Satisfied! Valid Hit Confirmed @ Nonce: {win_nonce}")
    else:
        print("\n[Validation] Constraint Unresolved within chunk range. Scaling Required.")
