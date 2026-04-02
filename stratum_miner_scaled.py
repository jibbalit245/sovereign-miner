import cupy as cp
import numpy as np
import socket
import json
import time
import binascii
import hashlib
import struct

def dsha256(data): return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# -------------------------------------------------------------
# 1. NETWORK & MATHEMATICS: Stratum Utilities
# -------------------------------------------------------------
def build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches):
    coinb = coinb1 + extranonce1 + extranonce2 + coinb2
    cb_hash = dsha256(binascii.unhexlify(coinb))
    for branch in branches:
        cb_hash = dsha256(cb_hash + binascii.unhexlify(branch))
    return binascii.hexlify(cb_hash).decode('utf-8')

def swap_endian_words(hex_str):
    """Swap byte order within each 4-byte word of a hex string."""
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        result += "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
    return result

def difficulty_to_target(difficulty):
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return diff1_target // int(difficulty)

def target_to_gpu_words(target_int):
    target_bytes = target_int.to_bytes(32, byteorder='big')
    words = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        words[i] = struct.unpack('>I', target_bytes[i*4:i*4+4])[0]
    return words

# -------------------------------------------------------------
# 2. AUTO-SCALING GPU DETECTION
# -------------------------------------------------------------
def detect_gpu_config(vram_budget_pct=0.80):
    """Query the CUDA device and compute optimal launch parameters.
    Scales to any GPU architecture: Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell."""
    
    dev = cp.cuda.Device()
    dev_id = dev.id
    free_mem, total_mem = dev.mem_info
    
    # Device attributes via CUDA runtime
    sm_count = dev.attributes.get('MultiProcessorCount', 1)
    max_threads_per_sm = dev.attributes.get('MaxThreadsPerMultiProcessor', 2048)
    max_threads_per_block = dev.attributes.get('MaxThreadsPerBlock', 1024)
    warp_size = dev.attributes.get('WarpSize', 32)
    cc_major = dev.attributes.get('ComputeCapabilityMajor', 7)
    cc_minor = dev.attributes.get('ComputeCapabilityMinor', 0)
    
    # SHA-256 kernel uses ~300 bytes of registers + local per thread (H[8]+W[64] = 288 bytes)
    # At 80% VRAM budget, the kernel itself is register-bound not memory-bound,
    # so we scale by SM occupancy, not VRAM
    
    # Optimal threads per block: 256 is safe across all archs,
    # but we test common values against SM limits
    # Rule: TPB should be a multiple of warp_size, and allow >= 2 blocks per SM
    candidate_tpb = [t for t in [64, 128, 256, 512] if t <= max_threads_per_block]
    
    # SHA-256 double-hash is register-heavy (~40 regs/thread on modern NVCC).
    # On SM_70+: 65536 regs/SM. At 40 regs/thread -> 1638 threads/SM max.
    # Practical ceiling: clamp to max_threads_per_sm
    est_regs_per_thread = 40
    reg_limited_threads_per_sm = min(65536 // est_regs_per_thread, max_threads_per_sm)
    
    # Pick TPB that allows the most blocks per SM without exceeding register limit
    best_tpb = 256  # safe default
    best_occupancy = 0
    for tpb in candidate_tpb:
        blocks_per_sm = min(reg_limited_threads_per_sm // tpb, 32)  # CUDA max 32 blocks/SM
        active_threads = blocks_per_sm * tpb
        occupancy = active_threads / max_threads_per_sm
        if occupancy > best_occupancy:
            best_occupancy = occupancy
            best_tpb = tpb
    
    # Total threads per kernel launch: fill all SMs, scale by occupancy
    # Each thread does one nonce — no VRAM needed beyond kernel args (trivial)
    # Scale up to saturate the GPU: target ~4-8x SM oversubscription for latency hiding
    saturation_factor = 8
    total_threads_per_launch = sm_count * reg_limited_threads_per_sm * saturation_factor
    
    # Round to multiple of TPB
    total_threads_per_launch = ((total_threads_per_launch + best_tpb - 1) // best_tpb) * best_tpb
    
    # Cap: don't exceed 4B (uint32 nonce space per extranonce2)
    total_threads_per_launch = min(total_threads_per_launch, 4_000_000_000)
    
    # VRAM sanity check: kernel args are tiny (~100 bytes), 
    # but CuPy itself uses some overhead. Ensure 80% budget isn't blown
    cupy_overhead = int(total_mem * 0.05)  # ~5% for CuPy runtime
    usable_vram = int(total_mem * vram_budget_pct) - cupy_overhead
    
    blocks_per_grid = (total_threads_per_launch + best_tpb - 1) // best_tpb
    
    config = {
        'device_id': dev_id,
        'compute_capability': f'{cc_major}.{cc_minor}',
        'sm_count': sm_count,
        'warp_size': warp_size,
        'max_threads_per_sm': max_threads_per_sm,
        'threads_per_block': best_tpb,
        'total_threads': total_threads_per_launch,
        'blocks_per_grid': blocks_per_grid,
        'est_occupancy': best_occupancy,
        'free_vram_gb': free_mem / 1e9,
        'total_vram_gb': total_mem / 1e9,
        'vram_budget_gb': usable_vram / 1e9,
    }
    return config

# -------------------------------------------------------------
# 3. CUDA KERNEL (Double SHA-256 — Stratum Compatible)
# -------------------------------------------------------------
cuda_source = r"""
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

__constant__ uint32_t H_INIT[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
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

extern "C" __global__ void run_stratum_miner(
    uint32_t base_nonce,
    const uint32_t *header_words,
    uint32_t *output_flag,
    uint32_t *output_nonce,
    const uint32_t *target_words
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = base_nonce + tid;

    if (*output_flag == 1) return;

    uint32_t H[8];
    uint32_t W[64];

    // --- Block 1: first 64 bytes of 80-byte header ---
    for(int i=0; i<8; i++) H[i] = H_INIT[i];
    for(int i=0; i<16; i++) W[i] = header_words[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    // --- Block 2: last 16 bytes + padding (nonce at word 3) ---
    for(int i=0; i<4; i++) W[i] = header_words[16+i];
    W[3] = nonce;
    W[4] = 0x80000000;
    for(int i=5; i<15; i++) W[i] = 0;
    W[15] = 640;  // 80 bytes * 8 bits
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    // --- Second SHA-256 pass on the 32-byte result ---
    for(int i=0; i<8; i++) W[i] = H[i];
    W[8] = 0x80000000;
    for(int i=9; i<15; i++) W[i] = 0;
    W[15] = 256;  // 32 bytes * 8 bits
    for(int i=0; i<8; i++) H[i] = H_INIT[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    // --- Compare hash vs target (both big-endian word arrays) ---
    for (int i = 0; i < 8; i++) {
        if (H[i] < target_words[i]) {
            if (atomicCAS(output_flag, 0, 1) == 0) {
                *output_nonce = nonce;
            }
            return;
        }
        if (H[i] > target_words[i]) {
            return;
        }
    }
    if (atomicCAS(output_flag, 0, 1) == 0) {
        *output_nonce = nonce;
    }
}
"""

# -------------------------------------------------------------
# 4. INITIALIZATION
# -------------------------------------------------------------
print("==========================================================")
print(" SOVEREIGN STRATUM MINER — AUTO-SCALING CUDA EDITION")
print("==========================================================")

gpu_cfg = detect_gpu_config(vram_budget_pct=0.80)

print(f"\n[GPU] Device {gpu_cfg['device_id']} | CC {gpu_cfg['compute_capability']}")
print(f"[GPU] SMs: {gpu_cfg['sm_count']} | Warp: {gpu_cfg['warp_size']} | Max Threads/SM: {gpu_cfg['max_threads_per_sm']}")
print(f"[GPU] VRAM: {gpu_cfg['free_vram_gb']:.1f} GB free / {gpu_cfg['total_vram_gb']:.1f} GB total (budget: {gpu_cfg['vram_budget_gb']:.1f} GB)")
print(f"[GPU] Launch Config: {gpu_cfg['total_threads']:,} threads | TPB={gpu_cfg['threads_per_block']} | Blocks={gpu_cfg['blocks_per_grid']:,}")
print(f"[GPU] Estimated Occupancy: {gpu_cfg['est_occupancy']*100:.0f}%")

module = cp.RawModule(code=cuda_source)
run_stratum_miner = module.get_function("run_stratum_miner")

WALLET = "bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0.antigravity"
POOL_PRIMARY = ('btc.viabtc.com', 3333)
POOL_FALLBACK = ('solo.ckpool.org', 3333)

# -------------------------------------------------------------
# 4b. BASE GPU HASH BENCHMARK — Measure real H/s, auto-tune batch
# -------------------------------------------------------------
TARGET_BATCH_SECONDS = 0.8  # sweet spot: responsive to new jobs, low launch overhead
BENCH_ROUNDS = 5

def run_benchmark(kernel_func, cfg):
    """Fire the kernel with a dummy header to measure actual throughput.
    Returns (measured_hps, tuned_total_threads, tuned_bpg)."""
    tpb = cfg['threads_per_block']
    init_threads = cfg['total_threads']
    init_bpg = cfg['blocks_per_grid']

    # Dummy 80-byte header (all zeros — we only care about speed, not validity)
    dummy_words = cp.zeros(20, dtype=np.uint32)
    # Impossible target so kernel never early-exits on a "find"
    impossible_target = cp.zeros(8, dtype=np.uint32)  # target = 0 → no hash can match
    d_flag = cp.zeros(1, dtype=np.uint32)
    d_nonce = cp.zeros(1, dtype=np.uint32)

    print(f"\n[Bench] Warming up kernel ({init_threads:,} threads x {BENCH_ROUNDS} rounds)...")

    # Warmup: 2 throwaway launches (JIT compile + cache fill)
    for _ in range(2):
        d_flag.fill(0)
        kernel_func((init_bpg,), (tpb,),
                     (np.uint32(0), dummy_words, d_flag, d_nonce, impossible_target))
        cp.cuda.Stream.null.synchronize()

    # Timed benchmark
    times = []
    for i in range(BENCH_ROUNDS):
        d_flag.fill(0)
        t0 = time.perf_counter()
        kernel_func((init_bpg,), (tpb,),
                     (np.uint32(i * init_threads), dummy_words, d_flag, d_nonce, impossible_target))
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_sec = sorted(times)[BENCH_ROUNDS // 2]
    measured_hps = init_threads / median_sec

    # Auto-tune: scale batch so each launch ≈ TARGET_BATCH_SECONDS
    tuned_threads = int(measured_hps * TARGET_BATCH_SECONDS)
    tuned_threads = ((tuned_threads + tpb - 1) // tpb) * tpb  # round up to TPB
    tuned_threads = max(tuned_threads, tpb)                    # at least one block
    tuned_threads = min(tuned_threads, 4_000_000_000)          # uint32 nonce cap
    tuned_bpg = tuned_threads // tpb

    return measured_hps, tuned_threads, tuned_bpg, median_sec


measured_hps, TOTAL_THREADS, BPG, bench_sec = run_benchmark(run_stratum_miner, gpu_cfg)
TPB = gpu_cfg['threads_per_block']

print(f"[Bench] Median kernel time: {bench_sec*1000:.1f} ms for {gpu_cfg['total_threads']:,} threads")
print(f"[Bench] Measured hash rate:  {measured_hps/1e6:.1f} MH/s  ({measured_hps/1e9:.3f} GH/s)")
print(f"[Bench] Tuned batch:         {TOTAL_THREADS:,} threads -> ~{TARGET_BATCH_SECONDS:.1f}s per launch")
print(f"[Bench] Tuned grid:          {BPG:,} blocks x {TPB} threads")

# Difficulty projection table
print(f"\n{'Difficulty':>14}  {'Hashes/Share':>16}  {'Est. Time/Share':>18}")
print(f"{'─'*14}  {'─'*16}  {'─'*18}")
for diff in [1, 16, 256, 4096, 16384, 65536, 1_000_000]:
    target = difficulty_to_target(diff)
    hashes_per_share = (2**256) // target if target > 0 else float('inf')
    est_sec = hashes_per_share / measured_hps
    if est_sec < 60:
        time_str = f"{est_sec:.1f}s"
    elif est_sec < 3600:
        time_str = f"{est_sec/60:.1f}m"
    elif est_sec < 86400:
        time_str = f"{est_sec/3600:.1f}h"
    else:
        time_str = f"{est_sec/86400:.1f}d"
    print(f"{diff:>14,}  {hashes_per_share:>16,}  {time_str:>18}")

print(f"\n[Cascade] Benchmark complete. Starting stratum cascade...")

# -------------------------------------------------------------
# 5. STRATUM PROTOCOL
# -------------------------------------------------------------
def parse_stratum(socket_obj, request_payload=None):
    if request_payload:
        socket_obj.sendall(request_payload.encode('utf-8') + b'\n')
    buffer = ""
    while True:
        try:
            buffer += socket_obj.recv(4096).decode('utf-8')
            if '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.strip():
                    return json.loads(line), buffer
        except socket.timeout:
            return None, buffer

# -------------------------------------------------------------
# 6. MINING LOOP
# -------------------------------------------------------------
def stratum_mining_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    print("\n[Network] Connecting to stratum pool...")
    try:
        sock.connect(POOL_PRIMARY)
        print(f"[Network] Connected to {POOL_PRIMARY[0]}:{POOL_PRIMARY[1]}")
    except Exception as e:
        print(f"[Network] Primary failed ({e}), falling back to {POOL_FALLBACK[0]}...")
        sock.connect(POOL_FALLBACK)

    sub_res, b1 = parse_stratum(sock, '{"id": 1, "method": "mining.subscribe", "params": []}')
    extranonce1 = sub_res['result'][1]
    en2_sz = sub_res['result'][2]
    print(f"[Stratum] Subscribed. Extranonce1: {extranonce1} | EN2 size: {en2_sz}")

    auth_res, b2 = parse_stratum(sock, f'{{"id": 2, "method": "mining.authorize", "params": ["{WALLET}", "x"]}}')
    print(f"[Stratum] Authorized. Waiting for jobs...")

    buffer = b2
    active_job = None
    extranonce2_int = 0
    pool_difficulty = 1
    total_hashes = 0
    session_start = time.time()

    while True:
        # --- Poll for new messages (non-blocking) ---
        sock.setblocking(False)
        try:
            new_data = sock.recv(8192).decode('utf-8')
            if not new_data:
                raise ConnectionError("Server closed connection")
            buffer += new_data
        except BlockingIOError:
            pass
        except (ConnectionError, OSError) as e:
            sock.setblocking(True)
            raise e
        sock.setblocking(True)

        # --- Parse all buffered messages ---
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if not line.strip(): continue
            msg = json.loads(line)

            if msg.get('method') == 'mining.set_difficulty':
                pool_difficulty = msg['params'][0]
                print(f"\n[Stratum] Difficulty updated: {pool_difficulty}")

            if msg.get('method') == 'mining.notify':
                active_job = msg['params']
                job_id = active_job[0]
                print(f"\n[Job] New block (Job {job_id})")
                extranonce2_int = 0
                b_nonce_val = 0

            if msg.get('id') == 4:
                accepted = msg.get('result')
                err = msg.get('error')
                status = "ACCEPTED" if accepted else f"REJECTED ({err})"
                print(f"\n[Share] {status}")

        if active_job is None:
            time.sleep(0.5)
            continue

        # --- Build header ---
        job_id, prevhash, coinb1, coinb2, branches, version, nbits, ntime, clean = active_job[:9]
        extranonce2 = hex(extranonce2_int)[2:].zfill(en2_sz * 2)

        merkle_root = build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches)

        header_hex = (swap_endian_words(version)
                     + swap_endian_words(prevhash)
                     + merkle_root
                     + swap_endian_words(ntime)
                     + swap_endian_words(nbits)
                     + "00000000")
        header_bytes = binascii.unhexlify(header_hex)

        words_cpu = np.zeros(20, dtype=np.uint32)
        try:
            for i in range(20):
                words_cpu[i] = struct.unpack('>I', header_bytes[i*4:i*4+4])[0]
        except (struct.error, IndexError):
            continue

        target_int = difficulty_to_target(pool_difficulty)
        target_cpu = target_to_gpu_words(target_int)

        d_words = cp.asarray(words_cpu)
        d_target = cp.asarray(target_cpu)
        d_flag = cp.zeros(1, dtype=np.uint32)
        d_nonce = cp.zeros(1, dtype=np.uint32)

        b_nonce_val = 0
        print(f"[Mining] EN2={extranonce2} | Diff={pool_difficulty} | {TOTAL_THREADS:,} threads/batch")

        while b_nonce_val < 4_200_000_000:
            b_nonce = np.uint32(b_nonce_val)
            d_flag.fill(0)
            d_nonce.fill(0)

            t0 = time.time()
            run_stratum_miner((BPG,), (TPB,), (b_nonce, d_words, d_flag, d_nonce, d_target))
            cp.cuda.Stream.null.synchronize()
            t1 = time.time()

            if d_flag.get()[0] == 1:
                win_nonce_hex = hex(d_nonce.get()[0])[2:].zfill(8)
                win_nonce_swap = "".join(reversed([win_nonce_hex[i:i+2] for i in range(0, 8, 2)]))
                print(f"\n$$$ SHARE FOUND @ Nonce: {win_nonce_swap} $$$")
                submit = {
                    "params": [WALLET, job_id, extranonce2, ntime, win_nonce_swap],
                    "id": 4,
                    "method": "mining.submit"
                }
                try:
                    sock.sendall(json.dumps(submit).encode('utf-8') + b'\n')
                    print(f"[Network] Share submitted to pool")
                except (BrokenPipeError, OSError) as e:
                    print(f"[Network] Submit failed: {e}")

            b_nonce_val += TOTAL_THREADS
            total_hashes += TOTAL_THREADS
            elapsed = max(t1 - t0, 0.001)
            session_elapsed = max(time.time() - session_start, 1)
            instant_rate = TOTAL_THREADS / elapsed
            avg_rate = total_hashes / session_elapsed
            drift_pct = ((instant_rate - measured_hps) / measured_hps) * 100
            print(f"[GPU] {instant_rate/1e6:>7.1f} MH/s (avg {avg_rate/1e6:.1f} | base {measured_hps/1e6:.1f}) [{drift_pct:+.1f}%] | {b_nonce_val:>12,} nonces", end='\r')

            # Check for new jobs mid-scan
            sock.setblocking(False)
            interrupt = False
            try:
                new_data = sock.recv(8192).decode('utf-8')
                if 'mining.notify' in new_data:
                    buffer += new_data
                    interrupt = True
                elif new_data:
                    buffer += new_data
            except BlockingIOError:
                pass
            except (ConnectionError, OSError):
                interrupt = True
            sock.setblocking(True)

            if interrupt:
                break

        if not interrupt:
            print(f"\n[Mining] Nonce space exhausted. Rotating extranonce2.")
            extranonce2_int += 1

# -------------------------------------------------------------
# 7. DAEMON LOOP WITH AUTO-RECONNECT
# -------------------------------------------------------------
while True:
    try:
        stratum_mining_loop()
    except KeyboardInterrupt:
        session_elapsed = max(time.time() - time.time(), 1)
        print(f"\n[Shutdown] Terminated by user.")
        break
    except Exception as e:
        print(f"\n[Reconnect] Connection lost: {e}")
        time.sleep(3)
