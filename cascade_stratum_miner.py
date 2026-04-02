"""
Sovereign Cascade-Amplified Stratum Miner
==========================================
Architecture:
  1. Midstate: SHA-256 compress Block 1 (header[0:64]) — same for all nonces
  2. Cascade:  Seed with midstate working vars, fan out to 2^D leaf cells
  3. Verify:   Each leaf x-value = candidate nonce → finish Block 2 + 2nd pass
  4. Submit:   Any hash < target → share to pool

The cascade uses SHA-256's own subfunctions (Ch, Maj, Σ0, Σ1, σ0, σ1) to
select which nonces to test, seeded from the block's own cryptographic state.
"""
import cupy as cp
import numpy as np
import socket
import json
import time
import binascii
import hashlib
import struct
import math

def dsha256(data): return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# -------------------------------------------------------------
# 1. STRATUM UTILITIES
# -------------------------------------------------------------
def build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches):
    coinb = coinb1 + extranonce1 + extranonce2 + coinb2
    cb_hash = dsha256(binascii.unhexlify(coinb))
    for branch in branches:
        cb_hash = dsha256(cb_hash + binascii.unhexlify(branch))
    return binascii.hexlify(cb_hash).decode('utf-8')

def swap_endian_words(hex_str):
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
# 2. GPU AUTO-DETECTION
# -------------------------------------------------------------
def detect_gpu_config(vram_budget_pct=0.80):
    dev = cp.cuda.Device()
    free_mem, total_mem = dev.mem_info
    sm_count = dev.attributes.get('MultiProcessorCount', 1)
    max_threads_per_sm = dev.attributes.get('MaxThreadsPerMultiProcessor', 2048)
    max_threads_per_block = dev.attributes.get('MaxThreadsPerBlock', 1024)
    warp_size = dev.attributes.get('WarpSize', 32)
    cc_major = dev.attributes.get('ComputeCapabilityMajor', 7)
    cc_minor = dev.attributes.get('ComputeCapabilityMinor', 0)

    # Cascade depth: 2^D leaves × 12 bytes × 2 (ping-pong) must fit in VRAM budget
    usable_vram = int(free_mem * vram_budget_pct)
    max_cascade_depth = int(math.log2(usable_vram / 24)) if usable_vram > 24 else 10
    # Cap at 30 (1B leaves) for sanity; verify kernel batch size
    max_cascade_depth = min(max_cascade_depth, 250)

    # Verify kernel TPB (heavier register usage than cascade)
    candidate_tpb = [t for t in [64, 128, 256, 512] if t <= max_threads_per_block]
    est_regs = 40
    reg_limited = min(65536 // est_regs, max_threads_per_sm)
    best_tpb = 256
    best_occ = 0
    for tpb in candidate_tpb:
        bps = min(reg_limited // tpb, 32)
        occ = (bps * tpb) / max_threads_per_sm
        if occ > best_occ:
            best_occ = occ
            best_tpb = tpb

    return {
        'compute_capability': f'{cc_major}.{cc_minor}',
        'sm_count': sm_count,
        'free_vram_gb': free_mem / 1e9,
        'total_vram_gb': total_mem / 1e9,
        'usable_vram_gb': usable_vram / 1e9,
        'max_cascade_depth': max_cascade_depth,
        'verify_tpb': best_tpb,
        'cascade_tpb': 64,  # optimal from tuning
    }

# -------------------------------------------------------------
# 3. CUDA KERNELS
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
__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ ((~x) & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t Sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ __forceinline__ uint32_t Sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ __forceinline__ uint32_t sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

__device__ void compress(uint32_t *H, uint32_t *W) {
    uint32_t a=H[0], b=H[1], c=H[2], d=H[3], e=H[4], f=H[5], g=H[6], h=H[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + d_K[i] + W[i];
        uint32_t T2 = Sigma0(a) + Maj(a, b, c);
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d; H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
}

// ============================================================
// KERNEL 1: Compute midstate from header Block 1 (64 bytes)
// Single-threaded — called once per job
// ============================================================
extern "C" __global__ void compute_midstate(
    const uint32_t *header_words,  // 20 words (80 bytes)
    uint32_t *midstate             // 8 words output
) {
    uint32_t H[8], W[64];
    for (int i = 0; i < 8; i++) H[i] = H_INIT[i];
    for (int i = 0; i < 16; i++) W[i] = header_words[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress(H, W);
    for (int i = 0; i < 8; i++) midstate[i] = H[i];
}

// ============================================================
// KERNEL 2: Cascade level (SHA-256 subfunction fan-out)
// Each thread: 3 inputs → 28 ops → 6 outputs (2 children)
// ============================================================
extern "C" __global__ void cascade_level(
    const uint32_t* __restrict__ xs_in,
    const uint32_t* __restrict__ ys_in,
    const uint32_t* __restrict__ zs_in,
    uint32_t* __restrict__ xs_out,
    uint32_t* __restrict__ ys_out,
    uint32_t* __restrict__ zs_out,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t x = xs_in[tid], y = ys_in[tid], z = zs_in[tid];

    // Layer 1: All 6 SHA-256 subfunctions
    uint32_t ch_a1 = x & y;
    uint32_t not_x = ~x;
    uint32_t maj_a2 = x & z;
    uint32_t maj_a3 = y & z;

    uint32_t s0_r2  = rotr(x, 2);   uint32_t s0_r13 = rotr(x, 13);  uint32_t s0_r22 = rotr(x, 22);
    uint32_t s1_r6  = rotr(y, 6);   uint32_t s1_r11 = rotr(y, 11);  uint32_t s1_r25 = rotr(y, 25);
    uint32_t g0_r7  = rotr(z, 7);   uint32_t g0_r18 = rotr(z, 18);  uint32_t g0_s3  = z >> 3;
    uint32_t xy = x ^ y;
    uint32_t g1_r17 = rotr(xy, 17); uint32_t g1_r19 = rotr(xy, 19); uint32_t g1_s10 = xy >> 10;

    // Layer 2: XOR aggregation
    uint32_t ch_a2  = not_x & z;
    uint32_t maj_x1 = ch_a1 ^ maj_a2;
    uint32_t s0_x1  = s0_r2 ^ s0_r13;
    uint32_t s1_x1  = s1_r6 ^ s1_r11;
    uint32_t g0_x1  = g0_r7 ^ g0_r18;
    uint32_t g1_x1  = g1_r17 ^ g1_r19;

    // Layer 3: Collapse
    uint32_t ch    = ch_a1 ^ ch_a2;
    uint32_t maj   = maj_x1 ^ maj_a3;
    uint32_t sig0  = s0_x1 ^ s0_r22;
    uint32_t sig1  = s1_x1 ^ s1_r25;
    uint32_t lsig0 = g0_x1 ^ g0_s3;
    uint32_t lsig1 = g1_x1 ^ g1_s10;

    // Fan-out with cross-entanglement
    uint32_t L = tid * 2, R = tid * 2 + 1;
    xs_out[L] = ch ^ lsig1;   xs_out[R] = sig1;
    ys_out[L] = maj;           ys_out[R] = lsig0;
    zs_out[L] = sig0;          zs_out[R] = ch ^ maj;
}

// ============================================================
// KERNEL 3: Verify candidate nonces from cascade leaves
// Each thread takes a candidate nonce, completes Block 2 +
// second SHA-256 pass using the pre-computed midstate
// ============================================================
extern "C" __global__ void verify_candidates(
    const uint32_t *midstate,         // 8 words (from Block 1)
    const uint32_t *tail_words,       // header words 16-19 (Block 2 prefix, word 3 = placeholder nonce)
    const uint32_t *candidate_nonces, // array of nonce candidates from cascade leaves
    uint32_t n_candidates,
    uint32_t *output_flag,
    uint32_t *output_nonce,
    const uint32_t *target_words
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_candidates) return;
    if (*output_flag == 1) return;

    uint32_t nonce = candidate_nonces[tid];
    uint32_t H[8], W[64];

    // Start from midstate (Block 1 already compressed)
    for (int i = 0; i < 8; i++) H[i] = midstate[i];

    // Block 2: last 16 bytes of header + padding
    W[0] = tail_words[0];
    W[1] = tail_words[1];
    W[2] = tail_words[2];
    W[3] = nonce;            // THE nonce
    W[4] = 0x80000000;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;             // 80 * 8 bits
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress(H, W);

    // Second SHA-256 pass on the 32-byte digest
    for (int i = 0; i < 8; i++) W[i] = H[i];
    W[8] = 0x80000000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;             // 32 * 8 bits
    for (int i = 0; i < 8; i++) H[i] = H_INIT[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress(H, W);

    // Compare hash vs target
    for (int i = 0; i < 8; i++) {
        if (H[i] < target_words[i]) {
            if (atomicCAS(output_flag, 0, 1) == 0) *output_nonce = nonce;
            return;
        }
        if (H[i] > target_words[i]) return;
    }
    if (atomicCAS(output_flag, 0, 1) == 0) *output_nonce = nonce;
}
"""

# -------------------------------------------------------------
# 4. INITIALIZATION
# -------------------------------------------------------------
print("==========================================================")
print(" SOVEREIGN CASCADE-AMPLIFIED STRATUM MINER")
print("==========================================================")

gpu_cfg = detect_gpu_config(vram_budget_pct=0.80)
print(f"\n[GPU] CC {gpu_cfg['compute_capability']} | SMs: {gpu_cfg['sm_count']}")
print(f"[GPU] VRAM: {gpu_cfg['free_vram_gb']:.1f} GB free / {gpu_cfg['total_vram_gb']:.1f} GB total")
print(f"[GPU] Budget: {gpu_cfg['usable_vram_gb']:.1f} GB ({gpu_cfg['usable_vram_gb']/gpu_cfg['total_vram_gb']*100:.0f}%)")
print(f"[GPU] Max cascade depth: {gpu_cfg['max_cascade_depth']}")

module = cp.RawModule(code=cuda_source)
k_midstate = module.get_function("compute_midstate")
k_cascade  = module.get_function("cascade_level")
k_verify   = module.get_function("verify_candidates")

WALLET = "bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0.antigravity"
POOL_PRIMARY = ('btc.viabtc.com', 3333)
POOL_FALLBACK = ('solo.ckpool.org', 3333)

CASCADE_TPB = gpu_cfg['cascade_tpb']
VERIFY_TPB  = gpu_cfg['verify_tpb']

# -------------------------------------------------------------
# 5. CASCADE ENGINE (memory-safe ping-pong)
# -------------------------------------------------------------
# Pick cascade depth: target ~100M-1B leaves per batch
# On B200 (180GB): depth 30 = 1B leaves, ~24GB ping-pong → easy
# On 4070 (8GB):   depth 25 = 33M leaves, ~0.8GB → safe
free_vram = cp.cuda.Device().mem_info[0]
# Each cascade level: 2^D cells × 12 bytes × 2 bufs
CASCADE_DEPTH = min(gpu_cfg['max_cascade_depth'], 28)
while (2**CASCADE_DEPTH) * 24 > free_vram * 0.60:
    CASCADE_DEPTH -= 1
CASCADE_DEPTH = max(CASCADE_DEPTH, 16)  # floor
LEAF_COUNT = 2 ** CASCADE_DEPTH

print(f"\n[Cascade] Depth: {CASCADE_DEPTH} | Leaves per batch: {LEAF_COUNT:,}")
print(f"[Cascade] VRAM for ping-pong: {LEAF_COUNT * 24 / 1e9:.2f} GB")

# Pre-allocate ping-pong buffers
buf_a = (cp.empty(LEAF_COUNT, dtype=cp.uint32),
         cp.empty(LEAF_COUNT, dtype=cp.uint32),
         cp.empty(LEAF_COUNT, dtype=cp.uint32))
buf_b = (cp.empty(LEAF_COUNT, dtype=cp.uint32),
         cp.empty(LEAF_COUNT, dtype=cp.uint32),
         cp.empty(LEAF_COUNT, dtype=cp.uint32))

# Pre-allocate verify outputs
d_flag  = cp.zeros(1, dtype=np.uint32)
d_nonce = cp.zeros(1, dtype=np.uint32)
d_midstate = cp.zeros(8, dtype=np.uint32)
d_tail     = cp.zeros(4, dtype=np.uint32)

print(f"[Cascade] Buffers pre-allocated.")

# -------------------------------------------------------------
# 6. BENCHMARK: Cascade + Verify pipeline
# -------------------------------------------------------------
def run_cascade(seed_x, seed_y, seed_z):
    """Run full cascade, return leaf x-values as candidate nonces."""
    buf_a[0][0] = seed_x
    buf_a[1][0] = seed_y
    buf_a[2][0] = seed_z

    src, dst = buf_a, buf_b
    n = 1
    for level in range(CASCADE_DEPTH):
        blocks = (n + CASCADE_TPB - 1) // CASCADE_TPB
        k_cascade((blocks,), (CASCADE_TPB,),
                  (src[0], src[1], src[2], dst[0], dst[1], dst[2], np.uint32(n)))
        n *= 2
        src, dst = dst, src
    cp.cuda.Stream.null.synchronize()
    return src[0][:LEAF_COUNT]  # leaf x-values = candidate nonces


# Warmup + benchmark
print("\n[Bench] Warming up cascade + verify pipeline...")
dummy_nonces = run_cascade(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372))
d_midstate.fill(0)
d_tail.fill(0)
d_target = cp.zeros(8, dtype=np.uint32)

# Benchmark cascade
times_c = []
for _ in range(5):
    t0 = time.perf_counter()
    leaves = run_cascade(np.uint32(0x6a09e667), np.uint32(0xbb67ae85), np.uint32(0x3c6ef372))
    t1 = time.perf_counter()
    times_c.append(t1 - t0)
cascade_sec = sorted(times_c)[2]

# Benchmark verify
times_v = []
vbpg = (LEAF_COUNT + VERIFY_TPB - 1) // VERIFY_TPB
for _ in range(5):
    d_flag.fill(0)
    t0 = time.perf_counter()
    k_verify((vbpg,), (VERIFY_TPB,),
             (d_midstate, d_tail, leaves, np.uint32(LEAF_COUNT), d_flag, d_nonce, d_target))
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    times_v.append(t1 - t0)
verify_sec = sorted(times_v)[2]

total_sec = cascade_sec + verify_sec
effective_hps = LEAF_COUNT / total_sec
cascade_cells_sec = (2**(CASCADE_DEPTH+1) - 1) / cascade_sec

print(f"[Bench] Cascade: {cascade_sec*1000:.1f} ms ({cascade_cells_sec/1e9:.2f} B cells/s)")
print(f"[Bench] Verify:  {verify_sec*1000:.1f} ms ({LEAF_COUNT/verify_sec/1e9:.2f} GH/s raw)")
print(f"[Bench] Total:   {total_sec*1000:.1f} ms per batch")
print(f"[Bench] Effective: {effective_hps/1e6:.1f} MH/s ({effective_hps/1e9:.3f} GH/s)")
print(f"[Bench] Candidates/batch: {LEAF_COUNT:,}")

print(f"\n[Cascade] Benchmark complete. Starting stratum cascade...")

# -------------------------------------------------------------
# 7. STRATUM PROTOCOL
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
# 8. MINING LOOP
# -------------------------------------------------------------
def stratum_mining_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    print("\n[Network] Connecting to stratum pool...")
    try:
        sock.connect(POOL_PRIMARY)
        print(f"[Network] Connected to {POOL_PRIMARY[0]}:{POOL_PRIMARY[1]}")
    except Exception as e:
        print(f"[Network] Primary failed ({e}), trying fallback...")
        sock.connect(POOL_FALLBACK)

    sub_res, b1 = parse_stratum(sock, '{"id": 1, "method": "mining.subscribe", "params": []}')
    extranonce1 = sub_res['result'][1]
    en2_sz = sub_res['result'][2]
    print(f"[Stratum] Subscribed. EN1: {extranonce1} | EN2 size: {en2_sz}")

    auth_res, b2 = parse_stratum(sock, f'{{"id": 2, "method": "mining.authorize", "params": ["{WALLET}", "x"]}}')
    print(f"[Stratum] Authorized. Waiting for jobs...")

    buffer = b2
    active_job = None
    extranonce2_int = 0
    pool_difficulty = 1
    total_hashes = 0
    total_cascades = 0
    session_start = time.time()
    batch_id = 0

    while True:
        # --- Poll for messages ---
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

        # --- Parse messages ---
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if not line.strip(): continue
            msg = json.loads(line)

            if msg.get('method') == 'mining.set_difficulty':
                pool_difficulty = msg['params'][0]
                print(f"\n[Stratum] Difficulty updated: {pool_difficulty}")

            if msg.get('method') == 'mining.notify':
                active_job = msg['params']
                print(f"\n[Job] New block (Job {active_job[0]})")
                extranonce2_int = 0
                batch_id = 0

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

        # --- Compute midstate (Block 1) ---
        d_header = cp.asarray(words_cpu)
        k_midstate((1,), (1,), (d_header, d_midstate))
        cp.cuda.Stream.null.synchronize()

        # Tail words (header[16:20]) for Block 2
        tail_cpu = words_cpu[16:20].copy()
        d_tail_job = cp.asarray(tail_cpu)

        # Target
        target_int = difficulty_to_target(pool_difficulty)
        d_target_job = cp.asarray(target_to_gpu_words(target_int))

        midstate_cpu = d_midstate.get()
        print(f"[Mining] EN2={extranonce2} | Diff={pool_difficulty} | Midstate[0]=0x{midstate_cpu[0]:08X}")
        print(f"[Cascade] Seeding from midstate → depth {CASCADE_DEPTH} → {LEAF_COUNT:,} candidates/batch")

        # --- Cascade mining loop ---
        while True:
            # Seed cascade from midstate + batch_id for diversity
            seed_x = np.uint32(int(midstate_cpu[0]) ^ batch_id)
            seed_y = np.uint32(int(midstate_cpu[4]))
            seed_z = np.uint32(int(midstate_cpu[2]))

            t0 = time.perf_counter()

            # Phase 1: Cascade generates candidate nonces
            candidate_nonces = run_cascade(seed_x, seed_y, seed_z)

            t_cascade = time.perf_counter()

            # Phase 2: Verify all candidates against target
            d_flag.fill(0)
            d_nonce.fill(0)
            vbpg = (LEAF_COUNT + VERIFY_TPB - 1) // VERIFY_TPB
            k_verify((vbpg,), (VERIFY_TPB,),
                     (d_midstate, d_tail_job, candidate_nonces,
                      np.uint32(LEAF_COUNT), d_flag, d_nonce, d_target_job))
            cp.cuda.Stream.null.synchronize()

            t1 = time.perf_counter()

            # Check for share
            if d_flag.get()[0] == 1:
                win_nonce = int(d_nonce.get()[0])
                win_hex = hex(win_nonce)[2:].zfill(8)
                win_swap = "".join(reversed([win_hex[i:i+2] for i in range(0, 8, 2)]))
                print(f"\n$$$ SHARE FOUND @ Nonce: {win_swap} (cascade batch {batch_id}) $$$")
                submit = {
                    "params": [WALLET, job_id, extranonce2, ntime, win_swap],
                    "id": 4, "method": "mining.submit"
                }
                try:
                    sock.sendall(json.dumps(submit).encode('utf-8') + b'\n')
                    print(f"[Network] Share submitted")
                except (BrokenPipeError, OSError) as e:
                    print(f"[Network] Submit failed: {e}")

            batch_id += 1
            total_hashes += LEAF_COUNT
            total_cascades += 1
            elapsed = max(t1 - t0, 0.001)
            c_ms = (t_cascade - t0) * 1000
            v_ms = (t1 - t_cascade) * 1000
            session_elapsed = max(time.time() - session_start, 1)
            instant_rate = LEAF_COUNT / elapsed
            avg_rate = total_hashes / session_elapsed

            print(f"[Cascade] batch={batch_id:>6} | c:{c_ms:>5.1f}ms + v:{v_ms:>5.1f}ms = {elapsed*1000:>6.1f}ms | "
                  f"{instant_rate/1e6:>7.1f} MH/s (avg {avg_rate/1e6:.1f}) | "
                  f"{total_hashes:>14,} candidates", end='\r')

            # Check for new jobs
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

            # Rotate extranonce2 after exhausting diverse seeds
            if batch_id > 0 and batch_id % 1000 == 0:
                extranonce2_int += 1
                break  # rebuild header with new EN2

# -------------------------------------------------------------
# 9. DAEMON
# -------------------------------------------------------------
while True:
    try:
        stratum_mining_loop()
    except KeyboardInterrupt:
        print(f"\n[Shutdown] Terminated by user.")
        break
    except Exception as e:
        print(f"\n[Reconnect] Lost connection: {e}")
        time.sleep(3)
