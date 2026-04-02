"""
Sovereign Cascade Miner v2 - Self-Spawning Fused Kernel
=======================================================
Each thread is the cascade. It walks root-to-leaf through D levels of SHA-256
subfunctions, chooses its child path from thread-id bits, and immediately
verifies the resulting nonce candidate. One kernel launch, no intermediate
buffers, no level-by-level synchronization, all visible GPUs used together.
"""

import binascii
import hashlib
import json
import socket
import struct
import time

import cupy as cp
import numpy as np


def dsha256(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


# -------------------------------------------------------------
# 1. STRATUM UTILITIES
# -------------------------------------------------------------
def build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches):
    coinb = coinb1 + extranonce1 + extranonce2 + coinb2
    cb_hash = dsha256(binascii.unhexlify(coinb))
    for branch in branches:
        cb_hash = dsha256(cb_hash + binascii.unhexlify(branch))
    return binascii.hexlify(cb_hash).decode("utf-8")


def swap_endian_words(hex_str):
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i + 8]
        result += "".join(reversed([word[j:j + 2] for j in range(0, 8, 2)]))
    return result


def difficulty_to_target(difficulty):
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return max(int(diff1_target / max(float(difficulty), 1.0)), 1)


def target_to_gpu_words(target_int):
    target_bytes = target_int.to_bytes(32, byteorder="big")
    words = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        words[i] = struct.unpack(">I", target_bytes[i * 4:(i + 1) * 4])[0]
    return words


def header_meets_target(header_prefix, nonce, target_int):
    header = header_prefix + struct.pack(">I", nonce)
    return int.from_bytes(dsha256(header), byteorder="little") <= target_int


# -------------------------------------------------------------
# 2. MULTI-GPU DETECTION
# -------------------------------------------------------------
def detect_all_gpus():
    n_gpus = cp.cuda.runtime.getDeviceCount()
    gpus = []
    for gid in range(n_gpus):
        with cp.cuda.Device(gid):
            dev = cp.cuda.Device(gid)
            free_mem, total_mem = dev.mem_info
            attrs = dev.attributes
            gpus.append(
                {
                    "id": gid,
                    "cc": f"{attrs.get('ComputeCapabilityMajor', 0)}.{attrs.get('ComputeCapabilityMinor', 0)}",
                    "sm_count": attrs.get("MultiProcessorCount", 1),
                    "max_tpb": attrs.get("MaxThreadsPerBlock", 1024),
                    "free_vram_gb": free_mem / 1e9,
                    "total_vram_gb": total_mem / 1e9,
                }
            )
    return gpus


# -------------------------------------------------------------
# 3. FUSED SELF-SPAWNING CUDA KERNEL
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
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) |
           ((x & 0xFF000000u) >> 24);
}
__device__ __forceinline__ uint32_t sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

__device__ void compress(uint32_t *H, uint32_t *W) {
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + s1 + ch + d_K[i] + W[i];
        uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

extern "C" __global__ void cascade_mine(
    const uint32_t *midstate,
    const uint32_t *tail_words,
    const uint32_t *target_words,
    const uint32_t seed_x,
    const uint32_t seed_y,
    const uint32_t seed_z,
    const uint32_t cascade_depth,
    uint32_t *output_flag,
    uint32_t *output_nonce,
    const uint32_t n_threads
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_threads) return;
    if (*output_flag == 1) return;

    uint32_t x = seed_x;
    uint32_t y = seed_y;
    uint32_t z = seed_z;

    for (uint32_t level = 0; level < cascade_depth; level++) {
        uint32_t ch_a1 = x & y;
        uint32_t not_x = ~x;
        uint32_t maj_a2 = x & z;
        uint32_t maj_a3 = y & z;

        uint32_t s0_r2 = rotr(x, 2);
        uint32_t s0_r13 = rotr(x, 13);
        uint32_t s0_r22 = rotr(x, 22);
        uint32_t s1_r6 = rotr(y, 6);
        uint32_t s1_r11 = rotr(y, 11);
        uint32_t s1_r25 = rotr(y, 25);
        uint32_t g0_r7 = rotr(z, 7);
        uint32_t g0_r18 = rotr(z, 18);
        uint32_t g0_s3 = z >> 3;
        uint32_t xy = x ^ y;
        uint32_t g1_r17 = rotr(xy, 17);
        uint32_t g1_r19 = rotr(xy, 19);
        uint32_t g1_s10 = xy >> 10;

        uint32_t ch_a2 = not_x & z;
        uint32_t maj_x1 = ch_a1 ^ maj_a2;
        uint32_t s0_x1 = s0_r2 ^ s0_r13;
        uint32_t s1_x1 = s1_r6 ^ s1_r11;
        uint32_t g0_x1 = g0_r7 ^ g0_r18;
        uint32_t g1_x1 = g1_r17 ^ g1_r19;

        uint32_t ch_out = ch_a1 ^ ch_a2;
        uint32_t maj_out = maj_x1 ^ maj_a3;
        uint32_t sig0_out = s0_x1 ^ s0_r22;
        uint32_t sig1_out = s1_x1 ^ s1_r25;
        uint32_t lsig0_out = g0_x1 ^ g0_s3;
        uint32_t lsig1_out = g1_x1 ^ g1_s10;

        if ((tid >> level) & 1) {
            x = sig1_out;
            y = lsig0_out;
            z = ch_out ^ maj_out;
        } else {
            x = ch_out ^ lsig1_out;
            y = maj_out;
            z = sig0_out;
        }
    }

    uint32_t nonce = x;
    uint32_t H[8];
    uint32_t W[64];
    for (int i = 0; i < 8; i++) H[i] = midstate[i];

    W[0] = tail_words[0];
    W[1] = tail_words[1];
    W[2] = tail_words[2];
    W[3] = nonce;
    W[4] = 0x80000000;
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j - 2]) + W[j - 7] + sigma0(W[j - 15]) + W[j - 16];
    compress(H, W);

    for (int i = 0; i < 8; i++) W[i] = H[i];
    W[8] = 0x80000000;
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;
    for (int i = 0; i < 8; i++) H[i] = H_INIT[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j - 2]) + W[j - 7] + sigma0(W[j - 15]) + W[j - 16];
    compress(H, W);

    for (int i = 0; i < 8; i++) {
        uint32_t hash_word = bswap32(H[7 - i]);
        if (hash_word < target_words[i]) {
            if (atomicCAS(output_flag, 0, 1) == 0) *output_nonce = nonce;
            return;
        }
        if (hash_word > target_words[i]) return;
    }

    if (atomicCAS(output_flag, 0, 1) == 0) *output_nonce = nonce;
}
"""


# -------------------------------------------------------------
# 4. CPU MIDSTATE
# -------------------------------------------------------------
_K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]
_H_INIT = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]
M32 = 0xFFFFFFFF


def _rotr32(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def compute_midstate(header_words_16):
    w = [int(word) & M32 for word in header_words_16[:16]]
    for j in range(16, 64):
        s0 = _rotr32(w[j - 15], 7) ^ _rotr32(w[j - 15], 18) ^ (w[j - 15] >> 3)
        s1 = _rotr32(w[j - 2], 17) ^ _rotr32(w[j - 2], 19) ^ (w[j - 2] >> 10)
        w.append((w[j - 16] + s0 + w[j - 7] + s1) & M32)

    a, b, c, d, e, f, g, h = [int(v) for v in _H_INIT]
    for i in range(64):
        s1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
        ch = (e & f) ^ ((~e & M32) & g)
        t1 = (h + s1 + ch + _K[i] + w[i]) & M32
        s0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (s0 + maj) & M32
        h = g
        g = f
        f = e
        e = (d + t1) & M32
        d = c
        c = b
        b = a
        a = (t1 + t2) & M32

    return np.array(
        [
            (_H_INIT[0] + a) & M32,
            (_H_INIT[1] + b) & M32,
            (_H_INIT[2] + c) & M32,
            (_H_INIT[3] + d) & M32,
            (_H_INIT[4] + e) & M32,
            (_H_INIT[5] + f) & M32,
            (_H_INIT[6] + g) & M32,
            (_H_INIT[7] + h) & M32,
        ],
        dtype=np.uint32,
    )


# -------------------------------------------------------------
# 5. INIT
# -------------------------------------------------------------
print("==========================================================")
print(" SOVEREIGN CASCADE MINER V2 - SELF-SPAWNING FUSED KERNEL")
print("==========================================================")

gpus = detect_all_gpus()
if not gpus:
    raise RuntimeError("No visible CUDA GPUs")

n_gpus = len(gpus)
total_sm = sum(gpu["sm_count"] for gpu in gpus)
print(f"[System] {n_gpus} GPU(s) visible | {total_sm} total SMs")
for gpu in gpus:
    print(
        f"  GPU {gpu['id']}: CC {gpu['cc']} | {gpu['sm_count']} SMs | "
        f"{gpu['free_vram_gb']:.1f}/{gpu['total_vram_gb']:.1f} GB"
    )

CASCADE_DEPTH = 28
THREADS_PER_GPU = 1 << CASCADE_DEPTH
TPB = 256
BPG = (THREADS_PER_GPU + TPB - 1) // TPB

print(f"[Cascade] Depth={CASCADE_DEPTH} | Threads/GPU={THREADS_PER_GPU:,}")
print(f"[Cascade] Total threads per batch={THREADS_PER_GPU * n_gpus:,}")
print(f"[Kernel] TPB={TPB} | BPG={BPG:,}")

WALLET = "bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0.antigravity"
POOL_PRIMARY = ("btc.viabtc.com", 3333)
POOL_FALLBACK = ("solo.ckpool.org", 3333)


gpu_kernels = []
gpu_flags = []
gpu_nonces = []
gpu_midstates = []
gpu_tails = []
gpu_targets = []

for gpu in gpus:
    with cp.cuda.Device(gpu["id"]):
        module = cp.RawModule(code=cuda_source)
        gpu_kernels.append(module.get_function("cascade_mine"))
        gpu_flags.append(cp.zeros(1, dtype=cp.uint32))
        gpu_nonces.append(cp.zeros(1, dtype=cp.uint32))
        gpu_midstates.append(cp.zeros(8, dtype=cp.uint32))
        gpu_tails.append(cp.zeros(3, dtype=cp.uint32))
        gpu_targets.append(cp.zeros(8, dtype=cp.uint32))

print(f"[Kernel] Compiled on {n_gpus} GPU(s)")


# -------------------------------------------------------------
# 6. BENCHMARK
# -------------------------------------------------------------
print("[Bench] Warmup")
for idx, gpu in enumerate(gpus):
    with cp.cuda.Device(gpu["id"]):
        gpu_flags[idx].fill(0)
        gpu_kernels[idx](
            (BPG,),
            (TPB,),
            (
                gpu_midstates[idx],
                gpu_tails[idx],
                gpu_targets[idx],
                np.uint32(0x6A09E667),
                np.uint32(0xBB67AE85),
                np.uint32(0x3C6EF372),
                np.uint32(CASCADE_DEPTH),
                gpu_flags[idx],
                gpu_nonces[idx],
                np.uint32(THREADS_PER_GPU),
            ),
        )
for gpu in gpus:
    with cp.cuda.Device(gpu["id"]):
        cp.cuda.Stream.null.synchronize()

bench_times = []
for _ in range(3):
    t0 = time.perf_counter()
    for idx, gpu in enumerate(gpus):
        with cp.cuda.Device(gpu["id"]):
            gpu_flags[idx].fill(0)
            gpu_kernels[idx](
                (BPG,),
                (TPB,),
                (
                    gpu_midstates[idx],
                    gpu_tails[idx],
                    gpu_targets[idx],
                    np.uint32(0x6A09E667),
                    np.uint32(0xBB67AE85),
                    np.uint32(0x3C6EF372),
                    np.uint32(CASCADE_DEPTH),
                    gpu_flags[idx],
                    gpu_nonces[idx],
                    np.uint32(THREADS_PER_GPU),
                ),
            )
    for gpu in gpus:
        with cp.cuda.Device(gpu["id"]):
            cp.cuda.Stream.null.synchronize()
    bench_times.append(time.perf_counter() - t0)

med_time = sorted(bench_times)[len(bench_times) // 2]
effective_hps = (THREADS_PER_GPU * n_gpus) / max(med_time, 1e-9)
print(f"[Bench] {med_time * 1000:.1f} ms/batch | {effective_hps / 1e9:.3f} GH/s effective")


# -------------------------------------------------------------
# 7. STRATUM
# -------------------------------------------------------------
def parse_stratum(socket_obj, request_payload=None, expected_id=None, buffer=""):
    if request_payload:
        socket_obj.sendall(request_payload.encode("utf-8") + b"\n")
    preserved = []
    while True:
        try:
            buffer += socket_obj.recv(4096).decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line)
                if expected_id is None or msg.get("id") == expected_id:
                    preserved_buffer = "".join(preserved)
                    return msg, preserved_buffer + buffer
                preserved.append(line + "\n")
        except socket.timeout:
            return None, "".join(preserved) + buffer


# -------------------------------------------------------------
# 8. MINING LOOP
# -------------------------------------------------------------
def stratum_mining_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    try:
        sock.connect(POOL_PRIMARY)
        print(f"[Network] Connected to {POOL_PRIMARY[0]}:{POOL_PRIMARY[1]}")
    except Exception as exc:
        print(f"[Network] Primary failed ({exc}), trying fallback")
        sock.connect(POOL_FALLBACK)
        print(f"[Network] Connected to {POOL_FALLBACK[0]}:{POOL_FALLBACK[1]}")

    sub_res, buffer = parse_stratum(sock, '{"id":1,"method":"mining.subscribe","params":[]}', expected_id=1)
    extranonce1 = sub_res["result"][1]
    en2_sz = sub_res["result"][2]
    _, buffer = parse_stratum(
        sock,
        f'{{"id":2,"method":"mining.authorize","params":["{WALLET}","x"]}}',
        expected_id=2,
        buffer=buffer,
    )
    print(f"[Stratum] Authorized | extranonce2 bytes={en2_sz}")

    active_job = None
    extranonce2_int = 0
    pool_difficulty = 1
    total_hashes = 0
    session_start = time.time()
    batch_id = 0
    submitted_shares = set()
    next_submit_id = 4
    pending_submits = {}

    while True:
        sock.setblocking(False)
        try:
            incoming = sock.recv(8192).decode("utf-8")
            if not incoming:
                raise ConnectionError("server closed connection")
            buffer += incoming
        except BlockingIOError:
            pass
        finally:
            sock.setblocking(True)

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not line.strip():
                continue
            msg = json.loads(line)
            if msg.get("method") == "mining.set_difficulty":
                pool_difficulty = msg["params"][0]
                print(f"[Stratum] Difficulty={pool_difficulty}")
            elif msg.get("method") == "mining.notify":
                active_job = msg["params"]
                extranonce2_int = 0
                batch_id = 0
                submitted_shares.clear()
                print(f"[Job] {active_job[0]}")
            elif msg.get("id") in pending_submits:
                share_label = pending_submits.pop(msg["id"])
                print(
                    f"[Share] {share_label} | "
                    f"{'ACCEPTED' if msg.get('result') else f'REJECTED {msg.get('error')}'}"
                )

        if active_job is None:
            time.sleep(0.25)
            continue

        job_id, prevhash, coinb1, coinb2, branches, version, nbits, ntime, _clean = active_job[:9]
        extranonce2 = hex(extranonce2_int)[2:].zfill(en2_sz * 2)
        merkle_root = build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches)
        header_hex = (
            swap_endian_words(version)
            + swap_endian_words(prevhash)
            + merkle_root
            + swap_endian_words(ntime)
            + swap_endian_words(nbits)
            + "00000000"
        )
        header_bytes = binascii.unhexlify(header_hex)

        header_words = np.zeros(20, dtype=np.uint32)
        for i in range(20):
            header_words[i] = struct.unpack(">I", header_bytes[i * 4:(i + 1) * 4])[0]

        midstate_cpu = compute_midstate(header_words[:16])
        tail_cpu = header_words[16:19].copy()
        target_int = difficulty_to_target(pool_difficulty)
        target_cpu = target_to_gpu_words(target_int)
        header_prefix = header_bytes[:76]

        for idx, gpu in enumerate(gpus):
            with cp.cuda.Device(gpu["id"]):
                gpu_midstates[idx].set(midstate_cpu)
                gpu_tails[idx].set(tail_cpu)
                gpu_targets[idx].set(target_cpu)

        while True:
            t0 = time.perf_counter()
            for idx, gpu in enumerate(gpus):
                with cp.cuda.Device(gpu["id"]):
                    seed_id = batch_id * n_gpus + idx
                    seed_x = np.uint32(int(midstate_cpu[0]) ^ seed_id)
                    seed_y = np.uint32(int(midstate_cpu[4]) ^ (seed_id >> 16))
                    seed_z = np.uint32(int(midstate_cpu[2]))
                    gpu_flags[idx].fill(0)
                    gpu_nonces[idx].fill(0)
                    gpu_kernels[idx](
                        (BPG,),
                        (TPB,),
                        (
                            gpu_midstates[idx],
                            gpu_tails[idx],
                            gpu_targets[idx],
                            seed_x,
                            seed_y,
                            seed_z,
                            np.uint32(CASCADE_DEPTH),
                            gpu_flags[idx],
                            gpu_nonces[idx],
                            np.uint32(THREADS_PER_GPU),
                        ),
                    )

            for gpu in gpus:
                with cp.cuda.Device(gpu["id"]):
                    cp.cuda.Stream.null.synchronize()
            elapsed = max(time.perf_counter() - t0, 1e-9)

            found_share = False
            for idx, gpu in enumerate(gpus):
                with cp.cuda.Device(gpu["id"]):
                    if int(gpu_flags[idx].get()[0]) != 1:
                        continue
                    nonce = int(gpu_nonces[idx].get()[0])
                    nonce_hex = hex(nonce)[2:].zfill(8)
                    nonce_submit = "".join(reversed([nonce_hex[i:i + 2] for i in range(0, 8, 2)]))
                    share_key = (job_id, extranonce2, ntime, nonce_submit)
                    if share_key in submitted_shares:
                        continue
                    if not header_meets_target(header_prefix, nonce, target_int):
                        print(f"[Share] Dropped invalid GPU hit on GPU {gpu['id']} nonce={nonce_submit}")
                        continue
                    submit_id = next_submit_id
                    next_submit_id += 1
                    submit = {
                        "params": [WALLET, job_id, extranonce2, ntime, nonce_submit],
                        "id": submit_id,
                        "method": "mining.submit",
                    }
                    sock.sendall(json.dumps(submit).encode("utf-8") + b"\n")
                    submitted_shares.add(share_key)
                    pending_submits[submit_id] = f"GPU {gpu['id']} nonce={nonce_submit} batch={batch_id}"
                    print(f"[Share] GPU {gpu['id']} nonce={nonce_submit} batch={batch_id}")
                    found_share = True

            batch_id += 1
            total_hashes += THREADS_PER_GPU * n_gpus
            avg_rate = total_hashes / max(time.time() - session_start, 1e-9)
            print(
                f"[Fused] batch={batch_id:>6} | {elapsed * 1000:>7.2f} ms | "
                f"{(THREADS_PER_GPU * n_gpus) / elapsed / 1e9:>7.3f} GH/s "
                f"(avg {avg_rate / 1e9:.3f})",
                end="\r",
            )

            sock.setblocking(False)
            new_job = False
            try:
                incoming = sock.recv(8192).decode("utf-8")
                if incoming:
                    buffer += incoming
                    if "mining.notify" in incoming:
                        new_job = True
            except BlockingIOError:
                pass
            finally:
                sock.setblocking(True)

            if new_job:
                break
            if found_share:
                pass
            if batch_id % 1000 == 0:
                extranonce2_int += 1
                break


# -------------------------------------------------------------
# 9. DAEMON
# -------------------------------------------------------------
while True:
    try:
        stratum_mining_loop()
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopped")
        break
    except Exception as exc:
        print(f"\n[Reconnect] {exc}")
        time.sleep(3)
