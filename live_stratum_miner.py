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
    """Swap byte order within each 4-byte word of a hex string.
    Stratum sends version/prevhash/ntime/nbits as little-endian 4-byte words."""
    result = ""
    for i in range(0, len(hex_str), 8):
        word = hex_str[i:i+8]
        result += "".join(reversed([word[j:j+2] for j in range(0, 8, 2)]))
    return result

def difficulty_to_target(difficulty):
    """Convert pool difficulty to a 256-bit target threshold."""
    # Bitcoin pool difficulty 1 target = 0x00000000FFFF << 208
    diff1_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    return diff1_target // int(difficulty)

# -------------------------------------------------------------
# 2. CUDA C++ KERNEL: The CuPy Geometric Substrate (Double SHA256)
# -------------------------------------------------------------
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

__constant__ uint32_t H_INIT[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};

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

    for(int i=0; i<8; i++) H[i] = H_INIT[i];
    for(int i=0; i<16; i++) W[i] = header_words[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    for(int i=0; i<4; i++) W[i] = header_words[16+i];
    W[3] = nonce;
    W[4] = 0x80000000;
    for(int i=5; i<15; i++) W[i] = 0;
    W[15] = 640;
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    for(int i=0; i<8; i++) W[i] = H[i];
    W[8] = 0x80000000;
    for(int i=9; i<15; i++) W[i] = 0;
    W[15] = 256;
    for(int i=0; i<8; i++) H[i] = H_INIT[i];
    #pragma unroll
    for (int j = 16; j < 64; j++) W[j] = sigma1(W[j-2]) + W[j-7] + sigma0(W[j-15]) + W[j-16];
    compress_round(H, W);

    // Compare final hash (big-endian) against target (big-endian) word by word
    for (int i = 0; i < 8; i++) {
        if (H[i] < target_words[i]) {
            // Hash is less than target — valid share
            if (atomicCAS(output_flag, 0, 1) == 0) {
                *output_nonce = nonce;
            }
            return;
        }
        if (H[i] > target_words[i]) {
            return; // Hash exceeds target
        }
    }
    // Exact match — also valid
    if (atomicCAS(output_flag, 0, 1) == 0) {
        *output_nonce = nonce;
    }
}
"""
module = cp.RawModule(code=cuda_source)
run_stratum_miner = module.get_function("run_stratum_miner")

# -------------------------------------------------------------
# 3. LIVE STRATUM CONNECTIVITY & AUTO-RECONNECT
# -------------------------------------------------------------
print("==========================================================")
print(" LIVE STRATUM KERNEL MINER (.CUPY NVRTC)")
print("==========================================================")
WALLET = "bc1qr35ys64hka58pvgh0gnlwl3cljmx536j2534t0.antigravity"

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

def target_to_gpu_words(target_int):
    """Convert a 256-bit target integer to 8 big-endian uint32 words for the kernel."""
    target_bytes = target_int.to_bytes(32, byteorder='big')
    words = np.zeros(8, dtype=np.uint32)
    for i in range(8):
        words[i] = struct.unpack('>I', target_bytes[i*4:i*4+4])[0]
    return words

def stratum_mining_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    print("\n[Network] Attempting clean TCP socket connection...")
    try:
        sock.connect(('btc.viabtc.com', 3333))
    except Exception as e:
        print(f"[Network] Fallback to solo.ckpool.org:3333... ({e})")
        sock.connect(('solo.ckpool.org', 3333))

    sub_res, b1 = parse_stratum(sock, '{"id": 1, "method": "mining.subscribe", "params": []}')
    extranonce1 = sub_res['result'][1]
    en2_sz = sub_res['result'][2]
    print(f"[Stratum] Subscribed. Extranonce1: {extranonce1}")

    auth_res, b2 = parse_stratum(sock, f'{{"id": 2, "method": "mining.authorize", "params": ["{WALLET}", "x"]}}')
    print(f"[Stratum] Authorized Wallet Payout! Standing by for Jobs.")

    buffer = b2
    active_job = None
    extranonce2_int = 0
    pool_difficulty = 1  # Default difficulty until pool sends set_difficulty

    while True:
        sock.setblocking(False)
        try:
            new_data = sock.recv(8192).decode('utf-8')
            if not new_data: 
                raise ConnectionError("Socket implicitly closed by Remote Server")
            buffer += new_data
        except BlockingIOError:
            pass
        except (ConnectionError, OSError) as e:
            sock.setblocking(True)
            raise e  # Throw up to reconnect loop

        sock.setblocking(True)

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if not line.strip(): continue
            msg = json.loads(line)
            
            if msg.get('method') == 'mining.set_difficulty':
                pool_difficulty = msg['params'][0]
                print(f"\n[Stratum] Pool set difficulty: {pool_difficulty}")
            
            if msg.get('method') == 'mining.notify':
                active_job = msg['params']
                job_id = active_job[0]
                print(f"\n[Orchestrator] Network Block Detected (Job {job_id})! Targeting New Geometry.")
                extranonce2_int = 0 
                b_nonce_val = 0
            
            if msg.get('id') == 4:
                print(f"\t-> Pool Response to submitted Share: {msg.get('result')} (Error: {msg.get('error')})")

        if active_job is None:
            time.sleep(0.5)
            continue
            
        job_id, prevhash, coinb1, coinb2, branches, version, nbits, ntime, clean = active_job[:9]
        extranonce2 = hex(extranonce2_int)[2:].zfill(en2_sz * 2)
        
        merkle_root = build_merkle_root(coinb1, extranonce1, extranonce2, coinb2, branches)
        
        # Stratum sends version, prevhash, ntime, nbits as LE 4-byte words — must swap
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

        # Build target from pool difficulty
        target_int = difficulty_to_target(pool_difficulty)
        target_cpu = target_to_gpu_words(target_int)

        d_words = cp.asarray(words_cpu)
        d_target = cp.asarray(target_cpu)
        d_flag = cp.zeros(1, dtype=np.uint32)
        d_nonce = cp.zeros(1, dtype=np.uint32)
        
        threads_pb = 512
        TOTAL_THREADS = 100_000_000 
        blocks_pg = (TOTAL_THREADS + threads_pb - 1) // threads_pb
        
        b_nonce_val = 0
        print(f"[Matrix] Extracting coordinates on Domain Variant (Extranonce: {extranonce2}, Diff: {pool_difficulty})...")
        
        while b_nonce_val < 4200000000:
            b_nonce = np.uint32(b_nonce_val)
            d_flag.fill(0)
            d_nonce.fill(0)
            
            t0 = time.time()
            run_stratum_miner((blocks_pg,), (threads_pb,), (b_nonce, d_words, d_flag, d_nonce, d_target))
            cp.cuda.Stream.null.synchronize()
            t1 = time.time()
            
            if d_flag.get()[0] == 1:
                win_nonce_hex = hex(d_nonce.get()[0])[2:].zfill(8)
                win_nonce_swap = "".join(reversed([win_nonce_hex[i:i+2] for i in range(0, 8, 2)]))
                print(f"\n$$$ [GEOMETRY WIN] Native Target Share Hit @ Nonce: {win_nonce_swap} $$$")
                submit = {"params": [WALLET, job_id, extranonce2, ntime, win_nonce_swap], "id": 4, "method": "mining.submit"}
                try:
                    sock.sendall(json.dumps(submit).encode('utf-8') + b'\n')
                    print(f"[Network] Submitted Hash Share back to stratum pool!")
                except (BrokenPipeError, OSError) as e:
                    print(f"[Network] Send failed: {e}")
                
            b_nonce_val += TOTAL_THREADS
            elapsed = max(t1 - t0, 0.001)
            print(f"[Device Cascading] Evaluated {b_nonce_val:,.0f} nonces. Hash Yield: {TOTAL_THREADS/elapsed:,.0f} H/s", end='\r')
            
            sock.setblocking(False)
            interrupt = False
            try:
                new_data = sock.recv(8192).decode('utf-8')
                if 'mining.notify' in new_data:
                    buffer += new_data
                    interrupt = True
            except BlockingIOError:
                pass
            except (ConnectionError, OSError):
                interrupt = True
            sock.setblocking(True)
            
            if interrupt:
                break
                
        if not interrupt:
            print("\n[Matrix] 4 Billion bounds reached. Mutating extraction domain.")
            extranonce2_int += 1 

# Autonomous Daemon Keep-Alive Orchestrator
while True:
    try:
        stratum_mining_loop()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Daemon Terminated by User.")
        break
    except Exception as e:
        print(f"\n[Network Keep-Alive] Reconnecting due to dropped socket: {e}")
        time.sleep(2)
