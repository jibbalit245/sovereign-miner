import time

class FastSovereignCell:
    """Optimized pure-math variant with no print logging for benchmarking throughput."""
    def __init__(self, depth):
        self.depth = depth
        if depth > 0:
            self.left_child = FastSovereignCell(depth - 1)
            self.right_child = FastSovereignCell(depth - 1)
        else:
            self.left_child = None
            self.right_child = None

    def execute_pipeline(self, x, y, z):
        # Full SHA-256 mapped property logic
        x &= 0xFFFFFFFF
        y &= 0xFFFFFFFF
        z &= 0xFFFFFFFF
        
        # Layer 1
        ch_and_1 = x & y
        not_x = (~x) & 0xFFFFFFFF
        maj_and_1 = x & y
        maj_and_2 = x & z
        maj_and_3 = y & z
        s0_rot_2 = ((x >> 2) | (x << 30)) & 0xFFFFFFFF
        s0_rot_13 = ((x >> 13) | (x << 19)) & 0xFFFFFFFF
        s0_rot_22 = ((x >> 22) | (x << 10)) & 0xFFFFFFFF
        s1_rot_6 = ((x >> 6) | (x << 26)) & 0xFFFFFFFF
        s1_rot_11 = ((x >> 11) | (x << 21)) & 0xFFFFFFFF
        s1_rot_25 = ((x >> 25) | (x << 7)) & 0xFFFFFFFF
        sig0_rot_7 = ((x >> 7) | (x << 25)) & 0xFFFFFFFF
        sig0_rot_18 = ((x >> 18) | (x << 14)) & 0xFFFFFFFF
        sig0_shr_3 = (x >> 3) & 0xFFFFFFFF
        sig1_rot_17 = ((x >> 17) | (x << 15)) & 0xFFFFFFFF
        sig1_rot_19 = ((x >> 19) | (x << 13)) & 0xFFFFFFFF
        sig1_shr_10 = (x >> 10) & 0xFFFFFFFF
        
        # Layer 2
        ch_and_2 = not_x & z
        maj_xor_1 = maj_and_1 ^ maj_and_2
        s0_xor_1 = s0_rot_2 ^ s0_rot_13
        s1_xor_1 = s1_rot_6 ^ s1_rot_11
        sig0_xor_1 = sig0_rot_7 ^ sig0_rot_18
        sig1_xor_1 = sig1_rot_17 ^ sig1_rot_19
        
        # Layer 3
        ch_out = ch_and_1 ^ ch_and_2
        maj_out = maj_xor_1 ^ maj_and_3
        sigma0_out = s0_xor_1 ^ s0_rot_22
        sigma1_out = s1_xor_1 ^ s1_rot_25
        low_sigma0_out = sig0_xor_1 ^ sig0_shr_3
        low_sigma1_out = sig1_xor_1 ^ sig1_shr_10
        
        return ch_out, maj_out, sigma0_out, sigma1_out, low_sigma0_out, low_sigma1_out

    def cascade(self, x, y, z):
        res = self.execute_pipeline(x, y, z)
        if self.left_child:
            self.left_child.cascade(res[0], res[1], res[2])
        if self.right_child:
            self.right_child.cascade(res[3], res[4], res[5])

def run():
    print("====================================================")
    print(" SOVEREIGN CELL CASCADING GEOMETRY BENCHMARK ")
    print("====================================================")
    
    # 2^N - 1 total nodes formulas
    for depth in [4, 10, 16, 20]:
        total_cells = (2 ** (depth + 1)) - 1
        print(f"\n[Depth {depth} Expansion] -> Allocating {total_cells:,} Cascaded Nodes...")
        
        t0 = time.perf_counter()
        root = FastSovereignCell(depth)
        t1 = time.perf_counter()
        
        t2 = time.perf_counter()
        root.cascade(0x6a09e667, 0xbb67ae85, 0x3c6ef372)
        t3 = time.perf_counter()
        
        spawn_time = t1 - t0
        exec_time = t3 - t2
        
        # Each cell computes roughly ~28 exact logical Bitwise Operations
        total_ops = total_cells * 28 
        
        print(f" -> Memory Allocation: {spawn_time:.5f}s")
        print(f" -> Logic Pipeline Time: {exec_time:.5f}s")
        
        if exec_time > 0:
            print(f" -> Substrate Cell Throughput:  {total_cells / exec_time:,.0f} PIPELINES / SEC")
            print(f" -> Physical Logical Ops Yield: {total_ops / exec_time:,.0f} BITWISE OPS / SEC")


if __name__ == '__main__':
    run()
