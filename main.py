from graph_parser import parse_algorithm
from layered_optimizer import compute_layered_lanes, print_layered_geometry
import math

# SHA-256 Message Expansion & Compression Logic
ALGORITHM_SHA256 = """
# Ch(x, y, z) = (x & y) ^ (~x & z)
ch_and_1 = x & y
not_x = ~ x
ch_and_2 = not_x & z
Ch_out = ch_and_1 ^ ch_and_2

# Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)
maj_and_1 = x & y
maj_and_2 = x & z
maj_and_3 = y & z
maj_xor_1 = maj_and_1 ^ maj_and_2
Maj_out = maj_xor_1 ^ maj_and_3

# Sigma0(x) = ROTR_2(x) ^ ROTR_13(x) ^ ROTR_22(x)
s0_rot_2 = x ROTR 2
s0_rot_13 = x ROTR 13
s0_rot_22 = x ROTR 22
s0_xor_1 = s0_rot_2 ^ s0_rot_13
Sigma0_out = s0_xor_1 ^ s0_rot_22

# Sigma1(x) = ROTR_6(x) ^ ROTR_11(x) ^ ROTR_25(x)
s1_rot_6 = x ROTR 6
s1_rot_11 = x ROTR 11
s1_rot_25 = x ROTR 25
s1_xor_1 = s1_rot_6 ^ s1_rot_11
Sigma1_out = s1_xor_1 ^ s1_rot_25

# sigma0(x) = ROTR_7(x) ^ ROTR_18(x) ^ SHR_3(x)
sig0_rot_7 = x ROTR 7
sig0_rot_18 = x ROTR 18
sig0_shr_3 = x SHR 3
sig0_xor_1 = sig0_rot_7 ^ sig0_rot_18
sigma0_out = sig0_xor_1 ^ sig0_shr_3

# sigma1(x) = ROTR_17(x) ^ ROTR_19(x) ^ SHR_10(x)
sig1_rot_17 = x ROTR 17
sig1_rot_19 = x ROTR 19
sig1_shr_10 = x SHR 10
sig1_xor_1 = sig1_rot_17 ^ sig1_rot_19
sigma1_out = sig1_xor_1 ^ sig1_shr_10
"""

def map_algorithm_to_layered_lanes(algo_string):
    print("Parsing Algorithm...")
    nodes = parse_algorithm(algo_string)
    
    # Compute the Topological Layers (Lanes)
    layered_lanes, layers = compute_layered_lanes(nodes)
    
    print("\nExtracting Pipeline Logic...")
    # Formats output visually and drops JSON map data to console
    print_layered_geometry(layered_lanes, nodes)

if __name__ == "__main__":
    print("========================================")
    print(" SHA-256 TO PIPELINE GEOMETRY MAPPER ")
    print("========================================")
    map_algorithm_to_layered_lanes(ALGORITHM_SHA256)
