def rotr(x, n):
    """Right Rotate mapping matching SHA-256 architecture."""
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

def shr(x, n):
    """Right Shift mapping."""
    return (x >> n) & 0xFFFFFFFF

class SovereignCell:
    """
    A geometric node embodying the 4-Layer SHA-256 pipeline topology.
    It recursively spawns two identical cell instances beneath itself 
    to create a live fractal/cascading execution sequence.
    """
    def __init__(self, cell_id, cascade_depth):
        self.cell_id = cell_id
        self.depth = cascade_depth
        
        # SPAWN IDENTICAL OFFSPRING TO CONTINUE THE FRACTAL
        if self.depth > 0:
            self.left_child = SovereignCell(cell_id + "_L", self.depth - 1)
            self.right_child = SovereignCell(cell_id + "_R", self.depth - 1)
        else:
            self.left_child = None
            self.right_child = None

    def execute_pipeline(self, x, y, z):
        """
        Embeds the exact Layered Lane Pipeline mapped previously.
        Calculates all variables deterministically.
        """
        x &= 0xFFFFFFFF
        y &= 0xFFFFFFFF
        z &= 0xFFFFFFFF
        
        # ----------------------------------------------------------------
        # LAYER 1: First-Pass Raw Inputs and Rotations
        # ----------------------------------------------------------------
        ch_and_1 = x & y
        not_x = (~x) & 0xFFFFFFFF
        maj_and_1 = x & y
        maj_and_2 = x & z
        maj_and_3 = y & z
        
        s0_rot_2 = rotr(x, 2)
        s0_rot_13 = rotr(x, 13)
        s0_rot_22 = rotr(x, 22)
        
        s1_rot_6 = rotr(x, 6)
        s1_rot_11 = rotr(x, 11)
        s1_rot_25 = rotr(x, 25)
        
        sig0_rot_7 = rotr(x, 7)
        sig0_rot_18 = rotr(x, 18)
        sig0_shr_3 = shr(x, 3)
        
        sig1_rot_17 = rotr(x, 17)
        sig1_rot_19 = rotr(x, 19)
        sig1_shr_10 = shr(x, 10)
        
        # ----------------------------------------------------------------
        # LAYER 2: First phase Variable XOR Aggregations & Remaining ANDs
        # ----------------------------------------------------------------
        ch_and_2 = not_x & z
        maj_xor_1 = maj_and_1 ^ maj_and_2
        
        s0_xor_1 = s0_rot_2 ^ s0_rot_13
        s1_xor_1 = s1_rot_6 ^ s1_rot_11
        
        sig0_xor_1 = sig0_rot_7 ^ sig0_rot_18
        sig1_xor_1 = sig1_rot_17 ^ sig1_rot_19
        
        # ----------------------------------------------------------------
        # LAYER 3: Output Variable Result Collapse
        # ----------------------------------------------------------------
        ch_out = ch_and_1 ^ ch_and_2
        maj_out = maj_xor_1 ^ maj_and_3
        
        sigma0_out = s0_xor_1 ^ s0_rot_22
        sigma1_out = s1_xor_1 ^ s1_rot_25
        
        low_sigma0_out = sig0_xor_1 ^ sig0_shr_3
        low_sigma1_out = sig1_xor_1 ^ sig1_shr_10
        
        return ch_out, maj_out, sigma0_out, sigma1_out, low_sigma0_out, low_sigma1_out
        
    def cascade(self, x, y, z):
        """
        Executes internal pipeline and pushes output parameters downward 
        directly into the downstream descendant cell geometries.
        """
        print(f"[{self.cell_id}] Executing SHA-256 Pipeline Lane Map (Depth {self.depth})...")
        
        # 1. Pipeline Execution
        ch_out, maj_out, sigma0_out, sigma1_out, low_sigma0_out, low_sigma1_out = self.execute_pipeline(x, y, z)
        
        # 2. Splitting Logic (3 lanes to Left Child, 3 lanes to Right Child)
        if self.left_child:
            self.left_child.cascade(ch_out, maj_out, sigma0_out)
            
        if self.right_child:
            self.right_child.cascade(sigma1_out, low_sigma0_out, low_sigma1_out)
            
        return (ch_out, maj_out, sigma0_out, sigma1_out, low_sigma0_out, low_sigma1_out)
