import sovereign_cell

if __name__ == "__main__":
    print("==========================================================")
    print(" FRACTAL SOVEREIGN CELL INITIALIZATION TIER ")
    print("==========================================================")
    
    # Establish a depth-2 cascade:
    # Root (Depth 2) -> 2 Children (Depth 1) -> 4 Grandchildren (Depth 0)
    # Total distinct processing cell geometries operating in tandem = 7
    root = sovereign_cell.SovereignCell(cell_id="Root", cascade_depth=2)
    
    # Input triggers matching arbitrary standard SHA-256 initialization bits
    initial_x = 0x6a09e667
    initial_y = 0xbb67ae85
    initial_z = 0x3c6ef372
    
    print(f"\n[System] Hitting Master Root Pipeline Cell...")
    print(f"Input Stream -> x: 0x{initial_x:08X}, y: 0x{initial_y:08X}, z: 0x{initial_z:08X}\n")
    
    # Fire the cascade
    final_outputs = root.cascade(initial_x, initial_y, initial_z)
    
    print("\n[System] Fractal Sovereign Cascade Terminated.")
