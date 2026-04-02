def print_geometry(grid_w, grid_h, placement, nodes):
    # Reverse placement string
    grid = {pos: n for n, pos in placement.items()}
        
    for y in range(grid_h):
        row_str = "|"
        for x in range(grid_w):
            node = grid.get((x,y))
            if node:
                op = nodes[node].get('op', 'in')
                cell = f"{node}({op})"
            else:
                cell = " - "
            row_str += cell.center(12) + "|"
        print("-" * (grid_w * 13 + 1))
        print(row_str)
    print("-" * (grid_w * 13 + 1))
    
    print("\nLegend:")
    for n, p in placement.items():
        deps = nodes[n]['deps']
        if deps:
            for d in deps:
                d_pos = placement.get(d)
                if d_pos:
                    dist = abs(p[0]-d_pos[0]) + abs(p[1]-d_pos[1])
                    print(f"Routing {d} -> {n} (Distance: {dist})")
