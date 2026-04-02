import json

def compute_layered_lanes(nodes):
    """
    Computes a topological pipeline geometry (Layered Lanes) for the Data Flow Graph.
    Operations are pushed into sequential layers based on their dependencies.
    """
    layers = {} # node_id -> layer_index
    
    def get_depth(node):
        if node in layers:
            return layers[node]
        info = nodes.get(node)
        if not info or not info.get('deps'):
            layers[node] = 0
            return 0
            
        max_d = -1
        for dep in info['deps']:
            d = get_depth(dep)
            if d > max_d:
                max_d = d
                
        depth = max_d + 1
        layers[node] = depth
        return depth
        
    for n in nodes.keys():
        get_depth(n)
        
    max_layer = max(layers.values()) if layers else 0
    layered_lanes = [[] for _ in range(max_layer + 1)]
    
    for n, l in layers.items():
        layered_lanes[l].append(n)
        
    return layered_lanes, layers

def print_layered_geometry(layered_lanes, nodes):
    print("\n==============================================")
    print(" PIPELINE GEOMETRY (LAYERED LANES) ")
    print("==============================================")
    
    max_width = max(len(lane) for lane in layered_lanes)
    print(f"Geometry Profile: {len(layered_lanes)} Layers Deep x {max_width} Lanes Max Width\n")
    
    map_data = {}
    
    for layer_idx, lane in enumerate(layered_lanes):
        # Format for terminal visual
        visual_str = f"Layer {layer_idx:02d} | "
        for n in lane:
            op = nodes[n].get('op', 'in')
            visual_str += f"[{n}({op})]  "
        print(visual_str)
        
        # Build mapping data
        map_data[f"layer_{layer_idx}"] = []
        for lane_idx, n in enumerate(lane):
            op = nodes[n].get('op', 'in')
            deps = nodes[n].get('deps', [])
            map_data[f"layer_{layer_idx}"].append({
                "id": n,
                "type": op,
                "lane": lane_idx,
                "dependencies": deps
            })
            
    print("-" * 50)
    print("\nRAW MAPPING DATA (JSON format):")
    print(json.dumps(map_data, indent=2))
