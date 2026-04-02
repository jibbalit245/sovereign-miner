def parse_algorithm(text):
    """
    Parses an algorithm into a Dependency Graph.
    Supports binary (a OP b) and unary (OP a) assignments.
    """
    nodes = {}
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split('=')
        if len(parts) != 2: continue
        
        target = parts[0].strip()
        expr = parts[1].strip()
        
        tokens = expr.split()
        if len(tokens) == 3:
            dep1, op, dep2 = tokens
            nodes[target] = {'op': op, 'deps': [dep1, dep2]}
            # ensure input parameters exist as root nodes
            if dep1 not in nodes: nodes[dep1] = {'op': 'in', 'deps': []}
            if dep2 not in nodes: nodes[dep2] = {'op': 'in', 'deps': []}
        elif len(tokens) == 2:
            op, dep1 = tokens
            nodes[target] = {'op': op, 'deps': [dep1]}
            if dep1 not in nodes: nodes[dep1] = {'op': 'in', 'deps': []}
        elif len(tokens) == 1:
            dep1 = tokens[0]
            nodes[target] = {'op': '=', 'deps': [dep1]}
            if dep1 not in nodes: nodes[dep1] = {'op': 'in', 'deps': []}
            
    return nodes
