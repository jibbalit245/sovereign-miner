import random
import math

class GeometryOptimizer:
    def __init__(self, nodes, grid_w, grid_h):
        """
        nodes: Dict[node_id] -> {'op': str, 'deps': List[str]}
        grid_w, grid_h: width and height of the target PE mesh.
        """
        self.nodes = nodes
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.placement = {} # { node_id: (x, y) }
        self.grid = {}      # { (x, y): node_id }

    def initial_placement(self):
        cells = [(x,y) for x in range(self.grid_w) for y in range(self.grid_h)]
        random.shuffle(cells)
        for i, node in enumerate(self.nodes.keys()):
            pos = cells[i]
            self.placement[node] = pos
            self.grid[pos] = node
            
    def calculate_cost(self, current_placement):
        # Cost is sum of exact Manhattan distances across dependencies 
        # (Wire Length mapping cost). Shorter wires = faster routing.
        cost = 0
        for node, info in self.nodes.items():
            pos1 = current_placement.get(node)
            if not pos1: continue
            for dep in info['deps']:
                pos2 = current_placement.get(dep)
                if pos2:
                    dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                    cost += dist
        return cost

    def optimize(self, initial_temp=100.0, cooling_rate=0.99, iters=10000):
        self.initial_placement()
        current_cost = self.calculate_cost(self.placement)
        best_placement = self.placement.copy()
        best_cost = current_cost
        
        temp = initial_temp
        
        for i in range(iters):
            # Propose a neighbor state
            node_to_move = random.choice(list(self.nodes.keys()))
            old_pos = self.placement[node_to_move]
            
            rx = random.randint(0, self.grid_w - 1)
            ry = random.randint(0, self.grid_h - 1)
            new_pos = (rx, ry)
            
            if new_pos == old_pos:
                continue
                
            proposed_placement = self.placement.copy()
            
            # Allow swapping with another node
            node_at_new = self.grid.get(new_pos)
            
            proposed_placement[node_to_move] = new_pos
            if node_at_new:
                proposed_placement[node_at_new] = old_pos
                
            new_cost = self.calculate_cost(proposed_placement)
            
            # Acceptance criteria
            if new_cost < current_cost:
                accept = True
            else:
                try:
                    prob = math.exp(-(new_cost - current_cost) / temp)
                except OverflowError:
                    prob = 0
                accept = random.random() < prob
                
            if accept:
                self.placement = proposed_placement
                self.grid = {p: n for n, p in self.placement.items()}
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_placement = self.placement.copy()
            
            temp *= cooling_rate
            
        return best_placement, best_cost
