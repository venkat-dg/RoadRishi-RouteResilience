import numpy as np
import networkx as nx

class UnionFind:
    """Simple Union-Find (Disjoint Set Union) data structure for Kruskal's MST."""
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
        self.rank = {el: 0 for el in elements}

    def find(self, el):
        if self.parent[el] != el:
            self.parent[el] = self.find(self.parent[el])
        return self.parent[el]

    def union(self, el1, el2):
        root1 = self.find(el1)
        root2 = self.find(el2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
            return True
        return False


class GraphHealingEngine:
    """
    Graph-Healing Post-Processing Engine.
    Converts model outputs into spatial graphs, fractures them based on occlusions,
    and applies a customized Union-Find & MST routing routine using the composite cost formula.
    """
    def __init__(self, w1=1.0, w2=20.0, w3=5.0, w4=10.0, max_heal_dist=120.0):
        self.w1 = w1  # Distance weight
        self.w2 = w2  # Angular alignment weight
        self.w3 = w3  # Road probability path integral weight
        self.w4 = w4  # Attention/probability priority weight
        self.max_heal_dist = max_heal_dist

    def build_initial_graph(self, road_lines, img_shape):
        """Builds a spatial graph from predefined road lines."""
        G = nx.Graph()
        node_id_counter = 0
        node_positions = {}
        
        # Helper to find or add node at position
        def get_or_create_node(pos):
            nonlocal node_id_counter
            for nid, npos in node_positions.items():
                if np.linalg.norm(np.array(npos) - np.array(pos)) < 5.0:
                    return nid
            nid = f"n_{node_id_counter}"
            node_id_counter += 1
            node_positions[nid] = pos
            G.add_node(nid, pos=pos)
            return nid

        # Break lines into segments and add to graph
        for start, end in road_lines:
            n1 = get_or_create_node(start)
            n2 = get_or_create_node(end)
            
            # Subdivide edges for high spatial resolution
            pos1 = np.array(node_positions[n1])
            pos2 = np.array(node_positions[n2])
            dist = np.linalg.norm(pos1 - pos2)
            steps = int(dist / 40.0) # Subdivide every 40 pixels
            
            if steps > 1:
                prev_node = n1
                for step in range(1, steps):
                    fraction = step / steps
                    mid_pos = tuple(pos1 + fraction * (pos2 - pos1))
                    mid_node = get_or_create_node(mid_pos)
                    G.add_edge(prev_node, mid_node, weight=np.linalg.norm(np.array(node_positions[prev_node]) - np.array(mid_pos)))
                    prev_node = mid_node
                G.add_edge(prev_node, n2, weight=np.linalg.norm(np.array(node_positions[prev_node]) - pos2))
            else:
                G.add_edge(n1, n2, weight=dist)
                
        return G, node_positions

    def fracture_graph(self, G, node_positions, occlusion_mask):
        """
        Simulates model segmentation failure.
        Deletes nodes and edges that lie inside the environmental occlusions (canopy/shadows),
        resulting in fragmented road segments with dangling endpoints.
        """
        disrupted_G = G.copy()
        
        # Find nodes inside occlusions
        nodes_to_remove = []
        for node in disrupted_G.nodes():
            y, x = map(int, node_positions[node])
            if 0 <= y < occlusion_mask.shape[0] and 0 <= x < occlusion_mask.shape[1]:
                if occlusion_mask[y, x] > 0.5:
                    nodes_to_remove.append(node)
                    
        # Remove nodes
        disrupted_G.remove_nodes_from(nodes_to_remove)
        
        # Also remove edges whose midpoint lies in occlusions
        edges_to_remove = []
        for u, v in disrupted_G.edges():
            pos_u = np.array(node_positions[u])
            pos_v = np.array(node_positions[v])
            midpoint = (pos_u + pos_v) / 2.0
            my, mx = map(int, midpoint)
            if 0 <= my < occlusion_mask.shape[0] and 0 <= mx < occlusion_mask.shape[1]:
                if occlusion_mask[my, mx] > 0.5:
                    edges_to_remove.append((u, v))
                    
        disrupted_G.remove_edges_from(edges_to_remove)
        
        # Keep only components with > 1 node to clean isolated single-node noise
        small_components = [c for c in nx.connected_components(disrupted_G) if len(c) <= 1]
        for comp in small_components:
            disrupted_G.remove_nodes_from(comp)
            
        return disrupted_G

    def compute_edge_direction(self, G, node, pos_dict):
        """Estimates the direction vector of the road leading to a dangling node."""
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return np.array([0.0, 0.0])
        
        # Take the vector from neighbor to node
        pos_node = np.array(pos_dict[node])
        pos_neigh = np.array(pos_dict[neighbors[0]])
        vec = pos_node - pos_neigh
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else np.array([0.0, 0.0])

    def calculate_alignment(self, pos_i, dir_i, pos_j, dir_j):
        """
        Computes angular alignment cost.
        If the displacement vector matches the road directions, return a low cost.
        """
        disp = np.array(pos_j) - np.array(pos_i)
        disp_norm = np.linalg.norm(disp)
        if disp_norm == 0:
            return 0.0
        disp_dir = disp / disp_norm
        
        # Cosine alignment of node directions with the connection line
        cos_i = np.dot(dir_i, disp_dir)
        cos_j = np.dot(dir_j, -disp_dir)
        
        # We want the roads to point towards each other.
        # Max alignment means cos_i -> 1 and cos_j -> 1.
        # Angular cost = 1.0 - (cos_i + cos_j) / 2.0
        # Clip to [0, 2]
        angular_penalty = 1.0 - (cos_i + cos_j) / 2.0
        return max(0.0, angular_penalty)

    def calculate_path_integral(self, pos_i, pos_j, prob_map):
        """
        Computes the path integral of (1 - P_road) along the line between i and j.
        """
        y1, x1 = pos_i
        y2, x2 = pos_j
        
        # Sample 10 points along the segment
        num_samples = 15
        ys = np.linspace(y1, y2, num_samples)
        xs = np.linspace(x1, x2, num_samples)
        
        integral = 0.0
        h, w = prob_map.shape
        for y, x in zip(ys, xs):
            iy, ix = int(clip(y, 0, h - 1)), int(clip(x, 0, w - 1))
            # Integral sum of (1 - P_road)
            integral += (1.0 - prob_map[iy, ix])
            
        return integral / num_samples

    def heal_graph(self, disrupted_G, pos_dict, prob_map, attention_priors):
        """
        Graph-Healing using Union-Find & MST routing.
        Finds dangling nodes (degree == 1) and reconnects them based on:
        C(e_ij) = w1 * dist + w2 * alignment_penalty + w3 * path_integral + w4 * (1 - pe)
        """
        healed_G = disrupted_G.copy()
        
        # Identify dangling nodes (degree == 1)
        dangling_nodes = [node for node, deg in healed_G.degree() if deg == 1]
        
        if len(dangling_nodes) < 2:
            return healed_G, []

        # Extract direction vectors for each dangling node
        directions = {}
        for node in dangling_nodes:
            directions[node] = self.compute_edge_direction(healed_G, node, pos_dict)

        # Form candidate healing connections
        candidates = []
        for i in range(len(dangling_nodes)):
            node_i = dangling_nodes[i]
            pos_i = pos_dict[node_i]
            dir_i = directions[node_i]
            
            for j in range(i + 1, len(dangling_nodes)):
                node_j = dangling_nodes[j]
                pos_j = pos_dict[node_j]
                dir_j = directions[node_j]
                
                # 1. Euclidean distance
                dist = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
                if dist > self.max_heal_dist:
                    continue
                
                # 2. Angular alignment
                alignment = self.calculate_alignment(pos_i, dir_i, pos_j, dir_j)
                
                # 3. Path integral of (1 - P_road)
                integral = self.calculate_path_integral(pos_i, pos_j, prob_map)
                
                # 4. Attention overlap priority (pe)
                # Check if the connection crosses any known attention bridge
                pe = 0.0
                for prior in attention_priors:
                    center = np.array(prior["center"])
                    radius = prior["radius"]
                    # If both nodes are within the attention canopy zone, they are associated
                    dist_i = np.linalg.norm(np.array(pos_i) - center)
                    dist_j = np.linalg.norm(np.array(pos_j) - center)
                    if dist_i <= radius * 1.5 and dist_j <= radius * 1.5:
                        # High calibrated edge existence probability
                        pe = 0.85
                        break
                        
                # Compute composite cost
                cost = (self.w1 * dist + 
                        self.w2 * alignment + 
                        self.w3 * integral + 
                        self.w4 * (1.0 - pe))
                
                candidates.append({
                    "u": node_i,
                    "v": node_j,
                    "cost": cost,
                    "length": dist,
                    "pe": pe if pe > 0 else 0.15 # fallback low probability
                })

        # Sort candidate connections by cost (lowest first)
        candidates.sort(key=lambda x: x["cost"])

        # Run Kruskal's/Union-Find to connect segments without cycles
        uf = UnionFind(healed_G.nodes())
        
        # Initialize union-find with current connected components
        components = list(nx.connected_components(healed_G))
        for comp in components:
            comp_list = list(comp)
            first = comp_list[0]
            for node in comp_list[1:]:
                uf.union(first, node)

        healed_edges = []
        for cand in candidates:
            u = cand["u"]
            v = cand["v"]
            
            # Connect only if it bridges different components (MST principle)
            if uf.union(u, v):
                healed_G.add_edge(u, v, weight=cand["length"], healed=True, pe=cand["pe"])
                healed_edges.append({
                    "edge_id": f"healed_{u}_{v}",
                    "u": u,
                    "v": v,
                    "u_pos": pos_dict[u],
                    "v_pos": pos_dict[v],
                    "length": float(round(cand["length"], 1)),
                    "pe": float(round(cand["pe"], 2)),
                    "cost": float(round(cand["cost"], 2))
                })

        return healed_G, healed_edges


def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))
