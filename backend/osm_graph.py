"""
osm_graph.py — Real OSM City Graph Loader for RoadRishi
Fetches live OpenStreetMap road networks for Indian cities,
converts them to the internal healing.py-compatible format,
and provides three disaster simulation modes.
"""

import os
import json
import math
import random
import networkx as nx

# Optional imports — install via: pip install osmnx geopandas
try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# City configuration defaults
# ---------------------------------------------------------------------------
CITY_CONFIGS = {
    "Bengaluru": {
        "query": "Bengaluru, Karnataka, India",
        "bbox": (12.834, 77.491, 13.139, 77.748),   # (south, west, north, east)
        "center": (12.9716, 77.5946),
    },
    "Chennai": {
        "query": "Chennai, Tamil Nadu, India",
        "bbox": (12.901, 80.148, 13.234, 80.329),
        "center": (13.0827, 80.2707),
    },
    "Hyderabad": {
        "query": "Hyderabad, Telangana, India",
        "bbox": (17.270, 78.274, 17.594, 78.637),
        "center": (17.3850, 78.4867),
    },
    "Pune": {
        "query": "Pune, Maharashtra, India",
        "bbox": (18.421, 73.742, 18.634, 74.026),
        "center": (18.5204, 73.8567),
    },
    "Mumbai": {
        "query": "Mumbai, Maharashtra, India",
        "bbox": (18.896, 72.776, 19.269, 72.987),
        "center": (19.0760, 72.8777),
    },
}

# Default fallback cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "osm")


class OSMGraphLoader:
    """
    Loads real OSM road graphs for Indian cities and provides:
    - Conversion to the internal networkx format used by healing.py
    - Three disaster simulation modes (random, bridge, cascade)
    - Rasterization to pixel-mask for SegFormer overlay
    """

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: fetch or load a city graph
    # ------------------------------------------------------------------
    def fetch_city_graph(
        self,
        city_name: str = "Bengaluru",
        network_type: str = "drive",
        use_cache: bool = True,
    ) -> nx.MultiDiGraph:
        """
        Fetches the OSM driveable road network for *city_name*.
        Caches the result locally as a GraphML file to avoid repeated downloads.

        Returns a networkx MultiDiGraph with node attributes:
          x (longitude), y (latitude), osmid
        and edge attributes:
          length (metres), highway, name, geometry (if available)
        """
        if not OSMNX_AVAILABLE:
            raise RuntimeError(
                "osmnx is not installed. Run: pip install osmnx"
            )

        cache_file = os.path.join(
            self.cache_dir, f"{city_name.lower()}_{network_type}.graphml"
        )

        if use_cache and os.path.exists(cache_file):
            print(f"[OSMGraphLoader] Loading cached graph: {cache_file}")
            G = ox.load_graphml(cache_file)
            return G

        config = CITY_CONFIGS.get(city_name, CITY_CONFIGS["Bengaluru"])
        print(f"[OSMGraphLoader] Downloading OSM graph for {city_name} ...")

        G = ox.graph_from_place(
            config["query"],
            network_type=network_type,
            simplify=True,
        )

        # Save cache
        ox.save_graphml(G, cache_file)
        print(f"[OSMGraphLoader] Saved to cache: {cache_file}")
        return G

    # ------------------------------------------------------------------
    # Public: convert to healing.py-compatible undirected graph
    # ------------------------------------------------------------------
    def to_networkx(self, G_osm: nx.MultiDiGraph) -> tuple[nx.Graph, dict]:
        """
        Converts an OSM MultiDiGraph to the simple undirected Graph format
        expected by GraphHealingEngine in healing.py.

        Returns:
          G        — undirected nx.Graph with 'weight' edge attribute (metres)
          pos_dict — {node_id: (lat, lon)} mapping  (row-col style for healing.py)
        """
        # Project to UTM so distances are in metres
        if OSMNX_AVAILABLE:
            G_proj = ox.project_graph(G_osm)
        else:
            G_proj = G_osm

        G = nx.Graph()
        pos_dict = {}

        for node, data in G_proj.nodes(data=True):
            # Use projected x/y (metres from origin) for spatial computations
            x = data.get("x", 0.0)
            y = data.get("y", 0.0)
            # Store as (row=y, col=x) to match healing.py convention
            pos_dict[node] = (y, x)
            G.add_node(node, pos=(y, x), osmid=data.get("osmid", node))

        for u, v, data in G_proj.edges(data=True):
            length = data.get("length", 1.0)
            G.add_edge(u, v, weight=length)

        return G, pos_dict

    # ------------------------------------------------------------------
    # Public: rasterize road graph to a binary pixel mask
    # ------------------------------------------------------------------
    def rasterize_to_mask(
        self,
        G: nx.Graph,
        pos_dict: dict,
        shape: tuple[int, int] = (512, 512),
        line_thickness: int = 2,
    ):
        """
        Converts the road graph to a binary NumPy mask of given *shape*.
        Used to overlay the OSM ground-truth on the SegFormer prediction.

        Returns:
          mask — np.ndarray of shape (H, W) with dtype float32, values in [0, 1]
        """
        if not NUMPY_AVAILABLE:
            raise RuntimeError("numpy is required for rasterization")

        import numpy as np

        H, W = shape
        mask = np.zeros((H, W), dtype=np.float32)

        # Determine coordinate bounds
        ys = [p[0] for p in pos_dict.values()]
        xs = [p[1] for p in pos_dict.values()]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        y_range = max(y_max - y_min, 1.0)
        x_range = max(x_max - x_min, 1.0)

        def to_pixel(y, x):
            row = int((y - y_min) / y_range * (H - 1))
            col = int((x - x_min) / x_range * (W - 1))
            return (
                max(0, min(H - 1, row)),
                max(0, min(W - 1, col)),
            )

        def bresenham(r0, c0, r1, c1):
            """Bresenham's line algorithm."""
            points = []
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            sr = 1 if r0 < r1 else -1
            sc = 1 if c0 < c1 else -1
            err = dr - dc
            while True:
                points.append((r0, c0))
                if r0 == r1 and c0 == c1:
                    break
                e2 = 2 * err
                if e2 > -dc:
                    err -= dc
                    r0 += sr
                if e2 < dr:
                    err += dr
                    c0 += sc
            return points

        for u, v in G.edges():
            if u not in pos_dict or v not in pos_dict:
                continue
            r0, c0 = to_pixel(*pos_dict[u])
            r1, c1 = to_pixel(*pos_dict[v])
            for r, c in bresenham(r0, c0, r1, c1):
                for dr in range(-line_thickness, line_thickness + 1):
                    for dc in range(-line_thickness, line_thickness + 1):
                        rr = max(0, min(H - 1, r + dr))
                        cc = max(0, min(W - 1, c + dc))
                        mask[rr, cc] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Public: disaster simulation
    # ------------------------------------------------------------------
    def simulate_disaster(
        self,
        G: nx.Graph,
        pct: float = 0.3,
        mode: str = "random",
        seed: int = 42,
    ) -> nx.Graph:
        """
        Simulates a disaster by removing a fraction *pct* of edges/nodes.

        Modes:
          "random"   — uniform random edge removal (flood / earthquake)
          "bridge"   — removes highest-betweenness edges first (targeted attack)
          "cascade"  — iterative removal triggering LCC fragmentation
          "node"     — removes highest-degree nodes (hub disruption)

        Returns a copy of *G* with edges/nodes removed.
        """
        random.seed(seed)
        G_disrupted = G.copy()
        edges = list(G_disrupted.edges())
        n_remove = max(1, int(len(edges) * pct))

        if mode == "random":
            to_remove = random.sample(edges, min(n_remove, len(edges)))
            G_disrupted.remove_edges_from(to_remove)

        elif mode == "bridge":
            # Betweenness-weighted removal (highest-impact edges first)
            try:
                bc = nx.edge_betweenness_centrality(G_disrupted, weight="weight")
                sorted_edges = sorted(bc.keys(), key=lambda e: bc[e], reverse=True)
                G_disrupted.remove_edges_from(sorted_edges[:n_remove])
            except Exception:
                # Fallback to random
                to_remove = random.sample(edges, min(n_remove, len(edges)))
                G_disrupted.remove_edges_from(to_remove)

        elif mode == "cascade":
            # Iterative removal: always remove the current highest-BC edge
            # until pct of original edges are gone
            removed = 0
            while removed < n_remove and G_disrupted.number_of_edges() > 0:
                try:
                    bc = nx.edge_betweenness_centrality(G_disrupted, weight="weight")
                    if not bc:
                        break
                    worst_edge = max(bc, key=lambda e: bc[e])
                    G_disrupted.remove_edge(*worst_edge)
                    removed += 1
                except Exception:
                    break

        elif mode == "node":
            # Remove highest-degree nodes
            nodes = list(G_disrupted.nodes())
            n_nodes_remove = max(1, int(len(nodes) * pct * 0.3))
            degree_sorted = sorted(
                G_disrupted.degree(), key=lambda x: x[1], reverse=True
            )
            to_remove_nodes = [n for n, _ in degree_sorted[:n_nodes_remove]]
            G_disrupted.remove_nodes_from(to_remove_nodes)

        # Clean up isolated single nodes
        isolates = list(nx.isolates(G_disrupted))
        G_disrupted.remove_nodes_from(isolates)

        return G_disrupted

    # ------------------------------------------------------------------
    # Public: serialize graph to API-compatible dict
    # ------------------------------------------------------------------
    def serialize_graph_for_api(
        self,
        G: nx.Graph,
        pos_dict: dict,
        node_stats: dict = None,
        city_name: str = "Bengaluru",
    ) -> dict:
        """
        Serializes the graph to a JSON-compatible dict matching the
        schema used by the existing /api/pipeline endpoint in main.py.
        Lat/lon values are used as x/y so Leaflet can render them directly.
        """
        config = CITY_CONFIGS.get(city_name, CITY_CONFIGS["Bengaluru"])

        nodes_list = []
        for n in G.nodes():
            if n not in pos_dict:
                continue
            y, x = pos_dict[n]  # y=lat, x=lon in projected metres
            # Convert back to approximate lat/lon for Leaflet
            lat, lon = _metres_to_latlon(y, x, config["center"])
            nodes_list.append(
                {
                    "id": str(n),
                    "y": lat,
                    "x": lon,
                    "quadrant": (node_stats or {}).get(n, {}).get("quadrant", "SAFE"),
                    "meanBC": (node_stats or {}).get(n, {}).get("meanBC", 0.0),
                    "stdBC": (node_stats or {}).get(n, {}).get("stdBC", 0.0),
                }
            )

        edges_list = []
        for u, v, d in G.edges(data=True):
            if u not in pos_dict or v not in pos_dict:
                continue
            u_lat, u_lon = _metres_to_latlon(*pos_dict[u], config["center"])
            v_lat, v_lon = _metres_to_latlon(*pos_dict[v], config["center"])
            edges_list.append(
                {
                    "u": str(u),
                    "v": str(v),
                    "u_pos": [u_lat, u_lon],
                    "v_pos": [v_lat, v_lon],
                    "healed": d.get("healed", False),
                    "weight": round(d.get("weight", 1.0), 1),
                }
            )

        return {"nodes": nodes_list, "edges": edges_list}


# ---------------------------------------------------------------------------
# Helper: rough metres-to-latlon conversion
# ---------------------------------------------------------------------------
def _metres_to_latlon(
    y_m: float, x_m: float, center_latlon: tuple[float, float]
) -> tuple[float, float]:
    """
    Approximate inverse UTM projection centered on *center_latlon*.
    Suitable for Leaflet display (not for precise geodesy).
    """
    lat0, lon0 = center_latlon
    # 1 degree latitude ≈ 111,320 m
    # 1 degree longitude ≈ 111,320 * cos(lat) m
    lat = lat0 + y_m / 111_320.0
    lon = lon0 + x_m / (111_320.0 * math.cos(math.radians(lat0)))
    return round(lat, 6), round(lon, 6)


# ---------------------------------------------------------------------------
# Lightweight fallback: synthetic Bengaluru-style grid graph
# (used when osmnx is not installed or network is unavailable)
# ---------------------------------------------------------------------------
def build_synthetic_city_graph(
    city_name: str = "Bengaluru",
    grid_rows: int = 10,
    grid_cols: int = 10,
) -> tuple[nx.Graph, dict]:
    """
    Builds a realistic grid+diagonal graph centred on the city.
    Used as a fallback when osmnx is unavailable.
    Returns (G, pos_dict) in the same format as OSMGraphLoader.to_networkx().
    """
    config = CITY_CONFIGS.get(city_name, CITY_CONFIGS["Bengaluru"])
    lat0, lon0 = config["center"]

    # Grid spacing in metres (~500 m blocks)
    spacing = 500.0

    G = nx.Graph()
    pos_dict = {}

    def node_id(r, c):
        return f"n_{r}_{c}"

    # Build grid nodes
    for r in range(grid_rows):
        for c in range(grid_cols):
            nid = node_id(r, c)
            y_m = (r - grid_rows // 2) * spacing
            x_m = (c - grid_cols // 2) * spacing
            pos_dict[nid] = (y_m, x_m)
            G.add_node(nid, pos=(y_m, x_m))

    # Horizontal and vertical edges
    for r in range(grid_rows):
        for c in range(grid_cols):
            if c + 1 < grid_cols:
                u, v = node_id(r, c), node_id(r, c + 1)
                G.add_edge(u, v, weight=spacing)
            if r + 1 < grid_rows:
                u, v = node_id(r, c), node_id(r + 1, c)
                G.add_edge(u, v, weight=spacing)

    # Diagonal arterials (simulate ring roads)
    for r in range(grid_rows - 1):
        for c in range(grid_cols - 1):
            if (r + c) % 3 == 0:
                u, v = node_id(r, c), node_id(r + 1, c + 1)
                G.add_edge(u, v, weight=round(spacing * 1.414, 1))

    return G, pos_dict
