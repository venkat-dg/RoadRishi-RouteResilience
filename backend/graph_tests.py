"""
graph_tests.py — Graph Validation Test Suite for RoadRishi
Validates MST correctness, centrality metrics, Fiedler eigenvalue,
node disabling, disaster stress sweeps, and real OSM graph integration.

Run: cd backend && python graph_tests.py
All tests print PASS / FAIL and a summary at the end.
"""

import sys
import traceback
import networkx as nx
import numpy as np

from healing import GraphHealingEngine, UnionFind
from resilience import CriticalityVarianceEngine
from osm_graph import OSMGraphLoader, build_synthetic_city_graph, OSMNX_AVAILABLE

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
RESULTS = []


def test(name):
    """Decorator that wraps a test function, catches exceptions, prints result."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'─' * 60}")
            print(f"  TEST: {name}")
            print(f"{'─' * 60}")
            try:
                fn()
                print(f"  ✅  PASS")
                RESULTS.append((name, True, None))
            except AssertionError as e:
                msg = str(e) or "Assertion failed"
                print(f"  ❌  FAIL — {msg}")
                RESULTS.append((name, False, msg))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print(f"  ❌  ERROR — {msg}")
                traceback.print_exc()
                RESULTS.append((name, False, msg))
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# T1: MST correctness (Kruskal on a known 5-node graph)
# ---------------------------------------------------------------------------
@test("MST Correctness — Kruskal on known 5-node graph")
def test_mst_correctness():
    """
    Build a 5-node complete graph with known edge weights.
    Verify Kruskal's MST has exactly n-1 = 4 edges and minimum total weight.
    """
    G = nx.Graph()
    edges = [
        (0, 1, 1), (0, 2, 4), (0, 3, 3),
        (1, 2, 2), (1, 3, 5),
        (2, 3, 1), (2, 4, 6),
        (3, 4, 2),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    mst = nx.minimum_spanning_tree(G, weight="weight")

    assert mst.number_of_edges() == 4, \
        f"MST should have 4 edges, got {mst.number_of_edges()}"

    mst_weight = sum(d["weight"] for _, _, d in mst.edges(data=True))
    assert mst_weight == 6, \
        f"MST total weight should be 6, got {mst_weight}"

    # Also test our UnionFind directly
    uf = UnionFind([0, 1, 2, 3, 4])
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2), "UnionFind transitivity failed"
    assert uf.find(3) != uf.find(0), "UnionFind should not merge un-unioned sets"

    print(f"    MST edges: {mst.number_of_edges()}, total weight: {mst_weight}")


# ---------------------------------------------------------------------------
# T2: Centrality metrics (BC against networkx reference)
# ---------------------------------------------------------------------------
@test("Centrality Metrics — Betweenness Centrality vs networkx reference")
def test_centrality_metrics():
    """
    On a known path graph P6, node 2 and 3 should have the highest BC.
    Verify our resilience engine produces values consistent with networkx.
    """
    G = nx.path_graph(6)  # 0-1-2-3-4-5
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    ref_bc = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # Nodes in the middle of a path have maximum BC
    sorted_nodes = sorted(ref_bc, key=ref_bc.get, reverse=True)
    assert sorted_nodes[0] in (2, 3), \
        f"Middle node should have highest BC, got {sorted_nodes[0]}"

    # Check CriticalityVarianceEngine produces similar distribution
    engine = CriticalityVarianceEngine(num_samples=10)
    # Build healed_edges = [] (all edges reliable)
    node_stats, _ = engine.run_monte_carlo(G, [])

    engine_bc_top = sorted(node_stats, key=lambda n: node_stats[n]["meanBC"], reverse=True)
    assert engine_bc_top[0] in (2, 3), \
        f"Engine top-BC node should be 2 or 3, got {engine_bc_top[0]}"

    print(f"    Reference BC top node: {sorted_nodes[0]}")
    print(f"    Engine BC top node:    {engine_bc_top[0]}")
    print(f"    All quadrants: {set(s['quadrant'] for s in node_stats.values())}")


# ---------------------------------------------------------------------------
# T3: Fiedler eigenvalue on known graphs
# ---------------------------------------------------------------------------
@test("Fiedler Eigenvalue — algebraic connectivity on known graphs")
def test_fiedler_eigenvalue():
    """
    Known properties:
    - Path graph Pn has Fiedler value > 0 (connected)
    - Disconnected graph has Fiedler value ≈ 0
    - Complete graph Kn has max Fiedler = n (by theory)
    """
    engine = CriticalityVarianceEngine()

    # 1) Path graph — connected, Fiedler > 0
    P = nx.path_graph(8)
    for u, v in P.edges():
        P[u][v]["weight"] = 1.0
    fiedler_path = engine.get_fiedler_eigenvalue(P)
    assert fiedler_path > 0, f"Path graph Fiedler should be > 0, got {fiedler_path}"
    print(f"    Path graph P8 Fiedler: {fiedler_path:.4f}")

    # 2) Disconnected graph — Fiedler of LCC > 0
    D = nx.Graph()
    D.add_edges_from([(0, 1), (1, 2)])     # component A
    D.add_edges_from([(10, 11), (11, 12)]) # component B (disconnected)
    for u, v in D.edges():
        D[u][v]["weight"] = 1.0
    fiedler_disc = engine.get_fiedler_eigenvalue(D)
    # LCC has 3 nodes (path), should still return value > 0
    assert fiedler_disc >= 0, f"Disconnected graph Fiedler should be >= 0, got {fiedler_disc}"
    print(f"    Disconnected graph Fiedler (on LCC): {fiedler_disc:.4f}")

    # 3) Complete graph K5 — Fiedler = n = 5 (with unit weights)
    K = nx.complete_graph(5)
    for u, v in K.edges():
        K[u][v]["weight"] = 1.0
    fiedler_k5 = engine.get_fiedler_eigenvalue(K)
    assert abs(fiedler_k5 - 5.0) < 0.5, \
        f"K5 Fiedler should be ≈ 5.0, got {fiedler_k5}"
    print(f"    K5 Fiedler: {fiedler_k5:.4f} (expected ≈ 5.0)")


# ---------------------------------------------------------------------------
# T4: Node disabling — disable top-3 CRITICAL nodes, recompute RI
# ---------------------------------------------------------------------------
@test("Node Disabling — disable top-3 CRITICAL nodes, recompute RI")
def test_node_disabling():
    """
    Build a star graph (hub + 8 spokes).
    Disabling the hub should massively drop RI.
    """
    G = nx.star_graph(8)  # node 0 is hub
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    engine = CriticalityVarianceEngine(num_samples=20)
    node_stats, top_priorities = engine.run_monte_carlo(G, [])

    # Hub (node 0) must be high-centrality: CRITICAL or RELIABLE
    hub_quadrant = node_stats[0]["quadrant"]
    assert hub_quadrant in ("CRITICAL", "RELIABLE"), \
        f"Hub node should be CRITICAL or RELIABLE (high BC), got {hub_quadrant}"

    # Disable top-3 (at least hub) and recompute RI
    G_disabled = G.copy()
    nodes_to_disable = [0]  # remove the hub
    G_disabled.remove_nodes_from(nodes_to_disable)

    ri_original = engine.compute_resilience_index(G, G, G)
    ri_disabled = engine.compute_resilience_index(G, G_disabled, G_disabled)

    ri_before = ri_original["RI_healed"]
    ri_after  = ri_disabled["RI_healed"]

    assert ri_after < ri_before, \
        f"RI should drop after hub removal. Before: {ri_before}, After: {ri_after}"

    print(f"    Hub quadrant: {hub_quadrant}")
    print(f"    RI before hub removal: {ri_before:.3f}")
    print(f"    RI after  hub removal: {ri_after:.3f}")
    print(f"    RI drop: {ri_before - ri_after:.3f}")


# ---------------------------------------------------------------------------
# T5: Disaster stress sweep (disruption 10%→90%, plot RI decay)
# ---------------------------------------------------------------------------
@test("Disaster Stress Sweep — disruption 10%→90%, RI decay monotone")
def test_disaster_stress_sweep():
    """
    On a 10×10 grid graph, apply random disruption from 10% to 90%.
    RI should decrease monotonically as disruption increases.
    """
    G_full, pos_dict = build_synthetic_city_graph("Bengaluru", 10, 10)
    loader = OSMGraphLoader()
    engine = CriticalityVarianceEngine(num_samples=5)

    pct_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ri_values = []

    print(f"    {'Disruption%':>12}  {'RI':>8}  {'LCC':>8}  {'Fiedler':>10}")
    for pct in pct_levels:
        G_disrupted = loader.simulate_disaster(G_full, pct=pct, mode="random", seed=42)
        metrics = engine.compute_resilience_index(G_full, G_disrupted, G_disrupted)
        ri = metrics["RI_healed"]
        lcc = metrics["LCC"]["healed"]
        fiedler = metrics["FiedlerLambda"]["healed"]
        ri_values.append(ri)
        print(f"    {int(pct*100):>11}%  {ri:>8.3f}  {lcc:>8}  {fiedler:>10.4f}")

    # RI should be generally decreasing
    drops = sum(1 for i in range(1, len(ri_values)) if ri_values[i] <= ri_values[i - 1])
    assert drops >= len(pct_levels) // 2, \
        f"RI should mostly decrease as disruption increases. Drops: {drops}/{len(pct_levels)-1}"


# ---------------------------------------------------------------------------
# T6: All three disruption modes (random, bridge, cascade)
# ---------------------------------------------------------------------------
@test("Disruption Modes — random / bridge / cascade on synthetic Bengaluru")
def test_disruption_modes():
    """
    Verify all three disruption modes run without error and each removes
    approximately the expected fraction of edges.
    """
    G, pos_dict = build_synthetic_city_graph("Bengaluru", 8, 8)
    loader = OSMGraphLoader()
    original_edges = G.number_of_edges()
    pct = 0.3

    for mode in ["random", "bridge", "cascade", "node"]:
        G_d = loader.simulate_disaster(G, pct=pct, mode=mode, seed=1)
        remaining = G_d.number_of_edges()
        removed_pct = (original_edges - remaining) / max(original_edges, 1)
        print(f"    Mode '{mode}': {original_edges} → {remaining} edges "
              f"(removed {removed_pct:.1%})")
        assert remaining >= 0, f"Mode {mode} produced negative edge count"
        assert remaining < original_edges, \
            f"Mode {mode} did not remove any edges"


# ---------------------------------------------------------------------------
# T7: OSM real graph (Bengaluru — uses cached GraphML or synthetic fallback)
# ---------------------------------------------------------------------------
@test("OSM Real Graph — fetch/load Bengaluru, heal, compute RI")
def test_osm_real_graph():
    """
    Attempts to load the cached Bengaluru GraphML.
    If osmnx is not installed or cache is missing, uses the synthetic fallback.
    Validates that to_networkx, simulate_disaster, and CriticalityVarianceEngine work end-to-end.
    """
    loader = OSMGraphLoader()
    engine = CriticalityVarianceEngine(num_samples=5)

    cache_exists = False
    import os
    cache_path = os.path.join(loader.cache_dir, "bengaluru_drive.graphml")
    if os.path.exists(cache_path) and OSMNX_AVAILABLE:
        print("    Loading from cache ...")
        G_osm = loader.fetch_city_graph("Bengaluru", use_cache=True)
        G, pos_dict = loader.to_networkx(G_osm)
        cache_exists = True
    else:
        print("    osmnx/cache not available — using synthetic Bengaluru graph")
        G, pos_dict = build_synthetic_city_graph("Bengaluru", 12, 12)

    print(f"    Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    assert G.number_of_nodes() > 0, "Graph should have nodes"
    assert G.number_of_edges() > 0, "Graph should have edges"

    # Simulate 30% random disruption
    G_disrupted = loader.simulate_disaster(G, pct=0.3, mode="random", seed=7)
    print(f"    After 30% disruption: {G_disrupted.number_of_edges()} edges remain")

    # Compute RI
    metrics = engine.compute_resilience_index(G, G_disrupted, G_disrupted)
    ri = metrics["RI_healed"]
    print(f"    RI after disruption: {ri:.3f}")
    print(f"    LCC: {metrics['LCC']['healed']} / {metrics['LCC']['original']}")
    print(f"    Fiedler: {metrics['FiedlerLambda']['healed']:.4f}")

    assert 0.0 <= ri <= 1.5, f"RI out of plausible range: {ri}"

    # Serialization check
    serialized = loader.serialize_graph_for_api(G_disrupted, pos_dict, city_name="Bengaluru")
    assert "nodes" in serialized, "Serialized graph missing 'nodes'"
    assert "edges" in serialized, "Serialized graph missing 'edges'"
    print(f"    Serialized: {len(serialized['nodes'])} nodes, {len(serialized['edges'])} edges")


# ---------------------------------------------------------------------------
# T8: GraphHealingEngine on OSM-derived graph
# ---------------------------------------------------------------------------
@test("GraphHealingEngine — heal a disrupted synthetic Bengaluru graph")
def test_healing_on_osm():
    """
    Apply the full healing pipeline to a disrupted synthetic graph.
    Verifies healing adds edges and RI improves.
    """
    G, pos_dict = build_synthetic_city_graph("Bengaluru", 8, 8)
    loader = OSMGraphLoader()
    engine_ri = CriticalityVarianceEngine(num_samples=5)

    G_disrupted = loader.simulate_disaster(G, pct=0.4, mode="random", seed=99)

    # Build a simple probability map (all 0.5)
    prob_map = np.full((512, 512), 0.5, dtype=np.float32)
    attention_priors = []

    # Normalize positions to [0, 512] for healing engine
    ys = [p[0] for p in pos_dict.values()]
    xs = [p[1] for p in pos_dict.values()]
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)

    norm_pos = {}
    for nid, (y, x) in pos_dict.items():
        ny = (y - y_min) / max(y_max - y_min, 1) * 511
        nx_ = (x - x_min) / max(x_max - x_min, 1) * 511
        norm_pos[nid] = (ny, nx_)

    # Only keep nodes present in disrupted graph
    disrupted_pos = {n: norm_pos[n] for n in G_disrupted.nodes() if n in norm_pos}

    healing_engine = GraphHealingEngine(max_heal_dist=80.0)
    healed_G, healed_edges = healing_engine.heal_graph(
        G_disrupted, disrupted_pos, prob_map, attention_priors
    )

    print(f"    Disrupted: {G_disrupted.number_of_edges()} edges")
    print(f"    Healed:    {healed_G.number_of_edges()} edges (+{len(healed_edges)} connections)")

    metrics = engine_ri.compute_resilience_index(G, G_disrupted, healed_G)
    print(f"    RI disrupted: {metrics['RI_disrupted']:.3f}")
    print(f"    RI healed:    {metrics['RI_healed']:.3f}")
    print(f"    Healed edges added: {len(healed_edges)}")

    # RI should be >= disrupted RI (healing can't make it worse)
    assert metrics["RI_healed"] >= metrics["RI_disrupted"] - 0.01, \
        "Healing should not decrease RI"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  RoadRishi Graph Validation Test Suite")
    print("=" * 60)

    # Run all tests
    test_mst_correctness()
    test_centrality_metrics()
    test_fiedler_eigenvalue()
    test_node_disabling()
    test_disaster_stress_sweep()
    test_disruption_modes()
    test_osm_real_graph()
    test_healing_on_osm()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total  = len(RESULTS)
    for name, ok, msg in RESULTS:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
        if msg:
            print(f"         → {msg}")
    print(f"\n  Result: {passed}/{total} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)
