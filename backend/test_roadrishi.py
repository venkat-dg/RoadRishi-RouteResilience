import unittest
import sys
import os
from pathlib import Path
import numpy as np
import networkx as nx

# Add backend directory to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

from preprocessing import PreprocessPipeline
from model import RoadSegFormer
from healing import GraphHealingEngine
from resilience import CriticalityVarianceEngine


class TestRoadRishiPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessPipeline(tile_size=64, stride=32)
        self.model = RoadSegFormer()
        self.healing = GraphHealingEngine(max_heal_dist=100.0)
        self.resilience = CriticalityVarianceEngine(num_samples=10)

    def test_preprocessing(self):
        """Verify band loading, NDVI synthesis, and tiling structures."""
        # Synthesize dummy data
        red = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        nir = np.array([40.0, 80.0, 120.0], dtype=np.float32)
        
        # Test NDVI: (NIR - Red) / (NIR + Red)
        # (40-10)/(40+10) = 30/50 = 0.6
        # (80-20)/(80+20) = 60/100 = 0.6
        # (120-30)/(120+30) = 90/150 = 0.6
        ndvi = self.pipeline.synthesize_ndvi(red, nir)
        np.testing.assert_allclose(ndvi, 0.6)

    def test_model_inference_simulation(self):
        """Verify model inference simulation outputs correct fields and shapes."""
        width, height = 500, 500
        sim = self.model.simulate_scene_inference(width, height)
        
        self.assertIn("gt_mask", sim)
        self.assertIn("occlusion_mask", sim)
        self.assertIn("probabilities", sim)
        self.assertIn("attention_priors", sim)
        
        self.assertEqual(sim["gt_mask"].shape, (500, 500))
        self.assertTrue(np.all(sim["probabilities"] >= 0.0) and np.all(sim["probabilities"] <= 1.0))

    def test_graph_healing(self):
        """Verify graph building, fracturing, and Kruskal-based MST healing."""
        # Simple road structure: two straight segments separated by a gap
        # Left segment: (100, 100) -> (100, 200)
        # Right segment: (100, 250) -> (100, 350)
        # Gap is between x=200 and x=250 (length 50)
        
        # Build complete pre-disruption lines
        road_lines = [
            ((100, 100), (100, 200)),
            ((100, 250), (100, 350))
        ]
        
        pre_G, pos_dict = self.healing.build_initial_graph(road_lines, (500, 500))
        
        # Verify nodes and positions
        self.assertGreater(len(pre_G.nodes()), 0)
        
        # Disrupted graph - simulating deletion of middle segments
        # Let's say we have an occlusion mask covering the gap
        occlusion_mask = np.zeros((500, 500))
        occlusion_mask[80:120, 201:249] = 1.0
        
        disrupted_G = self.healing.fracture_graph(pre_G, pos_dict, occlusion_mask)
        
        # Verify that they are fractured into separate components
        self.assertFalse(nx.is_connected(disrupted_G))
        
        # Create a dummy probability map
        prob_map = np.ones((500, 500))
        
        # Heal the graph
        healed_G, healed_edges = self.healing.heal_graph(disrupted_G, pos_dict, prob_map, [])
        
        # Verify that they are connected again
        self.assertTrue(nx.is_connected(healed_G))
        self.assertEqual(len(healed_edges), 1)
        self.assertEqual(healed_edges[0]["length"], 50.0)

    def test_resilience_analysis(self):
        """Verify Monte Carlo simulations, Fiedler eigenvalues, and RI calculations."""
        # Create a simple connected graph: triangle G
        G = nx.Graph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("C", "A", weight=1.0)
        
        # Compute pre Fiedler eigenvalue
        fiedler = self.resilience.get_fiedler_eigenvalue(G)
        self.assertGreater(fiedler, 0.0)
        
        # Create fractured graph: missing A-B
        disrupted_G = G.copy()
        disrupted_G.remove_edge("A", "B")
        
        # Create healed graph: AB healed with pe=0.8
        healed_G = disrupted_G.copy()
        healed_G.add_edge("A", "B", weight=1.0, healed=True)
        
        # Compute RI
        ri_data = self.resilience.compute_resilience_index(G, disrupted_G, healed_G)
        self.assertIn("RI_healed", ri_data)
        self.assertIn("RI_disrupted", ri_data)
        self.assertGreater(ri_data["RI_healed"], ri_data["RI_disrupted"])
        
        # Run Monte Carlo
        healed_edges = [{"u": "A", "v": "B", "pe": 0.8}]
        node_stats, top_priorities = self.resilience.run_monte_carlo(healed_G, healed_edges)
        
        self.assertEqual(len(node_stats), 3)
        self.assertIn("quadrant", node_stats["A"])


if __name__ == "__main__":
    unittest.main()
