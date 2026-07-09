OSM GraphML cache files are stored here automatically by osm_graph.py.

After running:
  from backend.osm_graph import OSMGraphLoader
  loader = OSMGraphLoader()
  G = loader.fetch_city_graph("Bengaluru")

You will see: bengaluru_drive.graphml appear in this folder.
