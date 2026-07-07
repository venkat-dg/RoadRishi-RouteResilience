# RoadRishi

**RoadRishi** is a Geospatial Intelligence Platform for Route Resilience, Occlusion-Robust Road Extraction, and Graph-Theoretic Criticality Analysis for Urban Mobility.

This repository contains the complete pipeline: from raw satellite image tiling, to deep learning road segmentation simulations, minimum spanning tree (MST) gap healing, and an interactive dark-themed dashboard visualization.

---

## Folder Structure

```
IIITB/
├── preprocessing/      # Satellite imagery tiling tools
├── backend/            # FastAPI preprocessing backend & graph resilience algorithms
└── frontend/           # Vite + React + Tailwind CSS dashboard UI
```

---

## 🛠️ Installation & Setup

### 1. Preprocessing (Satellite Image Tiler)
The preprocessing tool chunks large satellite imagery into smaller, overlapping tiles suitable for neural network ingestion.

To test the image tiler:
```bash
cd preprocessing
python demo_tiling.py
```
This generates tiled output directories demonstrating the stride window sliding.

---

### 2. Backend (FastAPI & Graph Algorithms)
The backend simulates a SegFormer MiT-B3 road segmentation model and executes Graph-Theoretic MST gap healing and Betweenness Centrality uncertainty scoring.

#### Setup:
Navigate to the `backend` folder:
```bash
cd backend
```

Install the dependencies:
```bash
pip install fastapi uvicorn numpy networkx scipy
```

#### Run the server:
```bash
python main.py
```
The backend server will start at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

---

### 3. Frontend (React Dashboard)
The dashboard is a premium, client-side, dark-themed geospatial intelligence visualizer built using Vite, React, Leaflet, and Recharts.

#### Setup:
Navigate to the `frontend` folder:
```bash
cd frontend
```

Install the packages:
```bash
npm install
```

#### Run the dev server:
```bash
npm run dev
```
The dashboard will start at `http://localhost:5173/`. Open this link in your browser to interact with the visualizer.

---

## 🌍 Core Features
*   **Segment/Heal Toggle**: Swap between segmentation boundaries (Before Healing) and Accepted/Rejected healed edges (After Healing).
*   **Interactive Parameters**: Calibrate confidence thresholds, temperature coefficients, and healing search radius values.
*   **Sequential Pipeline Stepper**: Simulate the compilation pipeline steps (Preprocessing $\to$ SegFormer $\to$ Graph Compilation $\to$ Resilience Index calculation) with live status bars.
*   **Risk Uncertainty Matrix**: Interact with a 2x2 scatter matrix mapping average Node Centrality against Standard Deviation to flag critical segments for satellite re-tasking.
*   **NDVI Channels Stacked Bar Chart**: Display NDVI frequency distributions for background, canopy, and road segments.
