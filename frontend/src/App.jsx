import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Polyline, Polygon, CircleMarker, Popup, useMap } from 'react-leaflet';
import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import {
  Satellite,
  Compass,
  Play,
  RotateCw,
  Activity,
  AlertTriangle,
  Info,
  CheckCircle2,
  XCircle,
  Sliders,
  Layers,
  ChevronDown,
  LineChart,
  Eye,
  EyeOff
} from 'lucide-react';

// --- MOCK DATA ---
const METRICS = {
  mIoU: 0.783,
  occlusionRecall: 0.691,
  dice: 0.821,
  relaxedIoU: 0.847,
  connectivityBefore: 0.45,
  connectivityAfter: 0.78,
  apls: 91.3,
  resilienceIndex: 0.74,
  lccRatio: 0.79,
  fiedler: 0.68
};

const NODES = [
  { id: "0312", lat: 12.9731, lng: 77.5951, meanBC: 0.81, stdBC: 0.72, pe: 0.87 },
  { id: "0087", lat: 12.9712, lng: 77.6018, meanBC: 0.76, stdBC: 0.38, pe: 0.91 },
  { id: "0441", lat: 12.9748, lng: 77.5881, meanBC: 0.69, stdBC: 0.61, pe: 0.73 },
  { id: "0201", lat: 12.9688, lng: 77.5968, meanBC: 0.42, stdBC: 0.29, pe: 0.94 },
  { id: "0156", lat: 12.9762, lng: 77.6031, meanBC: 0.33, stdBC: 0.71, pe: 0.55 },
  { id: "0523", lat: 12.9721, lng: 77.5924, meanBC: 0.88, stdBC: 0.79, pe: 0.62 },
  { id: "0614", lat: 12.9744, lng: 77.5995, meanBC: 0.79, stdBC: 0.55, pe: 0.83 },
  { id: "0072", lat: 12.9779, lng: 77.5934, meanBC: 0.73, stdBC: 0.41, pe: 0.89 },
  { id: "0389", lat: 12.9669, lng: 77.5918, meanBC: 0.25, stdBC: 0.18, pe: 0.97 },
  { id: "0445", lat: 12.9803, lng: 77.5982, meanBC: 0.51, stdBC: 0.33, pe: 0.92 },
  { id: "0188", lat: 12.9701, lng: 77.5943, meanBC: 0.62, stdBC: 0.58, pe: 0.71 },
  { id: "0297", lat: 12.9757, lng: 77.6062, meanBC: 0.44, stdBC: 0.82, pe: 0.48 },
  { id: "0731", lat: 12.9798, lng: 77.5912, meanBC: 0.83, stdBC: 0.66, pe: 0.69 },
  { id: "0502", lat: 12.9666, lng: 77.6001, meanBC: 0.19, stdBC: 0.44, pe: 0.88 },
  { id: "0667", lat: 12.9812, lng: 77.5948, meanBC: 0.57, stdBC: 0.27, pe: 0.95 },
  { id: "0813", lat: 12.9724, lng: 77.5888, meanBC: 0.91, stdBC: 0.83, pe: 0.58 },
  { id: "0345", lat: 12.9689, lng: 77.5955, meanBC: 0.38, stdBC: 0.63, pe: 0.66 },
  { id: "0129", lat: 12.9771, lng: 77.5971, meanBC: 0.77, stdBC: 0.46, pe: 0.86 },
  { id: "0458", lat: 12.9738, lng: 77.6008, meanBC: 0.66, stdBC: 0.35, pe: 0.91 },
  { id: "0271", lat: 12.9695, lng: 77.5979, meanBC: 0.48, stdBC: 0.51, pe: 0.74 }
];

const ROAD_SEGMENTS = [
  { id: "r01", coords: [[12.9785, 77.5881], [12.9787, 77.5920], [12.9789, 77.5958], [12.9788, 77.5995], [12.9786, 77.6031]], stdBC: 0.65 },
  { id: "r02", coords: [[12.9798, 77.5912], [12.9779, 77.5934], [12.9761, 77.5952], [12.9744, 77.5971]], stdBC: 0.42 },
  { id: "r03", coords: [[12.9727, 77.5868], [12.9729, 77.5901], [12.9731, 77.5935], [12.9730, 77.5968], [12.9728, 77.6002]], stdBC: 0.78 },
  { id: "r04", coords: [[12.9691, 77.5934], [12.9712, 77.5953], [12.9731, 77.5968], [12.9748, 77.5985]], stdBC: 0.58 },
  { id: "r05", coords: [[12.9757, 77.6062], [12.9762, 77.6028], [12.9766, 77.5998], [12.9768, 77.5965], [12.9766, 77.5930]], stdBC: 0.22 },
  { id: "r06", coords: [[12.9726, 77.6082], [12.9731, 77.6048], [12.9733, 77.6012], [12.9731, 77.5968]], stdBC: 0.18 },
  { id: "r07", coords: [[12.9669, 77.5918], [12.9688, 77.5922], [12.9710, 77.5928], [12.9731, 77.5935]], stdBC: 0.29 },
  { id: "r08", coords: [[12.9798, 77.5951], [12.9779, 77.5952], [12.9761, 77.5952], [12.9718, 77.5954]], stdBC: 0.45 },
  { id: "r09", coords: [[12.9803, 77.5982], [12.9795, 77.5965], [12.9784, 77.5948], [12.9770, 77.5932]], stdBC: 0.35 },
  { id: "r10", coords: [[12.9711, 77.5970], [12.9718, 77.5985], [12.9728, 77.6001], [12.9739, 77.6018]], stdBC: 0.15 },
  { id: "r11", coords: [[12.9701, 77.5943], [12.9710, 77.5955], [12.9720, 77.5965], [12.9731, 77.5968]], stdBC: 0.28 },
  { id: "r12", coords: [[12.9762, 77.6005], [12.9751, 77.5998], [12.9741, 77.5990], [12.9731, 77.5982]], stdBC: 0.33 },
  { id: "r13", coords: [[12.9713, 77.5908], [12.9720, 77.5924], [12.9727, 77.5941], [12.9731, 77.5960]], stdBC: 0.61 },
  { id: "r14", coords: [[12.9718, 77.6005], [12.9723, 77.5985], [12.9728, 77.5968], [12.9730, 77.5948]], stdBC: 0.25 },
  { id: "r15", coords: [[12.9812, 77.5898], [12.9808, 77.5917], [12.9801, 77.5935], [12.9792, 77.5948]], stdBC: 0.55 }
];

const ROAD_SEGMENTS_PRE_HEAL = [
  { id: "r01", coords: [[12.9785, 77.5881], [12.9787, 77.5920], [12.9789, 77.5958]] },
  { id: "r02", coords: [[12.9798, 77.5912], [12.9779, 77.5934]] },
  { id: "r03", coords: [[12.9727, 77.5868], [12.9729, 77.5901], [12.9731, 77.5935]] },
  { id: "r04", coords: [[12.9691, 77.5934], [12.9712, 77.5953]] },
  { id: "r05", coords: [[12.9757, 77.6062], [12.9762, 77.6028], [12.9766, 77.5998]] },
  { id: "r07", coords: [[12.9669, 77.5918], [12.9688, 77.5922]] },
  { id: "r08", coords: [[12.9798, 77.5951], [12.9779, 77.5952], [12.9761, 77.5952]] },
  { id: "r09", coords: [[12.9803, 77.5982], [12.9795, 77.5965]] },
  { id: "r13", coords: [[12.9713, 77.5908], [12.9720, 77.5924]] }
];

const HEALED_EDGES = [
  { id: "e_0042", coords: [[12.9731, 77.5935], [12.9718, 77.5941]], pe: 0.87, status: "healed", length: 71 },
  { id: "e_0107", coords: [[12.9733, 77.6012], [12.9739, 77.6018]], pe: 0.61, status: "healed", length: 82 },
  { id: "e_0198", coords: [[12.9712, 77.5953], [12.9710, 77.5955]], pe: 0.34, status: "rejected", length: 15 },
  { id: "e_0055", coords: [[12.9718, 77.5954], [12.9710, 77.5928]], pe: 0.92, status: "healed", length: 110 },
  { id: "e_0334", coords: [[12.9788, 77.5995], [12.9748, 77.5985]], pe: 0.71, status: "healed", length: 95 }
];

const OCCLUSION_ZONES = [
  [[12.9760, 77.5920], [12.9775, 77.5925], [12.9778, 77.5945], [12.9762, 77.5948], [12.9752, 77.5935]],
  [[12.9780, 77.6010], [12.9792, 77.6015], [12.9788, 77.6028], [12.9775, 77.6022]],
  [[12.9718, 77.5960], [12.9728, 77.5968], [12.9725, 77.5980], [12.9712, 77.5975]]
];

const NDVI_HIST = [
  { ndvi: "-1.0", road: 0, canopy: 2, bg: 8 },
  { ndvi: "-0.8", road: 1, canopy: 3, bg: 14 },
  { ndvi: "-0.6", road: 2, canopy: 5, bg: 18 },
  { ndvi: "-0.4", road: 4, canopy: 8, bg: 22 },
  { ndvi: "-0.2", road: 9, canopy: 12, bg: 19 },
  { ndvi: "0.0", road: 9, canopy: 12, bg: 15 },
  { ndvi: "0.1", road: 18, canopy: 14, bg: 10 },
  { ndvi: "0.2", road: 12, canopy: 20, bg: 6 },
  { ndvi: "0.4", road: 5, canopy: 35, bg: 3 },
  { ndvi: "0.6", road: 2, canopy: 48, bg: 1 },
  { ndvi: "0.8", road: 0, canopy: 30, bg: 1 },
  { ndvi: "1.0", road: 0, canopy: 10, bg: 0 }
];

const PIPELINE_STEPS = [
  { label: "Preprocessing", ms: 312 },
  { label: "Segmentation", ms: 1847 },
  { label: "Graph Build", ms: 203 },
  { label: "Resilience", ms: 891 }
];

const ALERTS = [
  { id: 1, level: "critical", title: "Critical Node Detected", time: "2m ago", desc: "Node 0312 — high BC + high σBC, re-tasking flagged" },
  { id: 2, level: "warning", title: "Connectivity Gap Found", time: "5m ago", desc: "Edge e_0198 rejected (pₑ = 0.34 < threshold 0.40)" },
  { id: 3, level: "info", title: "Calibration Complete", time: "8m ago", desc: "Temperature T* = 1.24 · val NLL improved by 12.3%" },
  { id: 4, level: "info", title: "Graph Healed", time: "12m ago", desc: "Connectivity ratio 0.45 → 0.78 · +73% improvement" },
  { id: 5, level: "warning", title: "Occlusion Detected", time: "15m ago", desc: "Canopy mask 22,117px · λ = 2.0 hidden BCE applied" }
];

// Helper to determine node risk color and label
function getNodeRisk(node) {
  if (node.meanBC > 0.7 && node.stdBC > 0.5) {
    return { color: '#EF4444', label: 'CRITICAL', short: 'C' };
  }
  if (node.meanBC > 0.7 && node.stdBC <= 0.5) {
    return { color: '#3B82F6', label: 'RELIABLE', short: 'R' };
  }
  if (node.meanBC <= 0.7 && node.stdBC > 0.5) {
    return { color: '#EAB308', label: 'UNCERTAIN', short: 'U' };
  }
  return { color: '#22C55E', label: 'SAFE', short: 'S' };
}

// React-Leaflet component to center/zoom map view dynamically
function MapController({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

// Custom Slider component following specific design specs
function CustomSlider({ label, min, max, step, value, onChange, theme = 'blue' }) {
  const percentage = ((value - min) / (max - min)) * 100;
  return (
    <div className="mb-12 select-none">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-[var(--text-secondary)]">{label}</span>
        <span
          className="font-mono font-semibold"
          style={{ color: theme === 'blue' ? 'var(--accent-blue)' : 'var(--accent-green)' }}
        >
          {value.toFixed(2)}
        </span>
      </div>
      <div className="relative h-[18px] flex items-center">
        <div className="absolute w-full h-[4px] bg-[var(--border)] rounded-full" />
        <div
          className="absolute h-[4px] rounded-full"
          style={{
            width: `${percentage}%`,
            background: theme === 'blue'
              ? 'linear-gradient(90deg, #2563EB, #3B82F6)'
              : 'linear-gradient(90deg, #059669, #22C55E)'
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute w-full h-[18px] opacity-0 cursor-pointer z-10"
        />
        <div
          className="absolute w-[14px] h-[14px] border-[2px] border-[#0B0F1A] rounded-full pointer-events-none transform -translate-x-1/2"
          style={{
            left: `${percentage}%`,
            backgroundColor: theme === 'blue' ? '#3B82F6' : '#22C55E',
            boxShadow: theme === 'blue' ? '0 0 8px rgba(59,130,246,0.5)' : '0 0 8px rgba(34,197,94,0.5)'
          }}
        />
      </div>
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('Segmentation');
  const [scene, setScene] = useState('Bengaluru Urban 5km²');
  
  // Sliders state
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.50);
  const [temperature, setTemperature] = useState(1.00);
  const [gapHealRadius, setGapHealRadius] = useState(50);
  const [angularTolerance, setAngularTolerance] = useState(30);
  const [mcSamples, setMcSamples] = useState('100');
  const [isAttentionGuided, setIsAttentionGuided] = useState(true);

  // Map settings based on scene selection
  const [mapCenter, setMapCenter] = useState([12.9716, 77.5946]);
  const [mapZoom, setMapZoom] = useState(15);

  // Visibility of Map Layers
  const [layersVisibility, setLayersVisibility] = useState({
    segmentation: true,
    confidence: false,
    healedEdges: false,
    nodeRisk: true,
    uncertaintyHeatmap: false,
    occlusionZones: false
  });
  const [isLayerSwitcherOpen, setIsLayerSwitcherOpen] = useState(false);

  // Interactive UI elements
  const [isBeforeHealing, setIsBeforeHealing] = useState(false);
  const [hoverCoords, setHoverCoords] = useState(null);
  const [activeNodeId, setActiveNodeId] = useState(null);
  const [selectedQuadrant, setSelectedQuadrant] = useState(null);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  
  // Animation counting up progress (0 to 1)
  const [animationProgress, setAnimationProgress] = useState(1);

  // Watch scene change and pan map
  useEffect(() => {
    if (scene === 'Bengaluru Urban 5km²') {
      setMapCenter([12.9716, 77.5946]);
      setMapZoom(15);
    } else if (scene === 'Forested Suburban') {
      setMapCenter([12.9516, 77.5146]);
      setMapZoom(14);
    } else if (scene === 'Rural Highway') {
      setMapCenter([13.0716, 77.4946]);
      setMapZoom(13);
    }
  }, [scene]);

  // Handle Pipeline execution animation
  const handleRunPipeline = () => {
    if (isPipelineRunning) return;
    setIsPipelineRunning(true);
    setCurrentStepIndex(0);
    setAnimationProgress(0);

    const stepsDelays = [312, 1847, 203, 891];
    let currentStep = 0;

    const runStep = () => {
      if (currentStep < stepsDelays.length) {
        setTimeout(() => {
          currentStep++;
          setCurrentStepIndex(currentStep);
          runStep();
        }, stepsDelays[currentStep]);
      } else {
        // Complete
        setIsPipelineRunning(false);
        setCurrentStepIndex(-1);
        
        // Trigger Toast
        setToastMessage("Pipeline complete • 3.25s");
        setShowToast(true);
        setTimeout(() => setShowToast(false), 3000);
        
        // Turn on relevant layers
        setLayersVisibility(prev => ({
          ...prev,
          confidence: true,
          healedEdges: true,
          occlusionZones: true
        }));

        // Animate metrics count up
        let start = null;
        const duration = 800;
        const animate = (timestamp) => {
          if (!start) start = timestamp;
          const elapsed = timestamp - start;
          const progress = Math.min(1, elapsed / duration);
          setAnimationProgress(progress);
          if (progress < 1) {
            requestAnimationFrame(animate);
          }
        };
        requestAnimationFrame(animate);
      }
    };

    runStep();
  };

  // Generate coordinate-based hover readings
  const getPseudoMetrics = (lat, lng) => {
    if (!lat || !lng) return { ndvi: 0.12, conf: 0.87 };
    const seed = Math.sin(lat) * Math.cos(lng);
    const ndvi = 0.12 + Math.abs(seed * 0.4);
    const conf = 0.87 - Math.abs(seed * 0.15);
    return { ndvi: ndvi.toFixed(2), conf: conf.toFixed(2) };
  };

  const currentHoverMetrics = hoverCoords ? getPseudoMetrics(hoverCoords[0], hoverCoords[1]) : { ndvi: 0.12, conf: 0.87 };

  // Filter nodes by selected risk quadrant
  const filteredNodes = selectedQuadrant
    ? NODES.filter(node => getNodeRisk(node).label === selectedQuadrant)
    : NODES;

  return (
    <div className="h-screen w-screen bg-[var(--bg-base)] text-[var(--text-primary)] font-sans flex flex-col overflow-hidden relative">
      
      {/* Toast Alert */}
      {showToast && (
        <div className="absolute top-16 left-1/2 transform -translate-x-1/2 z-[1000] bg-emerald-950 border border-emerald-500 text-emerald-400 font-semibold px-6 py-3 rounded-lg shadow-xl flex items-center gap-2 animate-bounce">
          <CheckCircle2 size={16} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* --- TOPBAR --- */}
      <header className="h-[48px] bg-[var(--bg-base)] border-b border-[var(--border)] shrink-0 flex items-center justify-between px-5 z-[500] relative select-none">
        {/* Left - Wordmark */}
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center relative w-4 h-4">
            <svg viewBox="0 0 24 24" className="w-4 h-4 text-[var(--accent-blue)] fill-none stroke-current stroke-[2.5]" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="5" />
              <path d="M4 4l3 3M20 4l-3 3M4 20l3-3M20 20l-3-3" />
            </svg>
          </div>
          <span className="font-outfit font-extrabold text-sm tracking-wide">
            Road<span className="text-white">Rishi</span>
          </span>
        </div>

        {/* Center - Navigation Tabs */}
        <nav className="flex h-full items-center gap-6 shrink-0">
          {['Segmentation', 'Graph Analysis', 'Resilience', 'Alerts'].map(tab => {
            const isActive = activeTab === tab;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{ background: 'transparent', border: 'none' }}
                className={`relative px-4 h-full flex items-center justify-center text-[13px] font-medium transition-colors hover:text-[var(--text-secondary)] shrink-0 ${
                  isActive ? 'text-[var(--text-primary)] font-semibold' : 'text-[var(--text-muted)]'
                }`}
              >
                {tab}
                {isActive && (
                  <span className="absolute bottom-0 left-4 right-4 h-[2px] bg-[var(--accent-blue)] rounded-t-sm" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Right - Badges */}
        <div className="flex items-center gap-2.5">
          <div className="border border-[var(--accent-blue)] text-[var(--accent-blue)] bg-transparent px-3 py-1 rounded-[4px] text-[11px] font-semibold tracking-wider font-outfit uppercase">
            ISRO H2S 2026
          </div>
          <div className="bg-emerald-950/30 border border-emerald-900/60 text-[var(--accent-green)] px-3 py-1 rounded-[4px] text-[11px] font-semibold flex items-center gap-1.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[var(--accent-green)]"></span>
            </span>
            Pipeline Ready
          </div>
        </div>
      </header>

      {/* Main Panel Container */}
      <div className="flex-1 flex overflow-hidden relative p-3 gap-3 bg-[var(--bg-base)]">
        
        {/* --- LEFT SIDEBAR --- */}
        <aside className="w-[280px] h-full bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-y-auto flex flex-col gap-6 z-[400] shrink-0">
          
          {/* Section: INPUT */}
          <div className="py-9 px-6 border-b border-[var(--border)] space-y-8">
            <div className="flex items-center gap-2 mb-6 select-none">
              <div className="w-[3px] h-[14px] bg-[var(--accent-blue)] rounded-sm" />
              <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">INPUT</span>
            </div>
            
            <div className="mb-8">
              <label className="block text-[11px] text-[#64748B] mb-1.5">Scene</label>
              <div className="relative">
                <select
                  value={scene}
                  onChange={(e) => setScene(e.target.value)}
                  className="w-full bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-md text-[13px] text-[#E2E8F0] p-2 pr-8 appearance-none focus:outline-none focus:border-[var(--accent-blue)] focus:ring-[3px] focus:ring-blue-500/15 cursor-pointer font-medium"
                >
                  <option>Bengaluru Urban 5km²</option>
                  <option>Forested Suburban</option>
                  <option>Rural Highway</option>
                </select>
                <ChevronDown size={14} className="absolute right-2.5 top-1/2 transform -translate-y-1/2 text-[#64748B] pointer-events-none" />
              </div>
            </div>

            <div className="mb-8">
              <label className="block text-[11px] text-[#64748B] mb-1.5">Band Config</label>
              <div className="text-[12px] bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-md p-2 text-[var(--text-secondary)] opacity-45 cursor-not-allowed select-none">
                G / R / NIR / NDVI (4ch)
              </div>
            </div>

            <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md p-2 flex items-center gap-2 text-[11px] text-[var(--text-muted)] font-medium select-none">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full shrink-0" />
              Resolution: 5.8m · Sensor: LISS-IV MX
            </div>
          </div>

          {/* Section: MODEL */}
          <div className="py-9 px-6 border-b border-[var(--border)] space-y-9">
            <div className="flex items-center gap-2 mb-4 select-none">
              <div className="w-[3px] h-[14px] bg-[var(--accent-purple)] rounded-sm" />
              <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">MODEL</span>
            </div>

            <div className="bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-md p-2.5 flex items-center justify-between mb-9 select-none">
              <div className="flex items-center gap-2">
                <div className="flex items-end gap-0.5 h-3">
                  <div className="w-[2px] h-[6px] bg-[var(--accent-purple)] rounded-full" />
                  <div className="w-[2px] h-[12px] bg-[var(--accent-purple)] rounded-full" />
                  <div className="w-[2px] h-[8px] bg-[var(--accent-purple)] rounded-full" />
                  <div className="w-[2px] h-[14px] bg-[var(--accent-purple)] rounded-full" />
                </div>
                <span className="text-[13px] font-semibold text-[#E2E8F0]">SegFormer MiT-B3</span>
              </div>
              <div className="bg-emerald-950/20 border border-emerald-900/60 text-[var(--accent-green)] text-[9px] font-bold tracking-widest px-2 py-0.5 rounded-[4px] uppercase flex items-center gap-1">
                <span className="w-1 h-1 bg-[var(--accent-green)] rounded-full animate-pulse" />
                LOADED
              </div>
            </div>

            <CustomSlider
              label="Confidence Threshold"
              min={0.0}
              max={1.0}
              step={0.05}
              value={confidenceThreshold}
              onChange={setConfidenceThreshold}
              theme="blue"
            />

            <CustomSlider
              label="Temperature T (Calibrated)"
              min={0.5}
              max={3.0}
              step={0.1}
              value={temperature}
              onChange={setTemperature}
              theme="blue"
            />

            <div className="bg-yellow-950/15 border border-yellow-900/30 rounded-md p-2 px-3 flex items-center justify-between select-none">
              <span className="text-xs text-[var(--text-secondary)]">λ hidden BCE</span>
              <span className="text-sm font-bold text-[var(--accent-yellow)] font-mono">2.0</span>
            </div>
          </div>

          {/* Section: GRAPH HEALING */}
          <div className="py-9 px-6 border-b border-[var(--border)] space-y-9">
            <div className="flex items-center gap-2 mb-4 select-none">
              <div className="w-[3px] h-[14px] bg-[var(--accent-green)] rounded-sm" />
              <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">GRAPH HEALING</span>
            </div>

            <CustomSlider
              label="Gap Heal Radius"
              min={10}
              max={100}
              step={5}
              value={gapHealRadius}
              onChange={setGapHealRadius}
              theme="green"
            />

            <CustomSlider
              label="Angular Tolerance"
              min={10}
              max={60}
              step={5}
              value={angularTolerance}
              onChange={setAngularTolerance}
              theme="green"
            />

            <div className="flex items-center justify-between gap-3 mb-9 select-none">
              <span className="text-xs text-[var(--text-secondary)]">MC Samples</span>
              <input
                type="text"
                value={mcSamples}
                onChange={(e) => setMcSamples(e.target.value.replace(/\D/g, ''))}
                className="w-16 bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-md text-[13px] text-right font-mono font-medium p-1.5 px-2 focus:outline-none focus:border-[var(--accent-green)] focus:ring-[3px] focus:ring-emerald-500/10"
              />
            </div>

            <div className="flex items-center justify-between select-none">
              <span className="text-xs text-[var(--text-secondary)]">Attention-Guided</span>
              <button
                onClick={() => setIsAttentionGuided(!isAttentionGuided)}
                className={`w-[36px] h-[20px] rounded-full p-0.5 relative transition-colors duration-200 focus:outline-none ${
                  isAttentionGuided ? 'bg-[var(--accent-green)]' : 'bg-[#2A3347]'
                }`}
              >
                <div
                  className="w-3.5 h-3.5 bg-white rounded-full transition-all duration-200"
                  style={{ transform: isAttentionGuided ? 'translateX(16px)' : 'translateX(0px)' }}
                />
              </button>
            </div>
          </div>

          {/* Section: HEALING VISUALIZER */}
          <div className="py-8 px-6 border-b border-[var(--border)] space-y-6">
            <div className="flex items-center gap-2 mb-4 select-none">
              <div className="w-[3px] h-[14px] bg-[var(--accent-cyan)] rounded-sm" />
              <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">HEALING VISUALIZER</span>
            </div>
            
            <div className="flex items-center bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-md p-1 overflow-hidden select-none w-full">
              <button
                onClick={() => setIsBeforeHealing(true)}
                className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all text-center cursor-pointer ${
                  isBeforeHealing
                    ? 'bg-[var(--border)] text-white shadow'
                    : 'text-[var(--text-secondary)] hover:text-white'
                }`}
              >
                Before
              </button>
              <button
                onClick={() => setIsBeforeHealing(false)}
                className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all text-center cursor-pointer ${
                  !isBeforeHealing
                    ? 'bg-[var(--border)] text-white shadow'
                    : 'text-[var(--text-secondary)] hover:text-white'
                }`}
              >
                After
              </button>
            </div>
            
            <div className="select-none">
              {isBeforeHealing ? (
                <div className="bg-rose-950/20 border border-rose-900/40 text-[var(--accent-red)] font-semibold py-2 rounded-[4px] text-xs text-center font-outfit uppercase">
                  Connectivity: 0.45 (Baseline)
                </div>
              ) : (
                <div className="bg-emerald-950/20 border border-emerald-900/40 text-[var(--accent-green)] font-semibold py-2 rounded-[4px] text-xs text-center font-outfit uppercase">
                  Connectivity: {(0.45 + 0.33 * animationProgress).toFixed(2)} (+{Math.round(73 * animationProgress)}%)
                </div>
              )}
            </div>
          </div>

          {/* Run Pipeline Container */}
          <div className="py-8 px-6 mt-auto border-t border-[var(--border)]">
            <button
              onClick={handleRunPipeline}
              disabled={isPipelineRunning}
              className="w-full py-3 bg-gradient-to-r from-blue-700 to-blue-500 hover:from-blue-600 hover:to-blue-400 disabled:from-blue-900 disabled:to-blue-900 disabled:text-slate-500 rounded-lg text-white font-semibold text-[13px] flex items-center justify-center gap-2 cursor-pointer active:scale-[0.99] transition-all duration-150 shadow-lg shadow-blue-900/20"
            >
              {isPipelineRunning ? (
                <>
                  <RotateCw size={14} className="animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <svg viewBox="0 0 24 24" className="w-3 h-3 fill-current">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  <span>Run Pipeline</span>
                </>
              )}
            </button>

            {/* Stepper Logic on Run */}
            {isPipelineRunning && (
              <div className="mt-4 border border-[var(--border)] rounded-lg p-2.5 bg-[var(--bg-elevated)] text-[11px] font-medium space-y-2 select-none">
                {PIPELINE_STEPS.map((step, idx) => {
                  const isDone = currentStepIndex > idx;
                  const isCurrent = currentStepIndex === idx;
                  return (
                    <div key={step.label} className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <div
                          className={`w-3.5 h-3.5 rounded-full flex items-center justify-center font-bold text-[8px] border ${
                            isDone
                              ? 'bg-emerald-950 border-emerald-500 text-emerald-400'
                              : isCurrent
                              ? 'bg-blue-950 border-blue-500 text-blue-400 animate-pulse'
                              : 'bg-slate-900 border-slate-800 text-slate-500'
                          }`}
                        >
                          {isDone ? '✓' : idx + 1}
                        </div>
                        <span className={isDone ? 'text-slate-400' : isCurrent ? 'text-blue-400 font-semibold' : 'text-slate-500'}>
                          {step.label}
                        </span>
                      </div>
                      
                      <div className="flex items-center gap-2 font-mono text-[10px] text-slate-500">
                        {isCurrent && <span className="animate-pulse text-blue-400">●●●</span>}
                        {isDone && <span className="text-[var(--accent-green)] font-semibold">{step.ms}ms</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </aside>

        {/* --- CENTER MAP AREA --- */}
        <section className="flex-1 flex flex-col h-full relative min-w-0 bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-hidden">
          


          {/* Floating Layer Switcher Panel */}
          <div className="absolute top-3 right-3 z-[450] flex flex-col items-end">
            <button
              onClick={() => setIsLayerSwitcherOpen(!isLayerSwitcherOpen)}
              className="p-2.5 bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-elevated)] rounded-md text-[var(--text-primary)] shadow-xl cursor-pointer active:scale-95 transition-all"
            >
              <Layers size={15} />
            </button>
            
            {isLayerSwitcherOpen && (
              <div className="mt-2 bg-[var(--bg-elevated)] border border-[var(--border-bright)] rounded-lg p-3 shadow-2xl min-width-[180px] flex flex-col gap-2.5 z-[500] select-none">
                <span className="text-[9px] font-bold text-[#64748B] uppercase tracking-wider border-b border-[var(--border)] pb-1.5">MAP LAYERS</span>
                {Object.keys(layersVisibility).map(layerKey => {
                  const label = layerKey.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
                  return (
                    <label key={layerKey} className="flex items-center gap-2.5 cursor-pointer text-xs select-none">
                      <input
                        type="checkbox"
                        checked={layersVisibility[layerKey]}
                        onChange={() => setLayersVisibility(prev => ({ ...prev, [layerKey]: !prev[layerKey] }))}
                        className="rounded border-[var(--border-bright)] bg-[var(--bg-card)] text-[var(--accent-blue)] focus:ring-[var(--accent-blue)] w-3.5 h-3.5 cursor-pointer"
                      />
                      <span className={layersVisibility[layerKey] ? 'text-[#F1F5F9]' : 'text-[var(--text-secondary)]'}>
                        {label}
                      </span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>

          {/* Leaflet Map Visualizer */}
          <div className="flex-1 relative z-0">
            <MapContainer
              center={mapCenter}
              zoom={mapZoom}
              scrollWheelZoom={true}
              style={{ height: '100%', width: '100%' }}
              zoomControl={false}
            >
              <MapController center={mapCenter} zoom={mapZoom} />
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
                maxZoom={19}
              />

              {/* Analysis Tile Bounds Rectangle */}
              <Polygon
                positions={[[12.965, 77.582], [12.983, 77.582], [12.983, 77.607], [12.965, 77.607]]}
                pathOptions={{ color: '#EAB308', weight: 1, dashArray: '8,6', fillOpacity: 0, opacity: 0.35 }}
              />

              {/* 1. Road Segmentation polylines */}
              {layersVisibility.segmentation && (isBeforeHealing ? ROAD_SEGMENTS_PRE_HEAL : ROAD_SEGMENTS).map(seg => (
                <React.Fragment key={seg.id}>
                  {/* Glow layer (underneath) */}
                  <Polyline
                    positions={seg.coords}
                    pathOptions={{ color: '#1D4ED8', weight: 8, opacity: 0.12 }}
                  />
                  {/* Main Line */}
                  <Polyline
                    positions={seg.coords}
                    pathOptions={{
                      color: layersVisibility.uncertaintyHeatmap 
                        ? (seg.stdBC > 0.6 ? '#EF4444' : seg.stdBC > 0.3 ? '#EAB308' : '#22C55E')
                        : '#00E5FF',
                      weight: 2.5,
                      opacity: 0.85
                    }}
                  />
                </React.Fragment>
              ))}

              {/* 2. Confidence Heatmap (blurred cyan layer underneath) */}
              {layersVisibility.confidence && ROAD_SEGMENTS.map(seg => (
                <Polyline
                  key={`conf-${seg.id}`}
                  positions={seg.coords}
                  pathOptions={{ color: '#00E5FF', weight: 20, opacity: 0.10 }}
                />
              ))}

              {/* 3. Healed Edges */}
              {layersVisibility.healedEdges && !isBeforeHealing && HEALED_EDGES.map(edge => {
                const isAccepted = edge.pe >= 0.40;
                return (
                  <Polyline
                    key={edge.id}
                    positions={edge.coords}
                    pathOptions={{
                      color: isAccepted ? '#39FF14' : '#EF4444',
                      weight: isAccepted ? 2.5 : 1.5,
                      dashArray: isAccepted ? '7,5' : '3,7',
                      opacity: isAccepted ? 0.9 : 0.5
                    }}
                  />
                );
              })}

              {/* 4. Node Risk circles */}
              {layersVisibility.nodeRisk && filteredNodes.map(node => {
                const risk = getNodeRisk(node);
                const isActive = activeNodeId === node.id;
                
                // Fade out other nodes if filtered on plot
                const isFilteredOut = selectedQuadrant && risk.label !== selectedQuadrant;
                const opacity = isFilteredOut ? 0.2 : 0.9;

                return (
                  <CircleMarker
                    key={node.id}
                    center={[node.lat, node.lng]}
                    radius={risk.label === 'CRITICAL' ? 7 : 5.5}
                    pathOptions={{
                      fillColor: risk.color,
                      color: isActive ? '#FFFFFF' : risk.color,
                      weight: isActive ? 2.5 : 1.5,
                      fillOpacity: opacity,
                      opacity: opacity
                    }}
                    eventHandlers={{
                      click: () => setActiveNodeId(node.id === activeNodeId ? null : node.id)
                    }}
                  >
                    <Popup>
                      <div className="text-xs font-sans text-[var(--text-primary)]">
                        <div className="font-bold border-b border-[var(--border)] pb-1 mb-1 font-outfit text-sm">Node #{node.id}</div>
                        <div className="font-mono">BC = {node.meanBC.toFixed(2)} | &sigma;BC = {node.stdBC.toFixed(2)}</div>
                        <div className="font-mono">p<sub>e</sub> = {node.pe.toFixed(2)}</div>
                        <div className="mt-1 font-bold text-[10px] uppercase tracking-wider" style={{ color: risk.color }}>
                          Risk: {risk.label}
                        </div>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}

              {/* 5. Occlusion Zones */}
              {layersVisibility.occlusionZones && OCCLUSION_ZONES.map((zone, idx) => (
                <Polygon
                  key={`zone-${idx}`}
                  positions={zone}
                  pathOptions={{
                    fillColor: '#A855F7',
                    fillOpacity: 0.12,
                    color: '#A855F7',
                    weight: 1,
                    dashArray: '4,4',
                    opacity: 0.3
                  }}
                />
              ))}

              {/* Hook to capture coordinates on mousemove */}
              <div className="hidden">
                <MapEventHandler onMouseMove={(latlng) => setHoverCoords([latlng.lat, latlng.lng])} />
              </div>
            </MapContainer>
          </div>

          {/* Info bar coordinates (bottom of map) */}
          <div className="h-[32px] bg-[var(--bg-base)] border-t border-[var(--border)] px-4 flex items-center justify-between text-[11px] font-mono text-slate-300 select-none">
            <div>
              LISS-IV • 5km&times;5km tile • CRS: WGS84 • {hoverCoords ? `${hoverCoords[0].toFixed(4)}°N ${hoverCoords[1].toFixed(4)}°E` : '12.9731°N 77.5951°E'} | NDVI: ~{currentHoverMetrics.ndvi} | Conf: ~{currentHoverMetrics.conf} • Last processed: just now
            </div>
            <div>
              Tiles &copy; Esri &mdash; Source: Esri, Maxar
            </div>
          </div>
        </section>

        {/* --- RIGHT SIDEBAR PANEL --- */}
        <aside className="w-[380px] h-full bg-[var(--bg-card)] border border-[var(--border)] rounded-xl overflow-y-auto p-5 flex flex-col gap-6 shrink-0 z-[400]">
          
          {/* TAB 1: SEGMENTATION */}
          {activeTab === 'Segmentation' && (
            <>
              {/* 2x2 Metric Cards */}
              <div className="grid grid-cols-2 gap-3.5">
                {[
                  { label: 'mIoU', val: METRICS.mIoU, format: 'toFixed(3)', target: 'Target ≥ 0.75', isMet: METRICS.mIoU >= 0.75 },
                  { label: 'Occlusion-Recall', val: METRICS.occlusionRecall, format: 'toFixed(3)', target: 'Target ≥ 0.68', isMet: METRICS.occlusionRecall >= 0.68 },
                  { label: 'Dice Score', val: METRICS.dice, format: 'toFixed(3)', target: null },
                  { label: 'Relaxed IoU', val: METRICS.relaxedIoU, format: 'toFixed(3)', target: '4px buffer (23m)' }
                ].map(card => (
                  <div
                    key={card.label}
                    className="glass-card border border-[var(--border)] rounded-lg p-4 relative overflow-hidden transition-all duration-200 hover:border-blue-500/25 flex flex-col items-center justify-center text-center min-h-[112px]"
                  >
                    {/* Top Accent Strip */}
                    <div
                      className="absolute top-0 left-0 right-0 h-[3px]"
                      style={{ backgroundColor: card.target && card.isMet !== false ? 'var(--accent-green)' : 'var(--accent-blue)' }}
                    />
                    
                    <div className="flex flex-col items-center select-none mb-1">
                      <span className="text-[10px] text-[var(--text-secondary)] uppercase font-semibold tracking-wider mb-1">{card.label}</span>
                      <span className="text-[8px] font-bold tracking-wider px-1.5 py-0.5 rounded-[3px] bg-slate-900/60 uppercase" style={{ color: card.target && card.isMet !== false ? 'var(--accent-green)' : 'var(--accent-yellow)' }}>
                        {card.target && card.isMet !== false ? 'Target Met' : 'Projected'}
                      </span>
                    </div>

                    <div className="text-3xl font-bold font-mono text-white select-none my-1">
                      {(card.val * animationProgress).toFixed(3)}
                    </div>

                    {card.target && (
                      <span className="text-[11px] text-slate-400 font-semibold select-none mt-1">{card.target}</span>
                    )}
                  </div>
                ))}
              </div>

              <span className="text-[10px] text-slate-400 font-medium leading-relaxed mt-1 block select-none">
                Projected performance on Bengaluru 5km² tile based on SegFormer MiT-B3 benchmarks on comparable datasets.
              </span>

              {/* Loss Architecture Section */}
              <div className="mt-4 border-t border-[var(--border)] pt-5 select-none">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-[3px] h-[14px] bg-[var(--accent-purple)] rounded-sm" />
                  <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Loss Architecture (Design Weights)</span>
                </div>
                <p className="text-[10px] text-slate-400 mb-3 italic">Design parameters — not live training output</p>
                
                <div className="space-y-3.5">
                  {[
                    { label: 'L_BCE', weight: '11.4%', width: '11.4%', color: 'var(--accent-blue)' },
                    { label: 'L_clDice', weight: '~0.85 est.', width: '85.0%', color: 'var(--accent-purple)' },
                    { label: 'L_BCE_Hdn', weight: '5.7%', width: '5.7%', color: 'var(--accent-yellow)' }
                  ].map(row => (
                    <div key={row.label} className="flex items-center gap-3 py-1 text-xs">
                      <span className="w-16 font-mono text-[var(--text-secondary)]">{row.label}</span>
                      <div className="flex-1 h-[6px] bg-[var(--border)] rounded-full overflow-hidden relative">
                        <div
                          className="h-full rounded-full transition-all duration-[900ms] ease-out"
                          style={{
                            width: animationProgress > 0.2 ? row.width : '0%',
                            backgroundColor: row.color
                          }}
                        />
                      </div>
                      <span className="w-20 text-right font-mono text-white">{row.weight}</span>
                    </div>
                  ))}
                </div>

                <div className="flex gap-4 mt-4 text-[11px] text-[var(--text-secondary)] font-medium">
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--accent-blue)]" /> BCE</div>
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--accent-purple)]" /> clDice</div>
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--accent-yellow)]" /> Hidden</div>
                </div>
              </div>

              {/* NDVI distribution */}
              <div className="mt-5 border-t border-[var(--border)] pt-5 select-none flex-grow flex flex-col min-h-[240px]">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-[3px] h-[14px] bg-[var(--accent-green)] rounded-sm" />
                  <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">NDVI Channel Distribution</span>
                </div>
                
                <div className="flex gap-4 mb-2 text-[10px] text-[var(--text-secondary)] font-medium">
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--accent-blue)]" /> Road</div>
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--accent-green)]" /> Canopy</div>
                  <div className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-[var(--text-muted)]" /> Background</div>
                </div>

                <div className="flex-1 min-h-[160px] relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={NDVI_HIST} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                      <XAxis dataKey="ndvi" stroke="#64748B" fontSize={10} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: '#131929', borderColor: '#2A3347', borderRadius: '6px' }} />
                      <Bar dataKey="road" stackId="a" fill="var(--accent-blue)" radius={[2, 2, 0, 0]} />
                      <Bar dataKey="canopy" stackId="a" fill="var(--accent-green)" />
                      <Bar dataKey="bg" stackId="a" fill="var(--text-muted)" />
                      <ReferenceLine x="0.1" stroke="var(--accent-blue)" strokeDasharray="4 3" label={{ value: "Road NDVI", fill: 'var(--accent-blue)', fontSize: 9, position: 'top' }} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {/* TAB 2: GRAPH ANALYSIS */}
          {activeTab === 'Graph Analysis' && (
            <>
              {/* Connectivity Ratio Hero Card */}
              <div className="glass-card border border-[var(--border)] rounded-lg p-4 select-none">
                <span className="text-[11px] text-[var(--text-muted)] uppercase font-bold tracking-wider block mb-3">Connectivity Ratio</span>
                
                <div className="flex items-center justify-between gap-4">
                  {/* Before */}
                  <div className="flex-1 flex flex-col">
                    <span className="text-[11px] text-[#64748B] mb-0.5">Before</span>
                    <span className="text-3xl font-bold font-mono text-[var(--accent-red)] mb-1">0.45</span>
                    <div className="w-full bg-[var(--border)] h-2 rounded-full relative">
                      <div className="h-full bg-[var(--accent-red)] rounded-full" style={{ width: '45%' }} />
                    </div>
                  </div>

                  {/* Arrow */}
                  <span className="text-xl font-bold text-[var(--text-muted)] self-center pb-2">&#8594;</span>

                  {/* After */}
                  <div className="flex-1 flex flex-col">
                    <span className="text-[11px] text-[var(--accent-green)] mb-0.5">After Healing</span>
                    <span className="text-3xl font-bold font-mono text-[var(--accent-green)] mb-1">
                      {(0.45 + 0.33 * animationProgress).toFixed(2)}
                    </span>
                    <div className="w-full bg-[var(--border)] h-2 rounded-full relative">
                      <div className="h-full bg-[var(--accent-green)] rounded-full" style={{ width: `${45 + 33 * animationProgress}%` }} />
                    </div>
                  </div>
                </div>

                <div className="mt-4 text-center">
                  <div className="text-xl font-extrabold text-[var(--accent-green)] font-outfit">
                    +{Math.round(73 * animationProgress)}% improvement
                  </div>
                  <span className="text-[11px] text-[var(--text-muted)] block mt-0.5">✓ Target exceeded (&ge; +30% required)</span>
                </div>
              </div>

              {/* 3 mini cards */}
              <div className="grid grid-cols-3 gap-2 select-none">
                <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md p-2.5 text-center flex flex-col justify-center">
                  <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">Nodes</span>
                  <span className="text-lg font-bold font-mono text-[#F1F5F9] mt-0.5">847</span>
                </div>
                <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md p-2.5 text-center flex flex-col justify-center">
                  <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">Edges</span>
                  <span className="text-lg font-bold font-mono text-[#F1F5F9] mt-0.5">1,203</span>
                </div>
                <div className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded-md p-2.5 text-center flex flex-col justify-center">
                  <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">Components</span>
                  <div className="flex items-center justify-center gap-1 mt-0.5">
                    <span className="text-rose-500 font-bold font-mono line-through">3</span>
                    <span className="text-xs text-[var(--text-muted)]">&#8594;</span>
                    <span className="text-emerald-400 font-bold font-mono">1</span>
                  </div>
                </div>
              </div>

              {/* APLS Score card */}
              <div className="glass-card border border-[var(--border)] rounded-lg p-5 relative overflow-hidden flex flex-col items-center justify-center text-center select-none min-h-[110px]">
                <div className="absolute top-0 left-0 right-0 h-[3px] bg-[var(--accent-green)]" />
                <span className="text-[11px] text-slate-400 font-semibold uppercase tracking-wider mb-1">APLS Score</span>
                <span className="text-3xl font-bold font-mono text-[var(--accent-green)] my-0.5">91.3%</span>
                <div className="flex flex-col items-center mt-1">
                  <span className="text-[12px] font-semibold text-[var(--accent-green)]">8.7% avg path error</span>
                  <span className="text-[11px] text-slate-400 font-semibold mt-0.5">Target &le; 12% • ✓ Met</span>
                </div>
              </div>

              {/* Gap Healing Log */}
              <div className="flex-1 flex flex-col overflow-hidden min-h-[220px]">
                <div className="flex items-center gap-2 mb-3 select-none">
                  <div className="w-[3px] h-[14px] bg-[var(--accent-blue)] rounded-sm" />
                  <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Gap Healing Log</span>
                </div>

                <div className="flex-1 overflow-y-auto border border-[var(--border)] rounded-md bg-[var(--bg-elevated)]">
                  {HEALED_EDGES.map(edge => {
                    const isAccepted = edge.pe >= 0.40;
                    return (
                      <div
                        key={edge.id}
                        className={`flex items-center justify-between p-2.5 border-b border-[var(--border)]/40 text-xs transition-colors hover:bg-[var(--bg-card)] ${
                          !isAccepted ? 'bg-red-950/5 border-l-2 border-[var(--accent-red)]' : ''
                        }`}
                      >
                        <span className="font-mono text-[var(--text-secondary)] font-semibold">{edge.id}</span>
                        <span className="text-[var(--text-muted)]">{edge.length}m</span>
                        <span className={`font-mono font-bold ${isAccepted ? 'text-[var(--accent-green)]' : 'text-[var(--accent-red)]'}`}>
                          p<sub>e</sub> = {edge.pe.toFixed(2)}
                        </span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-[3px] border ${
                          isAccepted
                            ? 'bg-emerald-950/20 border-emerald-900/60 text-[var(--accent-green)]'
                            : 'bg-red-950/20 border-red-900/60 text-[var(--accent-red)]'
                        }`}>
                          {isAccepted ? '✓ Healed' : '✗ Rejected'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* TAB 3: RESILIENCE */}
          {activeTab === 'Resilience' && (
            <>
              {/* RI Gauge */}
              <div className="glass-card border border-[var(--border)] rounded-lg p-4 text-center select-none">
                <span className="text-[11px] text-[var(--text-muted)] uppercase font-bold tracking-wider block text-left mb-2">Resilience Index (RI)</span>
                
                {/* SVG Gauge */}
                <div className="relative w-full max-h-[120px] flex items-center justify-center overflow-hidden">
                  <svg viewBox="0 0 240 130" className="w-[85%] h-full">
                    {/* Background Arc */}
                    <path
                      d="M 30,115 A 90,90 0 0,1 210,115"
                      fill="none"
                      stroke="var(--border)"
                      strokeWidth="14"
                      strokeLinecap="round"
                    />
                    
                    {/* Red Zone (0 - 40%) */}
                    <path
                      d="M 30,115 A 90,90 0 0,1 102,50"
                      fill="none"
                      stroke="var(--accent-red)"
                      strokeWidth="14"
                      opacity="0.25"
                    />

                    {/* Yellow Zone (40 - 70%) */}
                    <path
                      d="M 102,50 A 90,90 0 0,1 174,50"
                      fill="none"
                      stroke="var(--accent-yellow)"
                      strokeWidth="14"
                      opacity="0.25"
                    />

                    {/* Green Zone (70 - 100%) */}
                    <path
                      d="M 174,50 A 90,90 0 0,1 210,115"
                      fill="none"
                      stroke="var(--accent-green)"
                      strokeWidth="14"
                      opacity="0.25"
                    />

                    {/* Active Needle (74%) */}
                    {/* 0.74 of 180 degrees is 133.2 degrees from left end, which is 180 - 133.2 = 46.8 degrees from right baseline */}
                    <line
                      x1="120"
                      y1="115"
                      x2={120 + 70 * Math.cos((180 - 0.74 * 180) * Math.PI / 180)}
                      y2={115 - 70 * Math.sin((180 - 0.74 * 180) * Math.PI / 180)}
                      stroke="white"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                    
                    {/* Pivot */}
                    <circle cx="120" cy="115" r="5" fill="white" />
                    
                    {/* Value text */}
                    <text x="120" y="98" textAnchor="middle" fill="white" className="font-mono font-bold text-3xl">
                      {(0.74 * animationProgress).toFixed(2)}
                    </text>

                    {/* Zone labels */}
                    <text x="40" y="128" fill="var(--accent-red)" fontSize="9" fontWeight="600">LOW</text>
                    <text x="120" y="118" textAnchor="middle" fill="var(--accent-yellow)" fontSize="9" fontWeight="600">MED</text>
                    <text x="200" y="128" fill="var(--accent-green)" fontSize="9" fontWeight="600">HIGH</text>
                  </svg>
                </div>

                <div className="text-[11px] text-[#64748B] mt-2">
                  RI = 0.74 • Post-Disruption Estimate
                </div>
              </div>

              {/* RI component bars */}
              <div className="space-y-3.5 select-none border-b border-[var(--border)] pb-4 mb-2">
                {[
                  { label: 'LCC ratio', weight: 'w=0.6', width: '79%', color: 'var(--accent-blue)', val: 0.79 },
                  { label: 'Fiedler λ', weight: 'w=0.4', width: '68%', color: 'var(--accent-purple)', val: 0.68 }
                ].map(row => (
                  <div key={row.label} className="flex items-center gap-2.5 text-xs">
                    <span className="w-16 text-[var(--text-secondary)] font-semibold">{row.label}</span>
                    <span className="text-[10px] text-[var(--text-muted)] bg-slate-900 border border-[var(--border)] px-1.5 py-0.5 rounded-[3px] font-medium">{row.weight}</span>
                    <div className="flex-grow h-[6px] bg-[var(--border)] rounded-full overflow-hidden relative mx-1">
                      <div
                        className="h-full rounded-full transition-all duration-[800ms] ease-out"
                        style={{
                          width: `${row.val * animationProgress * 100}%`,
                          backgroundColor: row.color
                        }}
                      />
                    </div>
                    <span className="w-10 text-right font-mono text-white">{(row.val * animationProgress).toFixed(2)}</span>
                  </div>
                ))}
              </div>

              {/* 2x2 Risk Matrix Custom SVG plot */}
              <div className="flex flex-col gap-2.5 border-b border-[var(--border)] pb-4 mb-2">
                <div className="flex flex-col select-none">
                  <div className="flex items-center gap-2">
                    <div className="w-[3px] h-[14px] bg-[var(--accent-red)] rounded-sm" />
                    <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">Betweenness &times; Uncertainty Risk Matrix</span>
                  </div>
                  <span className="text-[10px] text-slate-400 italic mt-0.5">Novel stdBC metric — absent from published resilience literature</span>
                </div>

                <div className="bg-[#0B0F1A] border border-[var(--border)] rounded-lg p-3 relative flex items-center justify-center h-[230px]">
                  <svg viewBox="0 0 300 240" className="w-full h-full">
                    {/* Background quadrants */}
                    {/* Top-Right (Critical) */}
                    <rect
                      x="162" y="10" width="128" height="97"
                      fill="rgba(239,68,68,0.07)"
                      onClick={() => setSelectedQuadrant(selectedQuadrant === 'CRITICAL' ? null : 'CRITICAL')}
                      className={`cursor-pointer transition-all hover:fill-red-500/15 ${selectedQuadrant === 'CRITICAL' ? 'stroke border-[1.5px] stroke-red-500/35' : ''}`}
                    />
                    {/* Top-Left (Uncertain) */}
                    <rect
                      x="35" y="10" width="127" height="97"
                      fill="rgba(234,179,8,0.07)"
                      onClick={() => setSelectedQuadrant(selectedQuadrant === 'UNCERTAIN' ? null : 'UNCERTAIN')}
                      className={`cursor-pointer transition-all hover:fill-yellow-500/15 ${selectedQuadrant === 'UNCERTAIN' ? 'stroke border-[1.5px] stroke-yellow-500/35' : ''}`}
                    />
                    {/* Bottom-Right (Reliable) */}
                    <rect
                      x="162" y="107" width="128" height="98"
                      fill="rgba(59,130,246,0.07)"
                      onClick={() => setSelectedQuadrant(selectedQuadrant === 'RELIABLE' ? null : 'RELIABLE')}
                      className={`cursor-pointer transition-all hover:fill-blue-500/15 ${selectedQuadrant === 'RELIABLE' ? 'stroke border-[1.5px] stroke-blue-500/35' : ''}`}
                    />
                    {/* Bottom-Left (Safe) */}
                    <rect
                      x="35" y="107" width="127" height="98"
                      fill="rgba(34,197,94,0.07)"
                      onClick={() => setSelectedQuadrant(selectedQuadrant === 'SAFE' ? null : 'SAFE')}
                      className={`cursor-pointer transition-all hover:fill-green-500/15 ${selectedQuadrant === 'SAFE' ? 'stroke border-[1.5px] stroke-green-500/35' : ''}`}
                    />

                    {/* Divider Lines */}
                    <line x1="162" y1="10" x2="162" y2="205" stroke="#2A3347" strokeWidth="1" strokeDasharray="3,3" />
                    <line x1="35" y1="107" x2="290" y2="107" stroke="#2A3347" strokeWidth="1" strokeDasharray="3,3" />

                    {/* Quadrant Labels */}
                    <text x="285" y="26" textAnchor="end" fill="#EF4444" fontSize="10" fontWeight="600" opacity="0.6">&#9888; CRITICAL</text>
                    <text x="40" y="26" textAnchor="start" fill="#EAB308" fontSize="10" fontWeight="600" opacity="0.6">UNCERTAIN</text>
                    <text x="285" y="200" textAnchor="end" fill="#3B82F6" fontSize="10" fontWeight="600" opacity="0.6">RELIABLE</text>
                    <text x="40" y="200" textAnchor="start" fill="#22C55E" fontSize="10" fontWeight="600" opacity="0.6">SAFE</text>

                    {/* Axes */}
                    <text x="162" y="228" textAnchor="middle" fill="#475569" fontSize="10" fontWeight="600">meanBC (Traffic Criticality) &rarr;</text>
                    <text x="-107" y="12" fill="#475569" fontSize="10" fontWeight="600" transform="rotate(-90)">stdBC (Variance) &rarr;</text>
                    
                    {/* Tick labels */}
                    <text x="35" y="217" fill="#475569" fontSize="9" textAnchor="middle">0</text>
                    <text x="162" y="217" fill="#475569" fontSize="9" textAnchor="middle">0.5</text>
                    <text x="290" y="217" fill="#475569" fontSize="9" textAnchor="middle">1.0</text>
                    <text x="28" y="13" fill="#475569" fontSize="9" textAnchor="end">1.0</text>
                    <text x="28" y="108" fill="#475569" fontSize="9" textAnchor="end">0.5</text>
                    <text x="28" y="205" fill="#475569" fontSize="9" textAnchor="end">0</text>

                    {/* Plot Points */}
                    {NODES.map(node => {
                      const x = 35 + node.meanBC * 255;
                      const y = 205 - node.stdBC * 195;
                      const risk = getNodeRisk(node);
                      const isHighlighted = activeNodeId === node.id;
                      
                      return (
                        <circle
                          key={node.id}
                          cx={x}
                          cy={y}
                          r={isHighlighted ? 7.5 : 4.5}
                          fill={risk.color}
                          stroke={isHighlighted ? '#FFFFFF' : '#0B0F1A'}
                          strokeWidth={isHighlighted ? 2.5 : 1}
                          className="transition-all duration-150 cursor-pointer"
                          onClick={(e) => {
                            e.stopPropagation();
                            setActiveNodeId(node.id === activeNodeId ? null : node.id);
                          }}
                        />
                      );
                    })}
                  </svg>
                </div>
              </div>

              {/* Satellite Re-tasking Recommendation */}
              <div className="flex flex-col gap-2">
                <div className="flex flex-col select-none">
                  <div className="flex items-center gap-2">
                    <div className="w-[3px] h-[14px] bg-[var(--accent-blue)] rounded-sm" />
                    <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">📡 Satellite Re-tasking Priority</span>
                  </div>
                  <span className="text-[10px] text-slate-400 mt-0.5">Nodes: high meanBC + high &sigma;BC · Re-observe within 48hr</span>
                </div>

                <div className="flex flex-col gap-4">
                  {[
                    { id: '0312', lat: '12.97°N', lng: '77.59°E', bc: '0.81', std: '0.72', p: 'PRIORITY 1', color: 'var(--accent-red)' },
                    { id: '0813', lat: '12.97°N', lng: '77.59°E', bc: '0.91', std: '0.83', p: 'PRIORITY 2', color: '#F97316' },
                    { id: '0523', lat: '12.97°N', lng: '77.59°E', bc: '0.88', std: '0.79', p: 'PRIORITY 3', color: 'var(--accent-yellow)' }
                  ].map(item => {
                    const isSelected = activeNodeId === item.id;
                    return (
                      <div
                        key={item.id}
                        onClick={() => setActiveNodeId(item.id === activeNodeId ? null : item.id)}
                        className={`bg-[var(--bg-elevated)] border rounded-md p-3 px-3.5 flex items-center gap-3 cursor-pointer transition-all duration-150 ${
                          isSelected ? 'bg-[var(--bg-elevated)] shadow-lg' : 'border-[var(--border)] hover:border-blue-500/35'
                        }`}
                        style={{ borderColor: isSelected ? item.color : 'var(--border)' }}
                      >
                        <div
                          className="text-[9px] font-bold text-white px-2 py-0.5 rounded-[4px] shrink-0"
                          style={{ backgroundColor: item.color }}
                        >
                          {item.p}
                        </div>
                        <div className="flex flex-col">
                          <span className="text-xs font-bold text-[#F1F5F9] font-outfit">Node {item.id}</span>
                          <span className="text-[10px] text-[#64748B] font-mono mt-0.5">{item.lat} &nbsp; {item.lng}</span>
                        </div>
                        <div className="ml-auto text-right font-mono text-[10px] text-[var(--text-secondary)] flex flex-col gap-0.5">
                          <span>BC = {item.bc}</span>
                          <span className="text-[var(--accent-purple)]">&sigma; = {item.std}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* TAB 4: ALERTS */}
          {activeTab === 'Alerts' && (
            <div className="flex-1 flex flex-col overflow-hidden select-none pb-4">
              <div className="flex items-center gap-2 mb-6">
                <div className="w-[3px] h-[14px] bg-[var(--accent-red)] rounded-sm" />
                <span className="text-[10px] font-bold tracking-widest text-slate-400 uppercase">SYSTEM ALERTS</span>
              </div>

              <div className="flex flex-grow flex-col gap-6 overflow-y-auto pr-0.5">
                {ALERTS.map(alert => (
                  <div
                    key={alert.id}
                    className="bg-[var(--bg-card)] border border-[var(--border)] hover:bg-[var(--bg-elevated)] hover:border-[var(--border-bright)] rounded-md p-5 pl-6 relative overflow-hidden transition-all duration-150"
                  >
                    {/* Left border strip */}
                    <div
                      className="absolute left-0 top-0 bottom-0 w-[3px]"
                      style={{
                        backgroundColor:
                          alert.level === 'critical'
                            ? 'var(--accent-red)'
                            : alert.level === 'warning'
                            ? 'var(--accent-yellow)'
                            : 'var(--accent-blue)'
                      }}
                    />
                    
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-[#F1F5F9]">{alert.title}</span>
                      <span className="text-[10px] text-[var(--text-muted)] font-mono">{alert.time}</span>
                    </div>
                    
                    <p className="text-[11px] text-[#64748B] mt-1.5 leading-relaxed">
                      {alert.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

        </aside>

      </div>

    </div>
  );
}

// Map Event Listener Helper Sub-Component
function MapEventHandler({ onMouseMove }) {
  const map = useMap();
  useEffect(() => {
    const handleMove = (e) => {
      onMouseMove(e.latlng);
    };
    map.on('mousemove', handleMove);
    return () => {
      map.off('mousemove', handleMove);
    };
  }, [map, onMouseMove]);
  return null;
}
