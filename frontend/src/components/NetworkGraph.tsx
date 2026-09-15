'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { NodeItem, EdgeItem } from '@/services/api';
import { ZoomIn, ZoomOut, RefreshCw, Eye, EyeOff, Layers } from 'lucide-react';

interface NetworkGraphProps {
  nodes: NodeItem[];
  edges: EdgeItem[];
  selectedClusterId: string | null;
  onSelectNode: (node: NodeItem) => void;
  onSelectCluster: (clusterId: string | null) => void;
}

interface ForceNode extends NodeItem {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  nodes,
  edges,
  selectedClusterId,
  onSelectNode,
  onSelectCluster,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [graphNodes, setGraphNodes] = useState<ForceNode[]>([]);
  const [zoom, setZoom] = useState<number>(1.0);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<ForceNode | null>(null);
  const [showLabels, setShowLabels] = useState<boolean>(false);
  const [activeCluster, setActiveCluster] = useState<string>('ALL');

  // Initialize nodes with 2D positions in cluster rings
  useEffect(() => {
    if (!nodes || nodes.length === 0) return;

    const width = 800;
    const height = 600;

    const clusterAngles: { [key: string]: number } = {};
    const clusterIds = Array.from(new Set(nodes.map((n) => n.cluster_id)));
    clusterIds.forEach((cid, idx) => {
      clusterAngles[cid] = (idx / clusterIds.length) * 2 * Math.PI;
    });

    const initialized: ForceNode[] = nodes.map((node) => {
      const angle = clusterAngles[node.cluster_id] || 0;
      const radiusOffset = 180 + (Math.random() * 80 - 40);
      const randomSpreadAngle = angle + (Math.random() * 0.8 - 0.4);

      const cx = width / 2 + Math.cos(randomSpreadAngle) * radiusOffset;
      const cy = height / 2 + Math.sin(randomSpreadAngle) * radiusOffset;

      return {
        ...node,
        x: cx,
        y: cy,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: node.bot_probability >= 0.5 ? 6 : 4.5,
      };
    });

    setGraphNodes(initialized);
  }, [nodes]);

  // Main Canvas Render Loop
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();

    // Apply Zoom & Pan Transformations
    ctx.translate(width / 2 + pan.x, height / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.translate(-width / 2, -height / 2);

    const nodeMap = new Map<string, ForceNode>();
    graphNodes.forEach((n) => nodeMap.set(n.id, n));

    // Render Edges
    ctx.lineWidth = 0.6;
    edges.forEach((edge) => {
      const src = nodeMap.get(edge.source);
      const tgt = nodeMap.get(edge.target);
      if (src && tgt) {
        const isClusterMatch =
          activeCluster === 'ALL' ||
          (src.cluster_id === activeCluster && tgt.cluster_id === activeCluster);

        ctx.strokeStyle = isClusterMatch
          ? 'rgba(6, 182, 212, 0.15)'
          : 'rgba(255, 255, 255, 0.03)';
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.stroke();
      }
    });

    // Render Nodes
    graphNodes.forEach((node) => {
      const isSelectedCluster = activeCluster === 'ALL' || node.cluster_id === activeCluster;
      const isHovered = hoveredNode?.id === node.id;

      let color = node.bot_probability >= 0.5 ? '#ef4444' : '#10b981';
      if (!isSelectedCluster) color = '#475569';

      ctx.beginPath();
      ctx.arc(node.x, node.y, isHovered ? node.radius * 1.6 : node.radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      // Glowing aura for bot nodes or hovered nodes
      if (node.bot_probability >= 0.5 || isHovered) {
        ctx.shadowColor = color;
        ctx.shadowBlur = isHovered ? 15 : 8;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = isHovered ? 2 : 0.8;
        ctx.stroke();
        ctx.shadowBlur = 0; // reset
      }

      // Render Text Label if enabled or hovered
      if (showLabels || isHovered) {
        ctx.font = isHovered ? '600 11px Inter' : '400 9px Inter';
        ctx.fillStyle = isHovered ? '#06b6d4' : 'rgba(241, 245, 249, 0.7)';
        ctx.fillText(node.id, node.x + node.radius + 4, node.y + 3);
      }
    });

    ctx.restore();
  }, [graphNodes, edges, zoom, pan, activeCluster, hoveredNode, showLabels]);

  useEffect(() => {
    render();
  }, [render]);

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = containerRef.current.clientHeight || 550;
        render();
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [render]);

  // Mouse Interaction Handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
      return;
    }

    // Node Hover Detection
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const width = canvas.width;
    const height = canvas.height;

    // Convert mouse canvas coordinates considering zoom and pan
    const worldX = (mx - width / 2 - pan.x) / zoom + width / 2;
    const worldY = (my - height / 2 - pan.y) / zoom + height / 2;

    const hovered = graphNodes.find((n) => {
      const dx = n.x - worldX;
      const dy = n.y - worldY;
      return Math.sqrt(dx * dx + dy * dy) <= n.radius * 2.5;
    });

    setHoveredNode(hovered || null);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (hoveredNode) {
      onSelectNode(hoveredNode);
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(Math.max(prev * zoomFactor, 0.4), 4.0));
  };

  const clusterOptions = ['ALL', 'cluster_00', 'cluster_01', 'cluster_02', 'cluster_03', 'cluster_04', 'cluster_05', 'cluster_06', 'cluster_07'];

  return (
    <div ref={containerRef} className="glass-card relative w-full h-[580px] overflow-hidden flex flex-col">
      {/* Top Overlay Controls Bar */}
      <div className="absolute top-4 left-4 right-4 z-10 flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-700/50 text-xs">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-slate-300">Cluster Filter:</span>
          <select
            value={activeCluster}
            onChange={(e) => {
              setActiveCluster(e.target.value);
              onSelectCluster(e.target.value === 'ALL' ? null : e.target.value);
            }}
            className="bg-slate-800 text-slate-200 border border-slate-600 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-400"
          >
            {clusterOptions.map((c) => (
              <option key={c} value={c}>
                {c === 'ALL' ? 'All Clusters (1,000 Nodes)' : c}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]" />
            <span className="text-slate-300">Human</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-[0_0_8px_#ef4444]" />
            <span className="text-slate-300">Bot / High Risk</span>
          </div>

          <div className="h-4 w-[1px] bg-slate-700 mx-1" />

          <button
            onClick={() => setShowLabels(!showLabels)}
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 transition-colors"
            title={showLabels ? 'Hide Labels' : 'Show Labels'}
          >
            {showLabels ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setZoom((prev) => Math.min(prev * 1.25, 4.0))}
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={() => setZoom((prev) => Math.max(prev * 0.8, 0.4))}
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={() => {
              setZoom(1.0);
              setPan({ x: 0, y: 0 });
            }}
            className="p-1.5 rounded hover:bg-slate-700 text-slate-300 transition-colors"
            title="Reset View"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas Viewport */}
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        onWheel={handleWheel}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Hover Node Tooltip */}
      {hoveredNode && (
        <div className="absolute bottom-4 left-4 z-10 p-3 bg-slate-900/90 backdrop-blur-md border border-cyan-500/40 rounded-xl shadow-xl text-xs space-y-1 max-w-xs">
          <div className="font-bold text-cyan-400 flex items-center justify-between">
            <span>{hoveredNode.id}</span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {hoveredNode.cluster_id}
            </span>
          </div>
          <div className="flex justify-between text-slate-300">
            <span>Bot Probability:</span>
            <span className={hoveredNode.bot_probability >= 0.5 ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
              {(hoveredNode.bot_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between text-slate-400 text-[10px]">
            <span>Predicted Class:</span>
            <span>{hoveredNode.predicted_label === 1 ? 'Bot Network' : 'Human Account'}</span>
          </div>
          <p className="text-[10px] text-cyan-400/80 italic mt-1">Click node to inspect 64-dim embedding vector</p>
        </div>
      )}
    </div>
  );
};
