'use client';

import React, { useEffect, useState, useCallback } from 'react';
import {
  fetchHealth,
  fetchClusters,
  fetchGraphTopology,
  HealthResponse,
  ClusterSummary,
  GraphTopologyResponse,
  NodeItem,
} from '@/services/api';
import { NetworkGraph } from '@/components/NetworkGraph';
import { ClusterPanel } from '@/components/ClusterPanel';
import { ReactiveControlPanel } from '@/components/ReactiveControlPanel';
import { NodeModal } from '@/components/NodeModal';
import {
  ShieldCheck,
  Activity,
  Cpu,
  Server,
  Network,
  Users,
  Sliders,
  AlertTriangle,
  RefreshCw,
  Zap,
} from 'lucide-react';

export default function SyncNetDashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [clusters, setClusters] = useState<ClusterSummary[]>([]);
  const [graphData, setGraphData] = useState<GraphTopologyResponse | null>(null);

  const [activeTab, setActiveTab] = useState<'topology' | 'clusters' | 'simulation'>('topology');
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NodeItem | null>(null);

  const [loading, setLoading] = useState<boolean>(true);

  const loadAllData = useCallback(async () => {
    setLoading(true);
    try {
      const [hRes, cRes, gRes] = await Promise.all([
        fetchHealth(),
        fetchClusters(),
        fetchGraphTopology(500),
      ]);
      setHealth(hRes);
      setClusters(cRes);
      setGraphData(gRes);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  // Aggregate Metrics
  const totalBots = clusters.reduce((acc, c) => acc + c.num_bots, 0);
  const totalNodes = graphData ? graphData.total_nodes : 1000;
  const highRiskClusters = clusters.filter((c) => c.coordination_score >= 0.60).length;
  const throttledClusters = clusters.filter((c) => c.status === 'THROTTLED' || c.status === 'ESCALATED').length;

  return (
    <main className="dashboard-container">
      {/* Header Bar */}
      <header className="cyber-header">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-2xl bg-cyan-500/10 border border-cyan-500/40 text-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
          </div>
          <div>
            <h1 className="cyber-title">
              SyncNet Operations Defense Center
            </h1>
            <p className="cyber-subtitle">
              Coordinated Bot-Network Detection & Reactive Throttling • <span className="text-cyan-400 font-mono">v0.1.0</span>
            </p>
          </div>
        </div>

        {/* Hardware Status Widget */}
        <div className="hardware-widget">
          <div className="hardware-pill">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-bold">GPU ACCELERATOR</span>
              <span className="font-mono text-emerald-400 font-semibold">
                {health ? health.gpu_name.replace('Laptop GPU', '') || 'RTX 5050' : 'RTX 5050'}
              </span>
            </div>
          </div>

          <div className="hardware-pill">
            <Server className="w-4 h-4 text-cyan-400" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-bold">MEM BUDGET</span>
              <span className="font-mono text-cyan-400 font-semibold">
                VRAM: 8GB | RAM: 16GB
              </span>
            </div>
          </div>

          <button
            onClick={loadAllData}
            className="p-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            title="Refresh System Data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
          </button>
        </div>
      </header>

      {/* Top Overview Metrics Row */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span>Monitored Accounts</span>
            <Users className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="metric-value text-slate-100">{totalNodes}</div>
          <p className="metric-footer">1,000 Accounts | 4,332 Edges</p>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Bot Detection Ratio</span>
            <Activity className="w-4 h-4 text-rose-400" />
          </div>
          <div className="metric-value text-rose-400">
            {((totalBots / Math.max(1, totalNodes)) * 100).toFixed(1)}%
          </div>
          <p className="metric-footer">{totalBots} Bot Accounts Detected</p>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>High-Risk Clusters</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="metric-value text-amber-400">
            {highRiskClusters} / {clusters.length || 8}
          </div>
          <p className="metric-footer">S_coord(C_k) ≥ 0.60</p>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Simulated Throttled</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="metric-value text-purple-400">
            {throttledClusters} Clusters
          </div>
          <p className="metric-footer">Reactive Decision Engine</p>
        </div>
      </div>

      {/* Main Content Tabs Navigation */}
      <div className="tab-bar">
        <button
          onClick={() => setActiveTab('topology')}
          className={`tab-btn ${activeTab === 'topology' ? 'tab-btn-active' : ''}`}
        >
          <Network className="w-4 h-4" />
          Interaction Network Graph
        </button>

        <button
          onClick={() => setActiveTab('clusters')}
          className={`tab-btn ${activeTab === 'clusters' ? 'tab-btn-active' : ''}`}
        >
          <Users className="w-4 h-4" />
          Cluster Coordination Intelligence
        </button>

        <button
          onClick={() => setActiveTab('simulation')}
          className={`tab-btn ${activeTab === 'simulation' ? 'tab-btn-active' : ''}`}
        >
          <Sliders className="w-4 h-4" />
          Reactive Throttling Controls
        </button>
      </div>

      {/* Main Tab Views */}
      <div className="w-full">
        {activeTab === 'topology' && (
          <div className="w-full">
            {graphData && (
              <NetworkGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                selectedClusterId={selectedClusterId}
                onSelectNode={(node) => setSelectedNode(node)}
                onSelectCluster={(cid) => setSelectedClusterId(cid)}
              />
            )}
          </div>
        )}

        {activeTab === 'clusters' && (
          <ClusterPanel
            clusters={clusters}
            selectedClusterId={selectedClusterId}
            onSelectCluster={(cid) => {
              setSelectedClusterId(cid);
              setActiveTab('topology');
            }}
            onSimulateAction={() => loadAllData()}
          />
        )}

        {activeTab === 'simulation' && (
          <ReactiveControlPanel
            selectedClusterId={selectedClusterId}
            onRefreshClusters={() => loadAllData()}
          />
        )}
      </div>

      {/* Node Detail Inspection Modal */}
      {selectedNode && (
        <NodeModal node={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </main>
  );
}
