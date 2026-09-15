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
    <main className="min-h-screen p-4 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Header Bar */}
      <header className="glass-card p-5 border-cyan-500/30 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/40 text-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.3)]">
            <ShieldCheck className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400">
              SyncNet Operations Defense Center
            </h1>
            <p className="text-xs text-slate-400 flex items-center gap-2 mt-0.5">
              <span>Coordinated Bot-Network Detection & Reactive Throttling</span>
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              <span className="text-cyan-400 font-mono">v0.1.0</span>
            </p>
          </div>
        </div>

        {/* Hardware Status Widget */}
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-2 rounded-xl border border-slate-800">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="text-slate-400 block text-[10px]">GPU ACCELERATOR</span>
              <span className="font-mono text-emerald-400 font-semibold">
                {health ? health.gpu_name.split(' ')[2] || 'RTX 5050' : 'RTX 5050'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-2 rounded-xl border border-slate-800">
            <Server className="w-4 h-4 text-cyan-400" />
            <div>
              <span className="text-slate-400 block text-[10px]">MEM BUDGET</span>
              <span className="font-mono text-cyan-400 font-semibold">
                VRAM: 8GB | RAM: 16GB
              </span>
            </div>
          </div>

          <button
            onClick={loadAllData}
            className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
            title="Refresh System Data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      {/* Top Overview Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Monitored Accounts</span>
            <Users className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-extrabold font-mono text-slate-100">{totalNodes}</div>
          <p className="text-[10px] text-slate-400">1000 Accounts | 4,332 Edges</p>
        </div>

        <div className="glass-card p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Bot Ratio</span>
            <Activity className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-extrabold font-mono text-rose-400">
            {((totalBots / Math.max(1, totalNodes)) * 100).toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-400">{totalBots} Bot Accounts Detected</p>
        </div>

        <div className="glass-card p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>High-Risk Clusters</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-extrabold font-mono text-amber-400">
            {highRiskClusters} / {clusters.length || 8}
          </div>
          <p className="text-[10px] text-slate-400">S_coord(C_k) ≥ 0.60</p>
        </div>

        <div className="glass-card p-4 space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Simulated Throttled</span>
            <Zap className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-extrabold font-mono text-purple-400">
            {throttledClusters} Clusters
          </div>
          <p className="text-[10px] text-slate-400">Reactive Decision Engine</p>
        </div>
      </div>

      {/* Main Content Tabs Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveTab('topology')}
          className={`cyber-btn text-xs ${
            activeTab === 'topology'
              ? 'cyber-btn-primary'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 border-slate-800'
          }`}
        >
          <Network className="w-4 h-4" />
          Interaction Network Graph
        </button>

        <button
          onClick={() => setActiveTab('clusters')}
          className={`cyber-btn text-xs ${
            activeTab === 'clusters'
              ? 'cyber-btn-primary'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 border-slate-800'
          }`}
        >
          <Users className="w-4 h-4" />
          Cluster Coordination Intelligence
        </button>

        <button
          onClick={() => setActiveTab('simulation')}
          className={`cyber-btn text-xs ${
            activeTab === 'simulation'
              ? 'cyber-btn-primary'
              : 'bg-slate-900/60 text-slate-400 hover:text-slate-200 border-slate-800'
          }`}
        >
          <Sliders className="w-4 h-4" />
          Reactive Throttling Controls
        </button>
      </div>

      {/* Main Tab Views */}
      <div className="space-y-6">
        {activeTab === 'topology' && (
          <div className="space-y-4">
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
