'use client';

import React, { useState } from 'react';
import { ClusterSummary, ClusterDetail, fetchClusterDetail } from '@/services/api';
import { ShieldAlert, Users, Network, FileText, ChevronRight, Activity } from 'lucide-react';

interface ClusterPanelProps {
  clusters: ClusterSummary[];
  selectedClusterId: string | null;
  onSelectCluster: (clusterId: string) => void;
  onSimulateAction: (clusterId: string, actionType: string) => void;
}

export const ClusterPanel: React.FC<ClusterPanelProps> = ({
  clusters,
  selectedClusterId,
  onSelectCluster,
  onSimulateAction,
}) => {
  const [activeDetail, setActiveDetail] = useState<ClusterDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);

  const handleInspectCluster = async (clusterId: string) => {
    onSelectCluster(clusterId);
    setLoadingDetail(true);
    try {
      const detail = await fetchClusterDetail(clusterId);
      setActiveDetail(detail);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingDetail(false);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status.toUpperCase()) {
      case 'THROTTLED':
        return <span className="status-pill status-throttled">THROTTLED</span>;
      case 'ESCALATED':
        return <span className="status-pill status-escalated">ESCALATED</span>;
      case 'FLAGGED':
        return <span className="status-pill status-flagged">FLAGGED</span>;
      case 'APPEALED':
        return <span className="status-pill status-appealed">APPEALED</span>;
      case 'DECAYING':
        return <span className="status-pill status-decaying">DECAYING</span>;
      default:
        return <span className="status-pill status-normal">NORMAL</span>;
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-cyan-400" />
          Cluster Coordination Intelligence
        </h2>
        <span className="text-xs text-slate-400">8 Interaction Clusters Active</span>
      </div>

      {/* Cluster Grid Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {clusters.map((c) => {
          const isSelected = selectedClusterId === c.cluster_id;
          const isHighRisk = c.coordination_score >= 0.60;

          return (
            <div
              key={c.cluster_id}
              onClick={() => handleInspectCluster(c.cluster_id)}
              className={`glass-card p-4 cursor-pointer relative overflow-hidden transition-all ${
                isSelected
                  ? 'border-cyan-400 ring-2 ring-cyan-400/20 bg-slate-900/90'
                  : 'hover:border-slate-600'
              }`}
            >
              {/* High Risk Glowing Accent Line */}
              {isHighRisk && (
                <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-rose-500 via-amber-500 to-rose-500 animate-pulse" />
              )}

              <div className="flex items-center justify-between mb-3">
                <span className="font-mono text-sm font-bold text-cyan-400">{c.cluster_id}</span>
                {getStatusBadge(c.status)}
              </div>

              {/* Coordination Score Display */}
              <div className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Coordination Score S(C_k)</span>
                  <span
                    className={`font-mono font-bold ${
                      c.coordination_score >= 0.6
                        ? 'text-rose-400'
                        : c.coordination_score >= 0.35
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {c.coordination_score.toFixed(4)}
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700">
                  <div
                    className={`h-full transition-all duration-500 ${
                      c.coordination_score >= 0.6
                        ? 'bg-gradient-to-r from-amber-500 to-rose-500'
                        : c.coordination_score >= 0.35
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${c.coordination_score * 100}%` }}
                  />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-2 text-[11px] bg-slate-950/40 p-2.5 rounded-lg border border-slate-800/60 mb-3">
                <div>
                  <span className="text-slate-400 block">Bot Ratio</span>
                  <span className="font-mono font-semibold text-slate-200">
                    {(c.bot_ratio * 100).toFixed(1)}% ({c.num_bots}/{c.size})
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Edge Density</span>
                  <span className="font-mono font-semibold text-slate-200">
                    {c.edge_density.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Content Similarity</span>
                  <span className="font-mono font-semibold text-slate-200">
                    {c.content_similarity.toFixed(4)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Risk Tier</span>
                  <span
                    className={`font-semibold ${
                      c.risk_tier === 'HIGH'
                        ? 'text-rose-400'
                        : c.risk_tier === 'MEDIUM'
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {c.risk_tier}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs text-cyan-400 font-semibold pt-1 border-t border-slate-800/80">
                <span>Inspect Cluster Details</span>
                <ChevronRight className="w-4 h-4" />
              </div>
            </div>
          );
        })}
      </div>

      {/* Cluster Detail Modal / Drawer */}
      {activeDetail && (
        <div className="glass-card p-5 mt-6 border-cyan-500/40 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-base font-bold text-slate-100 flex items-center gap-2">
                <Users className="w-4 h-4 text-cyan-400" />
                Cluster Inspection: <span className="font-mono text-cyan-400">{activeDetail.cluster_id}</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Total Members: {activeDetail.size} | Bot Accounts: {activeDetail.num_bots} | Status: {activeDetail.status}
              </p>
            </div>
            <button
              onClick={() => setActiveDetail(null)}
              className="text-xs text-slate-400 hover:text-slate-200 px-3 py-1 bg-slate-800 rounded border border-slate-700"
            >
              Close Panel
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="p-2.5">Account ID</th>
                  <th className="p-2.5">Ground Truth</th>
                  <th className="p-2.5">GNN Predicted</th>
                  <th className="p-2.5">Bot Probability</th>
                  <th className="p-2.5 text-right">Action Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {activeDetail.members.slice(0, 10).map((m) => (
                  <tr key={m.account_id} className="hover:bg-slate-800/40">
                    <td className="p-2.5 font-mono text-cyan-400 font-semibold">{m.account_id}</td>
                    <td className="p-2.5">{m.ground_truth === 1 ? 'Bot' : 'Human'}</td>
                    <td className="p-2.5">
                      <span className={m.predicted_label === 1 ? 'text-rose-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                        {m.predicted_label === 1 ? 'Bot Network' : 'Human'}
                      </span>
                    </td>
                    <td className="p-2.5 font-mono">{(m.bot_probability * 100).toFixed(1)}%</td>
                    <td className="p-2.5 text-right">
                      <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                        {activeDetail.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
