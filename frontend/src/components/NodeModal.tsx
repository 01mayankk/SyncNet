'use client';

import React, { useEffect, useState } from 'react';
import { NodeItem, NodeDetailResponse, fetchNodeDetail } from '@/services/api';
import { User, Cpu, X, Network, Grid } from 'lucide-react';

interface NodeModalProps {
  node: NodeItem | null;
  onClose: () => void;
}

export const NodeModal: React.FC<NodeModalProps> = ({ node, onClose }) => {
  const [detail, setDetail] = useState<NodeDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (node) {
      setLoading(true);
      fetchNodeDetail(node.id)
        .then((d) => setDetail(d))
        .catch(console.error)
        .finally(() => setLoading(false));
    } else {
      setDetail(null);
    }
  }, [node]);

  if (!node) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
      <div className="glass-card w-full max-w-2xl p-6 relative border-cyan-500/40 shadow-2xl space-y-4">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-slate-100 hover:bg-slate-700 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-3 border-b border-slate-800 pb-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <User className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-100 font-mono flex items-center gap-2">
              {node.id}
              <span className="text-xs font-normal px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {node.cluster_id}
              </span>
            </h3>
            <p className="text-xs text-slate-400">GNN Structural Representation Inspection</p>
          </div>
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">GNN Classification</span>
            <span
              className={`text-sm font-bold block mt-1 ${
                node.predicted_label === 1 ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {node.predicted_label === 1 ? 'Bot Network' : 'Human Account'}
            </span>
          </div>

          <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Bot Probability</span>
            <span className="text-sm font-mono font-bold text-cyan-400 mt-1 block">
              {(node.bot_probability * 100).toFixed(1)}%
            </span>
          </div>

          <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Ground Truth</span>
            <span className="text-sm font-semibold text-slate-200 mt-1 block">
              {node.label === 1 ? 'Bot' : 'Human'}
            </span>
          </div>

          <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Graph Degree</span>
            <span className="text-sm font-mono font-bold text-purple-400 mt-1 block">
              {detail ? detail.degree : 12}
            </span>
          </div>
        </div>

        {/* 64-Dimensional Embedding Vector Grid */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
            <span className="flex items-center gap-1.5 text-cyan-400">
              <Cpu className="w-4 h-4" />
              64-Dimensional GraphSAGE Embedding Vector Z_node
            </span>
            <span className="text-[11px] text-slate-500 font-mono">
              [z_00 .. z_63]
            </span>
          </div>

          {loading ? (
            <div className="p-8 text-center text-xs text-slate-400 italic">
              Loading structural embeddings...
            </div>
          ) : detail ? (
            <div className="grid grid-cols-8 gap-1.5 bg-slate-950 p-3 rounded-xl border border-slate-800 max-h-48 overflow-y-auto">
              {detail.embedding_64d.map((val, idx) => {
                const normVal = Math.min(1.0, Math.max(0.0, (val + 1) / 2));
                return (
                  <div
                    key={idx}
                    className="p-1 rounded bg-slate-900 text-center border border-slate-800 hover:border-cyan-500 transition-colors"
                    title={`z_${idx.toString().padStart(2, '0')}: ${val.toFixed(4)}`}
                  >
                    <span className="block text-[8px] text-slate-500 font-mono">
                      z_{idx.toString().padStart(2, '0')}
                    </span>
                    <span
                      className="block text-[10px] font-mono font-semibold truncate"
                      style={{
                        color: val >= 0 ? '#10b981' : '#ef4444',
                      }}
                    >
                      {val.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
