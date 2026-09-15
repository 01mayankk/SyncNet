'use client';

import React, { useState } from 'react';
import {
  simulateThrottle,
  simulateDecay,
  simulateAppeal,
  SimulationActionResponse,
} from '@/services/api';
import { Sliders, Flame, ShieldCheck, Clock, History, AlertTriangle } from 'lucide-react';

interface ReactiveControlPanelProps {
  selectedClusterId: string | null;
  onRefreshClusters: () => void;
}

export const ReactiveControlPanel: React.FC<ReactiveControlPanelProps> = ({
  selectedClusterId,
  onRefreshClusters,
}) => {
  const [targetEntity, setTargetEntity] = useState<string>(selectedClusterId || 'cluster_00');
  const [throttleRate, setThrottleRate] = useState<number>(80);
  const [throttleReason, setThrottleReason] = useState<string>('Coordinated bot campaign detected via GraphSAGE');
  
  const [timeDelta, setTimeDelta] = useState<number>(2.0);
  const [lambdaDecay, setLambdaDecay] = useState<number>(0.1);
  
  const [appealReason, setAppealReason] = useState<string>('Submitted verification proof of organic activity.');

  const [logs, setLogs] = useState<SimulationActionResponse[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  const handleThrottle = async () => {
    setIsSubmitting(true);
    try {
      const res = await simulateThrottle(targetEntity, throttleRate, throttleReason);
      setLogs((prev) => [res, ...prev]);
      onRefreshClusters();
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDecay = async () => {
    setIsSubmitting(true);
    try {
      const res = await simulateDecay(targetEntity, timeDelta, lambdaDecay);
      setLogs((prev) => [res, ...prev]);
      onRefreshClusters();
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAppeal = async () => {
    setIsSubmitting(true);
    try {
      const res = await simulateAppeal(targetEntity, appealReason);
      setLogs((prev) => [res, ...prev]);
      onRefreshClusters();
    } catch (e) {
      console.error(e);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
          <Sliders className="w-5 h-5 text-cyan-400" />
          Reactive Throttling Decision Simulator
        </h2>
        <span className="text-xs text-amber-400 bg-amber-500/10 px-2.5 py-1 rounded border border-amber-500/20 font-semibold flex items-center gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5" />
          Simulation Mode (Non-Enforcement Prototype)
        </span>
      </div>

      {/* Target Entity Selector */}
      <div className="glass-card p-4 space-y-2">
        <label className="text-xs font-semibold text-slate-300 block">
          Target Cluster / Entity ID
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={targetEntity}
            onChange={(e) => setTargetEntity(e.target.value)}
            placeholder="e.g. cluster_00 or usr_0001"
            className="bg-slate-900 border border-slate-700 text-cyan-400 font-mono text-sm rounded-lg px-3 py-2 flex-1 focus:outline-none focus:border-cyan-400"
          />
          <select
            value={targetEntity}
            onChange={(e) => setTargetEntity(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-cyan-400"
          >
            {['cluster_00', 'cluster_01', 'cluster_02', 'cluster_03', 'cluster_04', 'cluster_05', 'cluster_06', 'cluster_07'].map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Simulation Controls Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* 1. Throttling Trigger Card */}
        <div className="glass-card p-4 space-y-3 flex flex-col justify-between border-rose-500/30">
          <div>
            <div className="flex items-center gap-2 mb-2 text-rose-400 font-bold text-sm">
              <Flame className="w-4 h-4" />
              Simulate Rate Throttling
            </div>
            <p className="text-[11px] text-slate-400 mb-3">
              Applies rate limit throttling to high-risk coordinated clusters.
            </p>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1 text-slate-300">
                  <span>Throttle Percentage</span>
                  <span className="font-mono text-rose-400 font-bold">{throttleRate}%</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={throttleRate}
                  onChange={(e) => setThrottleRate(Number(e.target.value))}
                />
              </div>

              <div>
                <label className="text-[11px] text-slate-400 block mb-1">Trigger Reason</label>
                <input
                  type="text"
                  value={throttleReason}
                  onChange={(e) => setThrottleReason(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2.5 py-1.5 focus:outline-none focus:border-rose-400"
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleThrottle}
            disabled={isSubmitting}
            className="cyber-btn cyber-btn-danger w-full text-xs mt-3"
          >
            Trigger Rate Throttle ({throttleRate}%)
          </button>
        </div>

        {/* 2. Exponential Decay Card */}
        <div className="glass-card p-4 space-y-3 flex flex-col justify-between border-purple-500/30">
          <div>
            <div className="flex items-center gap-2 mb-2 text-purple-400 font-bold text-sm">
              <Clock className="w-4 h-4" />
              Simulate Risk Score Decay
            </div>
            <p className="text-[11px] text-slate-400 mb-3 font-mono">
              S(t) = max(S_min, S(0) · e^(-λt))
            </p>

            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1 text-slate-300">
                  <span>Simulated Time Elapsed (Hours)</span>
                  <span className="font-mono text-purple-400 font-bold">{timeDelta}h</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="12.0"
                  step="0.5"
                  value={timeDelta}
                  onChange={(e) => setTimeDelta(Number(e.target.value))}
                />
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1 text-slate-300">
                  <span>Decay Constant λ</span>
                  <span className="font-mono text-purple-400 font-bold">{lambdaDecay}</span>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="0.30"
                  step="0.02"
                  value={lambdaDecay}
                  onChange={(e) => setLambdaDecay(Number(e.target.value))}
                />
              </div>
            </div>
          </div>

          <button
            onClick={handleDecay}
            disabled={isSubmitting}
            className="cyber-btn cyber-btn-warning w-full text-xs mt-3"
          >
            Simulate Time Decay ({timeDelta}h)
          </button>
        </div>

        {/* 3. Appeal Workflow Card */}
        <div className="glass-card p-4 space-y-3 flex flex-col justify-between border-blue-500/30">
          <div>
            <div className="flex items-center gap-2 mb-2 text-blue-400 font-bold text-sm">
              <ShieldCheck className="w-4 h-4" />
              Simulate Appeal Submission
            </div>
            <p className="text-[11px] text-slate-400 mb-3">
              Submits justification evidence to lift active throttling pending review.
            </p>

            <div className="space-y-2">
              <label className="text-[11px] text-slate-400 block">Appeal Evidence Justification</label>
              <textarea
                rows={3}
                value={appealReason}
                onChange={(e) => setAppealReason(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded p-2 focus:outline-none focus:border-blue-400"
              />
            </div>
          </div>

          <button
            onClick={handleAppeal}
            disabled={isSubmitting}
            className="cyber-btn cyber-btn-primary w-full text-xs mt-3"
          >
            Submit Appeal & Lift Throttle
          </button>
        </div>
      </div>

      {/* Reactive Log Feed */}
      <div className="glass-card p-4 space-y-3">
        <h3 className="text-xs font-bold text-slate-200 flex items-center gap-2">
          <History className="w-4 h-4 text-cyan-400" />
          Reactive Simulation Event Stream
        </h3>

        {logs.length === 0 ? (
          <p className="text-xs text-slate-500 italic py-2">
            No reactive simulation actions performed yet. Trigger a slider above to test state machine transitions.
          </p>
        ) : (
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {logs.map((log, idx) => (
              <div
                key={idx}
                className="p-2.5 bg-slate-900/90 rounded-lg border border-slate-800 text-xs flex items-center justify-between gap-3"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-cyan-400 font-bold">{log.entity_id}</span>
                  <span className="text-slate-400 text-[10px]">
                    {log.previous_state} → <strong className="text-slate-200">{log.current_state}</strong>
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] text-slate-300 truncate max-w-xs">{log.message}</span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {new Date(log.timestamp * 1000).toLocaleTimeString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
