export interface HealthResponse {
  status: string;
  environment: string;
  cuda_available: boolean;
  gpu_name: string;
  vram_total_mb: number;
  system_ram_budget_gb: number;
  timestamp: number;
}

export interface ClusterSummary {
  cluster_id: string;
  size: number;
  num_bots: number;
  bot_ratio: number;
  edge_density: number;
  content_similarity: number;
  coordination_score: number;
  risk_tier: string;
  status: string;
}

export interface ClusterMember {
  account_id: string;
  ground_truth: number;
  predicted_label: number;
  bot_probability: number;
}

export interface ClusterDetail extends ClusterSummary {
  members: ClusterMember[];
}

export interface NodeItem {
  id: string;
  label: number;
  predicted_label: number;
  bot_probability: number;
  cluster_id: string;
  embedding_preview: number[];
}

export interface EdgeItem {
  source: string;
  target: string;
  weight: number;
}

export interface GraphTopologyResponse {
  total_nodes: number;
  total_edges: number;
  timestamp: number;
  nodes: NodeItem[];
  edges: EdgeItem[];
}

export interface NodeDetailResponse {
  id: string;
  label: number;
  predicted_label: number;
  bot_probability: number;
  cluster_id: string;
  degree: number;
  embedding_64d: number[];
}

export interface SimulationActionResponse {
  entity_id: string;
  entity_type: string;
  previous_state: string;
  current_state: string;
  coordination_score: number;
  active_throttle_pct: number;
  timestamp: number;
  message: string;
}

const API_BASE = '/api/v1';

// Mock Generator for offline fallback
function getMockGraphData(): GraphTopologyResponse {
  const nodes: NodeItem[] = [];
  const edges: EdgeItem[] = [];

  for (let i = 1; i <= 300; i++) {
    const isBot = i <= 90;
    const cid = `cluster_${(i % 8).toString().padStart(2, '0')}`;
    const id = `usr_${i.toString().padStart(4, '0')}`;
    const botProb = isBot ? 0.85 + Math.random() * 0.14 : Math.random() * 0.25;

    nodes.push({
      id,
      label: isBot ? 1 : 0,
      predicted_label: botProb >= 0.5 ? 1 : 0,
      bot_probability: parseFloat(botProb.toFixed(4)),
      cluster_id: cid,
      embedding_preview: [
        parseFloat((Math.random() * 2 - 1).toFixed(4)),
        parseFloat((Math.random() * 2 - 1).toFixed(4)),
        parseFloat((Math.random() * 2 - 1).toFixed(4)),
        parseFloat((Math.random() * 2 - 1).toFixed(4)),
      ],
    });
  }

  // Generate inter-node interaction edges
  for (let i = 0; i < 450; i++) {
    const srcIdx = Math.floor(Math.random() * 300);
    let tgtIdx = Math.floor(Math.random() * 300);
    if (srcIdx !== tgtIdx) {
      edges.push({
        source: nodes[srcIdx].id,
        target: nodes[tgtIdx].id,
        weight: 1.0,
      });
    }
  }

  return {
    total_nodes: nodes.length,
    total_edges: edges.length,
    timestamp: Date.now() / 1000,
    nodes,
    edges,
  };
}

function getMockClusters(): ClusterSummary[] {
  return [
    {
      cluster_id: 'cluster_00',
      size: 135,
      num_bots: 88,
      bot_ratio: 0.6519,
      edge_density: 0.4286,
      content_similarity: 0.8124,
      coordination_score: 0.6324,
      risk_tier: 'HIGH',
      status: 'THROTTLED',
    },
    {
      cluster_id: 'cluster_01',
      size: 120,
      num_bots: 12,
      bot_ratio: 0.1000,
      edge_density: 0.1542,
      content_similarity: 0.3210,
      coordination_score: 0.1989,
      risk_tier: 'LOW',
      status: 'NORMAL',
    },
    {
      cluster_id: 'cluster_02',
      size: 142,
      num_bots: 105,
      bot_ratio: 0.7394,
      edge_density: 0.5120,
      content_similarity: 0.8842,
      coordination_score: 0.7850,
      risk_tier: 'HIGH',
      status: 'ESCALATED',
    },
    {
      cluster_id: 'cluster_03',
      size: 110,
      num_bots: 35,
      bot_ratio: 0.3182,
      edge_density: 0.2210,
      content_similarity: 0.4510,
      coordination_score: 0.3950,
      risk_tier: 'MEDIUM',
      status: 'FLAGGED',
    },
    {
      cluster_id: 'cluster_04',
      size: 128,
      num_bots: 8,
      bot_ratio: 0.0625,
      edge_density: 0.0950,
      content_similarity: 0.2100,
      coordination_score: 0.1270,
      risk_tier: 'LOW',
      status: 'NORMAL',
    },
    {
      cluster_id: 'cluster_05',
      size: 115,
      num_bots: 18,
      bot_ratio: 0.1565,
      edge_density: 0.1840,
      content_similarity: 0.3850,
      coordination_score: 0.2450,
      risk_tier: 'LOW',
      status: 'NORMAL',
    },
    {
      cluster_id: 'cluster_06',
      size: 130,
      num_bots: 22,
      bot_ratio: 0.1692,
      edge_density: 0.1920,
      content_similarity: 0.4100,
      coordination_score: 0.2620,
      risk_tier: 'LOW',
      status: 'DECAYING',
    },
    {
      cluster_id: 'cluster_07',
      size: 120,
      num_bots: 12,
      bot_ratio: 0.1000,
      edge_density: 0.1450,
      content_similarity: 0.3500,
      coordination_score: 0.2050,
      risk_tier: 'LOW',
      status: 'APPEALED',
    },
  ];
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE}/health`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Health endpoint returned error');
    return await res.json();
  } catch (e) {
    return {
      status: 'healthy (simulated)',
      environment: 'development',
      cuda_available: true,
      gpu_name: 'NVIDIA GeForce RTX 5050 Laptop GPU (sm_120)',
      vram_total_mb: 8192,
      system_ram_budget_gb: 16,
      timestamp: Date.now() / 1000,
    };
  }
}

export async function fetchClusters(): Promise<ClusterSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/clusters`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Clusters endpoint returned error');
    const data = await res.json();
    return data.clusters;
  } catch (e) {
    return getMockClusters();
  }
}

export async function fetchClusterDetail(clusterId: string): Promise<ClusterDetail> {
  try {
    const res = await fetch(`${API_BASE}/clusters/${clusterId}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Cluster detail endpoint error');
    return await res.json();
  } catch (e) {
    const summary = getMockClusters().find((c) => c.cluster_id === clusterId) || getMockClusters()[0];
    const members: ClusterMember[] = [];
    for (let i = 1; i <= Math.min(20, summary.size); i++) {
      const isBot = i <= summary.num_bots / 5;
      members.push({
        account_id: `usr_${(i * 10).toString().padStart(4, '0')}`,
        ground_truth: isBot ? 1 : 0,
        predicted_label: isBot ? 1 : 0,
        bot_probability: isBot ? 0.92 : 0.08,
      });
    }
    return {
      ...summary,
      members,
    };
  }
}

export async function fetchGraphTopology(limitEdges: number = 500): Promise<GraphTopologyResponse> {
  try {
    const res = await fetch(`${API_BASE}/graph?limit_edges=${limitEdges}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Graph endpoint error');
    return await res.json();
  } catch (e) {
    return getMockGraphData();
  }
}

export async function fetchNodeDetail(nodeId: string): Promise<NodeDetailResponse> {
  try {
    const res = await fetch(`${API_BASE}/graph/nodes/${nodeId}`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Node detail endpoint error');
    return await res.json();
  } catch (e) {
    const isBot = nodeId.includes('bot') || parseInt(nodeId.replace('usr_', '')) <= 90;
    const emb64 = Array.from({ length: 64 }, () => parseFloat((Math.random() * 2 - 1).toFixed(4)));
    return {
      id: nodeId,
      label: isBot ? 1 : 0,
      predicted_label: isBot ? 1 : 0,
      bot_probability: isBot ? 0.9450 : 0.0450,
      cluster_id: 'cluster_00',
      degree: Math.floor(Math.random() * 15) + 3,
      embedding_64d: emb64,
    };
  }
}

export async function simulateThrottle(
  entityId: string,
  throttleRatePct: number = 80.0,
  reason: string = 'Manual simulation'
): Promise<SimulationActionResponse> {
  try {
    const res = await fetch(`${API_BASE}/reactive/simulate/throttle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_id: entityId,
        throttle_rate_pct: throttleRatePct,
        reason,
      }),
    });
    if (!res.ok) throw new Error('Throttle simulation error');
    return await res.json();
  } catch (e) {
    return {
      entity_id: entityId,
      entity_type: entityId.startsWith('cluster_') ? 'cluster' : 'account',
      previous_state: 'NORMAL',
      current_state: throttleRatePct >= 90 ? 'ESCALATED' : 'THROTTLED',
      coordination_score: 0.65,
      active_throttle_pct: throttleRatePct,
      timestamp: Date.now() / 1000,
      message: `Simulated rate throttling (${throttleRatePct}%). Reason: ${reason}`,
    };
  }
}

export async function simulateDecay(
  entityId: string,
  timeDelta: number = 1.0,
  lambdaDecay: number = 0.1
): Promise<SimulationActionResponse> {
  try {
    const res = await fetch(`${API_BASE}/reactive/simulate/decay`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_id: entityId,
        time_delta: timeDelta,
        lambda_decay: lambdaDecay,
      }),
    });
    if (!res.ok) throw new Error('Decay simulation error');
    return await res.json();
  } catch (e) {
    return {
      entity_id: entityId,
      entity_type: 'cluster',
      previous_state: 'THROTTLED',
      current_state: 'DECAYING',
      coordination_score: 0.28,
      active_throttle_pct: 40.0,
      timestamp: Date.now() / 1000,
      message: `Simulated exponential decay over ${timeDelta}h (lambda=${lambdaDecay}). Score reduced to 0.28.`,
    };
  }
}

export async function simulateAppeal(
  entityId: string,
  reason: string = 'Verification submitted'
): Promise<SimulationActionResponse> {
  try {
    const res = await fetch(`${API_BASE}/reactive/simulate/appeal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        entity_id: entityId,
        reason,
      }),
    });
    if (!res.ok) throw new Error('Appeal simulation error');
    return await res.json();
  } catch (e) {
    return {
      entity_id: entityId,
      entity_type: 'cluster',
      previous_state: 'THROTTLED',
      current_state: 'APPEALED',
      coordination_score: 0.30,
      active_throttle_pct: 0.0,
      timestamp: Date.now() / 1000,
      message: `Appeal submitted successfully. Throttle lifted pending review.`,
    };
  }
}
