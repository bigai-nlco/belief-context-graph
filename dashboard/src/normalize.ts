import { applyLayout } from "./layout.ts";
import type {
  BeliefEdge,
  BeliefMemoryGraph,
  BeliefNode,
  BeliefNodeType,
  EdgeDir,
  JsonRecord,
} from "./types.ts";

/**
 * Graph normalizers extracted from main.ts (step 12): pure functions that map
 * any supported payload shape onto the dashboard's internal graph model. The
 * live HTTP source produces contract snapshots (contracts/http.schema.json);
 * legacy artifact shapes are still tolerated by design.
 */

export function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

export function stringValue(value: unknown): string {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}

export function numberValue(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

export function nodeType(value: unknown): BeliefNodeType {
  const raw = stringValue(value).replace(/[_\s-]+/g, "").toLowerCase();
  if (raw === "evidence") return "Evidence";
  if (raw === "claim") return "Claim";
  if (raw === "beliefvariable" || raw === "belief") return "BeliefVariable";
  if (raw === "factor") return "Factor";
  if (raw === "decision") return "Decision";
  return "Claim";
}

export function sourceClass(value: unknown): string {
  const raw = stringValue(value).trim().replace(/[\s-]+/g, "_").toLowerCase();
  if (
    [
      "user_input",
      "llm_reasoning",
      "tool_result",
      "historical_retrieval",
      "tool_call",
      "assistant_other",
    ].includes(raw)
  ) {
    return raw;
  }
  if (raw === "message" || raw === "dashboard") return "user_input";
  if (raw === "model_reasoning") return "llm_reasoning";
  return "assistant_other";
}

export function sourceLabel(value: unknown): string {
  return sourceClass(value).replace(/_/g, " ");
}

export function relationType(value: unknown): string {
  const raw = stringValue(value).trim().replace(/\s+/g, "_").toLowerCase();
  if (raw.includes("contradict") || raw.includes("refute")) return "contradicts";
  if (raw.includes("confirm") || raw.includes("support")) return "confirms";
  if (raw.includes("extend")) return "extends";
  return "informs";
}

export function edgeDir(value: unknown): EdgeDir {
  return stringValue(value).toLowerCase() === "backward" ? "backward" : "forward";
}

export function edgeRelationLabel(edge: BeliefEdge): string {
  return relationType(edge.type);
}

export function edgeDirection(type: string): number {
  return /contradict|refute|negative|against/i.test(type) ? -1 : 1;
}

export function normalizeNode(raw: unknown, index: number): BeliefNode {
  const node = asRecord(raw);
  const payload = asRecord(node.payload);
  const metadata = asRecord(node.metadata);
  const id = stringValue(node.id) || stringValue(node.uuid) || `node_${index}`;
  const type = nodeType(
    node.type ??
      node.node_type ?? // contracts/http.schema.json field
      payload.type ??
      metadata.type ??
      metadata.label ??
      payload.label,
  );
  return {
    id,
    type,
    label:
      stringValue(node.label) ||
      stringValue(node.name) ||
      stringValue(payload.name) ||
      id,
    posterior: numberValue(
      node.posterior ?? node.confidence ?? node.probability ?? payload.posterior,
    ),
    status: stringValue(node.status ?? payload.status ?? metadata.status),
    sourceType: sourceClass(node.source_type ?? payload.source_type ?? metadata.source_type),
    stance: stringValue(node.stance ?? payload.stance ?? metadata.stance) || "asserted",
    layer: stringValue(node.layer ?? payload.layer ?? metadata.layer) || type,
    beliefText:
      stringValue(node.belief) ||
      stringValue(payload.belief) ||
      stringValue(node.decision) ||
      stringValue(payload.claim) ||
      stringValue(node.label) ||
      stringValue(node.name) ||
      id,
    trajectoryIndex: numberValue(payload.trajectory_index ?? metadata.trajectory_index),
    x: numberValue(node.x ?? payload.x) ?? 0,
    y: numberValue(node.y ?? payload.y) ?? 0,
    payload: {
      ...payload,
      metadata,
      bcg_uuid: node.uuid,
    },
  };
}

export function normalizeEdge(raw: unknown, index: number): BeliefEdge {
  const edge = asRecord(raw);
  const payload = asRecord(edge.payload);
  const metadata = asRecord(edge.metadata);
  const type = stringValue(
    edge.type ?? edge.label ?? payload.type ?? metadata.type ?? payload.relation,
  );
  return {
    id: stringValue(edge.id) || stringValue(edge.uuid) || `edge_${index}`,
    source: stringValue(edge.source) || stringValue(edge.from_id), // contracts use from_id/to_id
    target: stringValue(edge.target) || stringValue(edge.to_id),
    type: relationType(type || "informs"),
    dir: edgeDir(payload.dir ?? payload._dir ?? metadata.dir),
    note: stringValue(edge.note ?? payload.note ?? metadata.note),
    weight: numberValue(edge.weight ?? payload.weight),
    direction: edgeDirection(type || "RELATED_TO"),
    payload: {
      ...payload,
      metadata,
      bcg_uuid: edge.uuid,
    },
  };
}

export function directedEdgeKey(edge: BeliefEdge): string {
  return `${edge.source}->${edge.target}@${edge.type}`;
}

export function uniqueDirectedEdges(edges: BeliefEdge[]): BeliefEdge[] {
  const seen = new Set<string>();
  const unique: BeliefEdge[] = [];
  for (const edge of edges) {
    const key = directedEdgeKey(edge);
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(edge);
  }
  return unique;
}

export function summarize(nodes: BeliefNode[], edges: BeliefEdge[]): BeliefMemoryGraph["summary"] {
  const count = (type: BeliefNodeType) =>
    nodes.filter((node) => node.type === type).length;
  return {
    claims: count("Claim"),
    belief_variables: count("BeliefVariable"),
    evidence: count("Evidence"),
    factors: count("Factor"),
    decisions: count("Decision"),
    confidence: nodes.length
      ? nodes.reduce((total, node) => total + (node.posterior ?? 0), 0) / nodes.length
      : undefined,
  };
}

export interface NormalizeDefaults {
  memgraphUri?: string;
  memgraphLabUrl?: string;
}

export function normalizeAnyGraph(
  payload: unknown,
  source: string,
  defaults: NormalizeDefaults = {},
): BeliefMemoryGraph {
  const record = asRecord(payload);
  const nestedMemory = asRecord(record.memory);
  const graph = asRecord(record.graph);
  const candidate = Object.keys(nestedMemory).length
    ? nestedMemory
    : Object.keys(graph).length
      ? graph
      : record;
  const nodes = Array.isArray(candidate.nodes)
    ? candidate.nodes.map((node, index) => normalizeNode(node, index))
    : [];
  const edges = Array.isArray(candidate.edges)
    ? candidate.edges.map((edge, index) => normalizeEdge(edge, index))
    : [];
  const memory: BeliefMemoryGraph = {
    memory_key: stringValue(candidate.memory_key) || stringValue(record.memory_key) || "default",
    title: stringValue(candidate.title) || "Graph Memory",
    description:
      stringValue(candidate.description) ||
      `Loaded through ${source}; normalized to BCG dashboard topology.`,
    mode: stringValue(candidate.mode) || "api",
    gdb: "memgraph",
    memgraph_uri: stringValue(candidate.memgraph_uri) || defaults.memgraphUri || "",
    memgraph_lab_url: stringValue(candidate.memgraph_lab_url) || defaults.memgraphLabUrl || "",
    nodes,
    edges,
    summary: summarize(nodes, edges),
  };
  return normalizeMemory(memory);
}

export function normalizeMemory(memory: BeliefMemoryGraph): BeliefMemoryGraph {
  const nodes = memory.nodes
    .filter((node) => node.id)
    .map((node) => {
      const payload = asRecord(node.payload);
      return {
        ...node,
        sourceType: sourceClass(node.sourceType ?? payload.source_type),
        stance: node.stance || stringValue(payload.stance) || "asserted",
        layer: node.layer || stringValue(payload.layer) || node.type,
        beliefText:
          node.beliefText ||
          stringValue(payload.belief) ||
          stringValue(payload.claim) ||
          node.label,
        trajectoryIndex:
          node.trajectoryIndex ?? numberValue(payload.trajectory_index) ?? 0,
      };
    });
  const nodeIds = new Set(nodes.map((node) => node.id));
  const edges = uniqueDirectedEdges(
    memory.edges.filter(
      (edge) => edge.source && edge.target && nodeIds.has(edge.source) && nodeIds.has(edge.target),
    ).map((edge) => ({
      ...edge,
      type: relationType(edge.type),
      dir: edge.dir || edgeDir(asRecord(edge.payload).dir),
    })),
  );
  const normalized = {
    ...memory,
    nodes,
    edges,
    summary: summarize(nodes, edges),
  };
  applyLayout(normalized, "original");
  return normalized;
}
