export type JsonRecord = Record<string, unknown>;

export type BeliefNodeType =
  | "Evidence"
  | "Claim"
  | "BeliefVariable"
  | "Factor"
  | "Decision"
  | "Other";

export type BeliefNode = {
  id: string;
  type: BeliefNodeType;
  label: string;
  posterior?: number;
  status?: string;
  sourceType?: string;
  stance?: string;
  layer?: string;
  beliefText?: string;
  trajectoryIndex?: number;
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  payload?: JsonRecord;
};

export type EdgeDir = "forward" | "backward";

export type BeliefEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  dir: EdgeDir;
  note?: string;
  weight?: number;
  direction?: number;
  payload?: JsonRecord;
};

export type BeliefMemoryGraph = {
  memory_key: string;
  title: string;
  description: string;
  mode: string;
  gdb: "memgraph";
  memgraph_uri: string;
  memgraph_lab_url: string;
  nodes: BeliefNode[];
  edges: BeliefEdge[];
  summary: {
    claims: number;
    belief_variables: number;
    evidence: number;
    factors: number;
    decisions: number;
    confidence?: number;
  };
};

export type LayoutMode = "original" | "star" | "layers";
