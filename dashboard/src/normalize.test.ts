import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  normalizeAnyGraph,
  normalizeEdge,
  normalizeMemory,
  normalizeNode,
  relationType,
} from "./normalize.ts";
import { applyLayout, layeredLayout, starLayout } from "./layout.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = JSON.parse(
  readFileSync(path.resolve(HERE, "../../contracts/fixtures/turns-response.json"), "utf8"),
) as {
  latest: Record<string, { nodes: unknown[]; relations?: unknown[] }>;
};

const CONTRACT_NODE = {
  id: 1,
  node_type: "belief",
  belief: "The user asked for a summary of key beliefs.",
  confidence: 0.88,
  stance: "asserted",
  role: "user",
  layer: "io",
  evidence_ids: [1],
  supporting_excerpts: ["Summarize the key beliefs in this conversation."],
};

const CONTRACT_RELATION = {
  id: 1,
  from_id: 1,
  to_id: 2,
  type: "depends_on",
  note: "because",
  weight: 0.5,
};

describe("normalize (step 12)", () => {
  it("maps contract snapshot nodes onto the dashboard model", () => {
    const node = normalizeNode(CONTRACT_NODE, 0);
    expect(node.id).toBe("1");
    expect(node.type).toBe("BeliefVariable");
    expect(node.beliefText).toBe("The user asked for a summary of key beliefs.");
    expect(node.posterior).toBe(0.88);
    expect(node.stance).toBe("asserted");
    expect(node.layer).toBe("io");
  });

  it("maps contract relations onto edges", () => {
    const edge = normalizeEdge(CONTRACT_RELATION, 0);
    expect(edge.source).toBe("1");
    expect(edge.target).toBe("2");
    expect(edge.type).toBe(relationType("depends_on"));
    expect(edge.note).toBe("because");
  });

  it("normalizes the live contract fixture end to end", () => {
    const snapshot = FIXTURE.latest["fixture-session:seed"];
    const memory = normalizeAnyGraph(snapshot, "fixture");
    expect(memory.nodes).toHaveLength(1);
    expect(memory.nodes[0]!.type).toBe("BeliefVariable");
    expect(memory.summary.belief_variables).toBe(1);
    expect(memory.memgraph_uri).toBe("");
  });

  it("accepts {memory: ...} and {graph: ...} wrapped payloads", () => {
    const wrappedMemory = normalizeAnyGraph({ memory: FIXTURE.latest["fixture-session:seed"] }, "w");
    const wrappedGraph = normalizeAnyGraph({ graph: FIXTURE.latest["fixture-session:seed"] }, "w");
    expect(wrappedMemory.nodes).toHaveLength(1);
    expect(wrappedGraph.nodes).toHaveLength(1);
  });

  it("drops edges whose endpoints are missing", () => {
    const memory = normalizeAnyGraph(
      {
        nodes: [CONTRACT_NODE],
        edges: [
          CONTRACT_RELATION,
          { id: 9, from_id: 99, to_id: 98, type: "depends_on" },
        ],
      },
      "t",
    );
    expect(memory.edges).toHaveLength(0);
  });

  it("deduplicates identical directed edges", () => {
    const memory = normalizeMemory({
      memory_key: "k",
      title: "t",
      description: "d",
      mode: "m",
      gdb: "memgraph",
      memgraph_uri: "",
      memgraph_lab_url: "",
      nodes: [normalizeNode(CONTRACT_NODE, 0), normalizeNode({ id: 2, node_type: "decision", decision: "go" }, 1)],
      edges: [
        { id: "a", source: "1", target: "2", type: "depends_on", dir: "forward" },
        { id: "b", source: "1", target: "2", type: "depends_on", dir: "forward" },
      ],
      summary: { claims: 0, belief_variables: 0, evidence: 0, factors: 0, decisions: 0 },
    });
    expect(memory.edges).toHaveLength(1);
  });
});

describe("layout (step 12)", () => {
  it("layered layout positions nodes by type", () => {
    const nodes = [normalizeNode(CONTRACT_NODE, 0), normalizeNode({ id: 2, node_type: "decision", decision: "go" }, 1)];
    const layout = layeredLayout(nodes);
    expect(layout.get("1")!.x).toBeLessThan(layout.get("2")!.x);
  });

  it("star layout centers a decision node", () => {
    const nodes = [normalizeNode(CONTRACT_NODE, 0), normalizeNode({ id: 2, node_type: "decision", decision: "go" }, 1)];
    const layout = starLayout(nodes);
    expect(layout.get("2")).toEqual({ x: 450, y: 280 });
  });

  it("applyLayout assigns coordinates in place", () => {
    const memory = normalizeAnyGraph(
      { nodes: [CONTRACT_NODE, { id: 2, node_type: "decision", decision: "go" }], edges: [] },
      "t",
    );
    memory.nodes.forEach((node) => {
      node.x = 0;
      node.y = 0;
    });
    applyLayout(memory, "star");
    expect(memory.nodes.some((node) => node.x !== 0 || node.y !== 0)).toBe(true);
  });
});
