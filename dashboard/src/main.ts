import "./style.css";

type BCGNode = {
  uuid?: string;
  id?: string;
  name?: string;
  probability?: number;
  payload?: JsonRecord;
  metadata?: JsonRecord;
};

type BCGEdge = {
  uuid?: string;
  id?: string;
  source?: string;
  target?: string;
  weight?: number;
  payload?: JsonRecord;
  metadata?: JsonRecord;
};

type BCGGraph = {
  nodes?: BCGNode[];
  edges?: BCGEdge[];
};

type DashboardState = {
  memory: BeliefMemoryGraph;
  selectedNodeId: string;
  selectedEdgeId: string;
  layoutMode: LayoutMode;
  showBackwardEdges: boolean;
  graphExpanded: boolean;
  graphPinned: boolean;
  status: string;
  statusKind: "" | "ok" | "error";
  source: "api" | "sample";
};

import { loadLiveGraph, sampleMemory } from "./data-sources.ts";
import { applyLayout } from "./layout.ts";
import type {
  BeliefEdge,
  BeliefMemoryGraph,
  BeliefNode,
  BeliefNodeType,
  EdgeDir,
  JsonRecord,
  LayoutMode,
} from "./types.ts";
import {
  edgeRelationLabel,
  normalizeAnyGraph,
  normalizeMemory,
  sourceClass,
  sourceLabel,
  uniqueDirectedEdges,
} from "./normalize.ts";

const app = document.querySelector<HTMLDivElement>("#app");
const apiBaseUrl = import.meta.env.VITE_BCG_API_URL || "http://127.0.0.1:8848";
const apiProblemId = import.meta.env.VITE_BCG_PROBLEM_ID || "";
const memgraphUri = import.meta.env.VITE_MEMGRAPH_URI || "bolt://localhost:7687";
const memgraphLabUrl =
  import.meta.env.VITE_MEMGRAPH_LAB_URL || "http://localhost:12345/lab";

const typeOrder: Record<BeliefNodeType, number> = {
  Evidence: 0,
  Claim: 1,
  BeliefVariable: 2,
  Factor: 3,
  Decision: 4,
  Other: 2,
};

const edgePriority: Record<string, number> = {
  confirms: 120,
  contradicts: 110,
  extends: 100,
  informs: 90,
  evaluated_by: 100,
  has_belief: 90,
  input_to: 80,
  output_to: 70,
  required_by: 60,
  supports: 50,
  owned_by: 10,
};

const state: DashboardState = {
  memory: sampleMemory({ memgraphUri, memgraphLabUrl }),
  selectedNodeId: "",
  selectedEdgeId: "",
  layoutMode: "original",
  showBackwardEdges: false,
  graphExpanded: false,
  graphPinned: false,
  status: "Dashboard initialized with local graph-memory sample.",
  statusKind: "",
  source: "sample",
};

type GraphDomCache = {
  nodes: Map<string, BeliefNode>;
  edges: Map<string, BeliefEdge>;
  nodeEls: Map<string, SVGGElement>;
  edgeEls: Array<{ edgeId: string; sourceId: string; targetId: string; el: SVGPathElement }>;
  incidentEdges: Map<string, Array<{ edgeId: string; sourceId: string; targetId: string; el: SVGPathElement }>>;
  adjacency: Map<string, Set<string>>;
  influenceCache: Map<string, Map<string, number>>;
};

let graphDomCache: GraphDomCache | null = null;
let graphDomUpdateFrame = 0;
let graphDirtyNodeIds: Set<string> | null = null;
let graphDragFrame = 0;
let graphPhysicsFrame = 0;
let graphInspectorFrame = 0;
let suppressGraphClickUntil = 0;

if (app) {
  app.innerHTML = shell();
  bindUi();
  void refreshGraph();
}

function shell(): string {
  return `
    <main class="dashboard-shell">
      <header class="topbar">
        <h1 class="brand-title">
          <img class="brand-logo" src="/favicon.svg" alt="" aria-hidden="true" />
          <span class="brand-name">BeliefTracer</span>
          <span class="brand-context">BCG Dashboard</span>
        </h1>
        <div class="topbar-actions">
          <span class="muted">Graph Memory</span>
          <span id="sourceBadge" class="badge">sample</span>
          <button id="refreshGraph" type="button" title="Refresh graph topology">Refresh</button>
        </div>
      </header>
      <section class="workbench">
        <aside class="left-pane">
          <div class="sidebar-inner">
            <section class="pane-section sidebar-block">
              <div class="section-label">Memory Runtime</div>
              <dl class="kv-list">
                <div><dt>Memory</dt><dd id="memoryKey"></dd></div>
                <div><dt>GDB</dt><dd>memgraph</dd></div>
                <div><dt>Bolt</dt><dd id="memgraphUri"></dd></div>
                <div><dt>Mode</dt><dd id="memoryMode"></dd></div>
              </dl>
              <button id="syncMemgraph" type="button" class="wide-button" title="Write current graph to Memgraph and fetch latest topology">Update Memgraph</button>
            </section>
            <section class="pane-section sidebar-block">
              <div class="section-label">Observe</div>
              <p class="muted compact-copy">Submit new evidence through the BCG memory boundary when the backend adapter is available.</p>
              <textarea id="observeText" rows="7" spellcheck="false"></textarea>
              <button id="observe" type="button" class="wide-button primary">Submit</button>
            </section>
          </div>
        </aside>
        <section class="content-pane">
          <section id="statsBar" class="stats-bar"></section>
          <section class="trajectory-panel">
            <div class="section-label">Conversation Trajectory</div>
            <div id="trajectory"></div>
          </section>
        </section>
        <aside class="right-pane graph-stage memory-sidebar" aria-label="Belief memory graph">
          <div class="memory-inner">
            <div class="memory-head">
              <div>
                <div id="graphTitle" class="memory-title"></div>
                <div id="graphSubtitle" class="muted"></div>
              </div>
              <div class="memory-actions">
                <button id="toggleBackward" type="button" class="toggle-bwd">+ backward edges</button>
              </div>
            </div>
            <div class="memory-body">
              <div class="memory-toolbar">
                <div class="segmented" role="group" aria-label="Graph layout">
                  <button type="button" data-layout="original">Original</button>
                  <button type="button" data-layout="star">Star</button>
                  <button type="button" data-layout="layers">Layers</button>
                </div>
                <span class="memory-mode">memgraph</span>
              </div>
              <div class="legend">
                <span><span class="swatch sw-informs"></span>informs</span>
                <span><span class="swatch sw-confirms"></span>confirms</span>
                <span><span class="swatch sw-contradicts"></span>contradicts</span>
                <span><span class="swatch sw-extends"></span>extends</span>
              </div>
              <div id="status" class="status-line"></div>
              <div id="graph" class="memory-graph"></div>
              <div id="inspector" class="inspector memory-inspector"></div>
            </div>
          </div>
        </aside>
      </section>
    </main>
  `;
}

function bindUi(): void {
  document
    .querySelector<HTMLButtonElement>("#refreshGraph")
    ?.addEventListener("click", () => void refreshGraph());
  document
    .querySelector<HTMLButtonElement>("#syncMemgraph")
    ?.addEventListener("click", () => void syncMemgraph());
  document
    .querySelector<HTMLButtonElement>("#observe")
    ?.addEventListener("click", () => void observeEvidence());
  document
    .querySelector<HTMLButtonElement>("#toggleBackward")
    ?.addEventListener("click", () => {
      state.showBackwardEdges = !state.showBackwardEdges;
      if (!state.showBackwardEdges && selectedEdge()?.dir === "backward") {
        state.selectedEdgeId = "";
      }
      render();
    });
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-layout]")) {
    button.addEventListener("click", () => {
      const layout = button.dataset.layout;
      if (layout === "original" || layout === "star" || layout === "layers") {
        state.layoutMode = layout;
        applyLayout(state.memory);
        render();
      }
    });
  }
}

async function refreshGraph(): Promise<void> {
  setStatus("Refreshing graph topology...");
  const loaded = await loadGraphFromApi();
  if (loaded) {
    state.memory = loaded;
    state.source = "api";
    state.selectedNodeId = state.memory.nodes[0]?.id ?? "";
    setStatus("Loaded graph topology from project API.", "ok");
  } else {
    state.memory = sampleMemory({ memgraphUri, memgraphLabUrl });
    state.source = "sample";
    state.selectedNodeId = state.memory.nodes[0]?.id ?? "";
    setStatus("Project API is not available yet; using local adapter sample.");
  }
  applyLayout(state.memory);
  render();
}

async function loadGraphFromApi(): Promise<BeliefMemoryGraph | null> {
  // Live source uses the versioned contract endpoint (contracts/http.schema.json):
  // GET /graph?problem_id=..., resolving the active session via /health.
  try {
    return await loadLiveGraph(
      { baseUrl: apiBaseUrl, problemId: apiProblemId || undefined },
      { memgraphUri, memgraphLabUrl },
    );
  } catch (error) {
    console.warn("BCG server unavailable, falling back to sample:", error);
    return null;
  }
}

async function syncMemgraph(): Promise<void> {
  setStatus("Syncing graph through Memgraph...");
  renderStatus();
  try {
    const payload = await fetchJson<{ memory?: unknown }>("/api/memory/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        memory_key: state.memory.memory_key,
        write_current: true,
        gdb: {
          type: "memgraph",
          uri: memgraphUri,
        },
        memory: state.memory,
        bcg_graph: memoryToBcgGraph(state.memory),
      }),
    });
    const updated = normalizeAnyGraph(payload.memory ?? payload, "/api/memory/update");
    if (updated.nodes.length > 0) {
      state.memory = updated;
      state.source = "api";
      applyLayout(state.memory);
    }
    setStatus("Memgraph topology synchronized.", "ok");
  } catch (error) {
    setStatus(`Memgraph sync boundary is not implemented yet: ${messageOf(error)}`, "error");
  }
  render();
}

async function observeEvidence(): Promise<void> {
  const input = document.querySelector<HTMLTextAreaElement>("#observeText");
  const content = input?.value.trim() ?? "";
  if (!content) {
    setStatus("Observe payload is empty.", "error");
    renderStatus();
    return;
  }
  try {
    const payload = await fetchJson<unknown>("/api/memory/observe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_type: "dashboard",
        content,
        metadata: { dashboard: "bcg" },
      }),
    });
    const memory = normalizeAnyGraph(payload, "/api/memory/observe");
    if (memory.nodes.length > 0) {
      state.memory = memory;
      state.source = "api";
      applyLayout(state.memory);
    }
    if (input) input.value = "";
    setStatus("Evidence submitted to BCG memory.", "ok");
  } catch (error) {
    setStatus(`BCGMemory.observe is not implemented yet: ${messageOf(error)}`, "error");
  }
  render();
}

function render(): void {
  renderHeader();
  renderMetrics();
  renderTrajectory();
  renderGraph();
  renderInspector();
  renderStatus();
  syncLayoutButtons();
}

function renderHeader(): void {
  setText("#sourceBadge", state.source);
  setText("#memoryKey", state.memory.memory_key);
  setText("#memgraphUri", state.memory.memgraph_uri);
  setText("#memoryMode", state.memory.mode);
  setText("#graphTitle", state.memory.title);
  setText("#graphSubtitle", state.memory.description);
  const toggle = document.querySelector<HTMLButtonElement>("#toggleBackward");
  if (toggle) {
    toggle.classList.toggle("active", state.showBackwardEdges);
    toggle.textContent = state.showBackwardEdges ? "showing backward" : "+ backward edges";
  }
  document.querySelector<HTMLDivElement>("#graph")?.classList.toggle("graph-expanded", state.graphExpanded);
}

function renderMetrics(): void {
  const target = document.querySelector<HTMLDivElement>("#statsBar");
  if (!target) return;
  const sourceCounts = countBy(state.memory.nodes, (node) => node.sourceType || "assistant_other");
  const forward = state.memory.edges.filter((edge) => edge.dir === "forward").length;
  const backward = state.memory.edges.length - forward;
  target.innerHTML = `
    <span><span class="label">Beliefs</span><b>${state.memory.nodes.length}</b></span>
    <span><span class="label"><span class="stat-dot user_input"></span>User input</span><b>${sourceCounts.user_input ?? 0}</b></span>
    <span><span class="label"><span class="stat-dot llm_reasoning"></span>LLM reasoning</span><b>${sourceCounts.llm_reasoning ?? 0}</b></span>
    <span><span class="label"><span class="stat-dot tool_result"></span>Tool result</span><b>${sourceCounts.tool_result ?? 0}</b></span>
    <span><span class="label">Forward</span><b>${forward}</b><em>informs</em></span>
    <span><span class="label">Backward</span><b>${backward}</b><em>evaluation</em></span>
  `;
}

function renderTrajectory(): void {
  const target = document.querySelector<HTMLDivElement>("#trajectory");
  if (!target) return;
  const messages = sampleTrajectory();
  target.innerHTML = messages
    .map((message) => {
      const count = message.beliefIds.length;
      return `
        <article class="msg msg-${esc(message.role)}">
          <header class="msg-head">
            <span class="msg-role role-${esc(message.role)}">${esc(message.role)}</span>
            <span class="msg-idx">trajectory_index <b>${message.index}</b></span>
            <span class="msg-belief-count ${count ? "" : "empty"}">${count ? `${count} beliefs` : "no beliefs"}</span>
          </header>
          <pre class="msg-body">${highlightMessage(message.text, message.beliefIds)}</pre>
        </article>`;
    })
    .join("");
  for (const item of target.querySelectorAll<HTMLElement>(".ev")) {
    item.addEventListener("click", () => {
      const id = item.dataset.beliefId || "";
      if (!id) return;
      state.selectedNodeId = id;
      state.selectedEdgeId = "";
      render();
    });
  }
}

function renderGraph(): void {
  const target = document.querySelector<HTMLDivElement>("#graph");
  if (!target) return;
  resetGraphDomCache();
  const nodesById = new Map(state.memory.nodes.map((node) => [node.id, node]));
  const edges = uniqueDirectedEdges(state.memory.edges)
    .filter((edge) => nodesById.has(edge.source) && nodesById.has(edge.target))
    .filter((edge) => state.showBackwardEdges || edge.dir !== "backward");
  const ticks = [1, 2, 3, 4, 5, 6]
    .map((index) => `<text class="col-label" x="${90 + index * 120}" y="545">traj[${index}]</text>`)
    .join("");
  const edgeHtml = edges
    .map((edge) => {
      const source = nodesById.get(edge.source);
      const targetNode = nodesById.get(edge.target);
      if (!source || !targetNode) return "";
      const relation = edgeRelationLabel(edge);
      const active = edge.id === state.selectedEdgeId ? "active" : "";
      return `
        <path
          class="edge edge-${esc(edge.dir)} type-${esc(relation)} ${active}"
          d="${edgeCurve(source, targetNode, edge.dir)}"
          marker-end="url(#arr-${esc(relation)})"
          data-edge-id="${esc(edge.id)}"
          data-source="${esc(edge.source)}"
          data-target="${esc(edge.target)}"
        ></path>`;
    })
    .join("");
  const nodeHtml = state.memory.nodes
    .map((node) => {
      const active = node.id === state.selectedNodeId ? "selected" : "";
      const label = truncate(node.label || node.id, 26);
      const posterior =
        node.posterior === undefined ? "--" : node.posterior.toFixed(2);
      const source = sourceClass(node.sourceType);
      return `
        <g class="node source-${esc(source)} ${active}" data-node-id="${esc(node.id)}" transform="translate(${node.x}, ${node.y})">
          <rect class="node-pill" x="-48" y="-18" width="96" height="36"></rect>
          <text class="node-id" y="-4">#${esc(node.id)}</text>
          <text class="node-conf" y="11">${esc(posterior)}</text>
          <title>${esc(label)}</title>
        </g>`;
    })
    .join("");

  target.innerHTML = `
    <div class="memory-graph-tools">
      <button type="button" id="memoryGraphPin" class="memory-graph-tool memory-graph-pin" aria-label="Pin graph drag" aria-pressed="${state.graphPinned ? "true" : "false"}" title="${state.graphPinned ? "Unpin graph drag" : "Pin graph drag"}">
        <span class="memory-graph-pin-icon" aria-hidden="true"></span>
      </button>
      <button type="button" id="memoryGraphScatter" class="memory-graph-tool memory-graph-scatter" aria-label="Cycle graph layout" aria-pressed="${state.layoutMode !== "original" ? "true" : "false"}" title="Cycle graph layout">
        <span class="memory-graph-scatter-icon" aria-hidden="true"></span>
      </button>
      <button type="button" id="memoryGraphExpand" class="memory-graph-tool memory-graph-expand" aria-label="Expand graph" aria-pressed="${state.graphExpanded ? "true" : "false"}" title="${state.graphExpanded ? "Dock graph" : "Expand graph"}">
        <span class="memory-graph-expand-icon" aria-hidden="true"></span>
      </button>
    </div>
    <svg class="belief-svg belief-graph ${state.showBackwardEdges ? "show-backward" : ""}" viewBox="0 0 900 560" role="img" aria-label="Belief memory topology">
      <defs>
        <marker id="arr-informs" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z"></path>
        </marker>
        <marker id="arr-confirms" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z"></path>
        </marker>
        <marker id="arr-contradicts" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z"></path>
        </marker>
        <marker id="arr-extends" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z"></path>
        </marker>
      </defs>
      ${ticks}
      ${edgeHtml}
      ${nodeHtml}
    </svg>`;

  target.classList.toggle("graph-expanded", state.graphExpanded);
  bindGraphTools();
  buildGraphDomCache();
  bindGraphSelection(target);
  bindEdgeSelection(target);
  bindGraphDrag(target);
}

function bindGraphTools(): void {
  document.querySelector<HTMLButtonElement>("#memoryGraphPin")?.addEventListener("click", () => {
    state.graphPinned = !state.graphPinned;
    syncGraphToolState();
  });
  document.querySelector<HTMLButtonElement>("#memoryGraphExpand")?.addEventListener("click", () => {
    state.graphExpanded = !state.graphExpanded;
    syncGraphToolState();
  });
  document.querySelector<HTMLButtonElement>("#memoryGraphScatter")?.addEventListener("click", () => {
    state.layoutMode =
      state.layoutMode === "original" ? "star" : state.layoutMode === "star" ? "layers" : "original";
    applyLayout(state.memory);
    render();
  });
  syncGraphToolState();
}

function syncGraphToolState(): void {
  const graph = document.querySelector<HTMLDivElement>("#graph");
  const pin = document.querySelector<HTMLButtonElement>("#memoryGraphPin");
  const expand = document.querySelector<HTMLButtonElement>("#memoryGraphExpand");
  const scatter = document.querySelector<HTMLButtonElement>("#memoryGraphScatter");
  graph?.classList.toggle("graph-expanded", state.graphExpanded);
  if (pin) {
    pin.setAttribute("aria-pressed", state.graphPinned ? "true" : "false");
    pin.title = state.graphPinned ? "Unpin graph drag" : "Pin graph drag";
  }
  if (expand) {
    expand.setAttribute("aria-pressed", state.graphExpanded ? "true" : "false");
    expand.title = state.graphExpanded ? "Dock graph" : "Expand graph";
  }
  if (scatter) {
    scatter.setAttribute("aria-pressed", state.layoutMode !== "original" ? "true" : "false");
    scatter.title =
      state.layoutMode === "original"
        ? "Scatter to star topology"
        : state.layoutMode === "star"
          ? "Transform to layered topology"
          : "Restore original topology";
  }
}

function bindGraphSelection(target: HTMLElement): void {
  for (const node of target.querySelectorAll<SVGGElement>(".node")) {
    node.addEventListener("click", (event) => {
      if (performance.now() < suppressGraphClickUntil) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      state.selectedNodeId = node.dataset.nodeId ?? "";
      state.selectedEdgeId = "";
      render();
    });
  }
}

function bindEdgeSelection(target: HTMLElement): void {
  for (const edge of target.querySelectorAll<SVGPathElement>(".edge")) {
    edge.addEventListener("click", () => {
      state.selectedEdgeId = edge.dataset.edgeId ?? "";
      state.selectedNodeId = "";
      render();
    });
  }
}

function resetGraphDomCache(): void {
  graphDomCache = null;
  graphDirtyNodeIds = null;
  if (graphDomUpdateFrame) {
    cancelAnimationFrame(graphDomUpdateFrame);
    graphDomUpdateFrame = 0;
  }
  if (graphInspectorFrame) {
    cancelAnimationFrame(graphInspectorFrame);
    graphInspectorFrame = 0;
  }
}

function buildGraphDomCache(): void {
  const graph = document.querySelector<HTMLDivElement>("#graph");
  if (!graph) {
    graphDomCache = null;
    return;
  }
  const nodes = new Map(state.memory.nodes.map((node) => [node.id, node]));
  const visibleEdges = uniqueDirectedEdges(state.memory.edges)
    .filter((edge) => state.showBackwardEdges || edge.dir !== "backward")
    .filter((edge) => nodes.has(edge.source) && nodes.has(edge.target));
  const edges = new Map(visibleEdges.map((edge) => [edge.id, edge]));
  const nodeEls = new Map(
    Array.from(graph.querySelectorAll<SVGGElement>(".node")).map((el) => [
      el.dataset.nodeId ?? "",
      el,
    ]),
  );
  const edgeEls = Array.from(graph.querySelectorAll<SVGPathElement>(".edge")).map((el) => ({
    edgeId: el.dataset.edgeId ?? "",
    sourceId: el.dataset.source ?? "",
    targetId: el.dataset.target ?? "",
    el,
  }));
  const incidentEdges = new Map<string, GraphDomCache["edgeEls"]>(
    Array.from(nodes.keys()).map((id) => [id, []]),
  );
  const adjacency = new Map<string, Set<string>>(
    Array.from(nodes.keys()).map((id) => [id, new Set<string>()]),
  );
  for (const record of edgeEls) {
    incidentEdges.get(record.sourceId)?.push(record);
    incidentEdges.get(record.targetId)?.push(record);
    adjacency.get(record.sourceId)?.add(record.targetId);
    adjacency.get(record.targetId)?.add(record.sourceId);
  }
  graphDomCache = {
    nodes,
    edges,
    nodeEls,
    edgeEls,
    incidentEdges,
    adjacency,
    influenceCache: new Map(),
  };
}

function updateGraphDom(dirtyNodeIds: Iterable<string> | null = null): void {
  if (!graphDomCache) buildGraphDomCache();
  const cache = graphDomCache;
  if (!cache) return;
  const dirty = dirtyNodeIds ? new Set(dirtyNodeIds) : null;
  const nodeEntries = dirty
    ? Array.from(dirty, (id) => [id, cache.nodeEls.get(id)] as const).filter(([, el]) => el)
    : Array.from(cache.nodeEls.entries());
  for (const [id, nodeEl] of nodeEntries) {
    const node = cache.nodes.get(id);
    if (!node || !nodeEl) continue;
    nodeEl.setAttribute("transform", `translate(${node.x}, ${node.y})`);
  }
  const edgeRecords = dirty
    ? Array.from(new Set(Array.from(dirty).flatMap((id) => cache.incidentEdges.get(id) ?? [])))
    : cache.edgeEls;
  for (const record of edgeRecords) {
    const source = cache.nodes.get(record.sourceId);
    const targetNode = cache.nodes.get(record.targetId);
    const edge = cache.edges.get(record.edgeId);
    if (!source || !targetNode || !edge) continue;
    record.el.setAttribute("d", edgeCurve(source, targetNode, edge.dir));
  }
}

function scheduleGraphDomUpdate(dirtyNodeIds: Iterable<string> | null = null): void {
  if (dirtyNodeIds === null) {
    graphDirtyNodeIds = null;
  } else if (graphDirtyNodeIds !== null) {
    for (const id of dirtyNodeIds) graphDirtyNodeIds.add(id);
  } else {
    graphDirtyNodeIds = new Set(dirtyNodeIds);
  }
  if (graphDomUpdateFrame) return;
  graphDomUpdateFrame = requestAnimationFrame(() => {
    graphDomUpdateFrame = 0;
    const dirty = graphDirtyNodeIds;
    graphDirtyNodeIds = null;
    updateGraphDom(dirty);
  });
}

function scheduleInspectorRender(): void {
  if (graphInspectorFrame) return;
  graphInspectorFrame = requestAnimationFrame(() => {
    graphInspectorFrame = 0;
    renderInspector();
  });
}

function graphDragInfluence(startId: string): Map<string, number> {
  if (state.graphPinned) return new Map([[startId, 1]]);
  if (!graphDomCache) buildGraphDomCache();
  const cache = graphDomCache;
  if (!cache) return new Map([[startId, 1]]);
  const cached = cache.influenceCache.get(startId);
  if (cached) return cached;
  const influence = new Map<string, number>([[startId, 1]]);
  const queue: Array<{ id: string; hop: number }> = [{ id: startId, hop: 0 }];
  const seen = new Set([startId]);
  while (queue.length) {
    const current = queue.shift();
    if (!current) continue;
    const nextHop = current.hop + 1;
    for (const next of cache.adjacency.get(current.id) ?? []) {
      if (seen.has(next)) continue;
      seen.add(next);
      influence.set(next, Math.max(0.1, Math.pow(0.56, nextHop)));
      queue.push({ id: next, hop: nextHop });
    }
  }
  cache.influenceCache.set(startId, influence);
  return influence;
}

function clampGraphPoint(svg: SVGSVGElement, x: number, y: number): { x: number; y: number } {
  const viewBox = svg.viewBox.baseVal;
  return {
    x: clamp(x, 38, (viewBox.width || 900) - 38),
    y: clamp(y, 34, (viewBox.height || 560) - 34),
  };
}

function graphClientPoint(matrix: DOMMatrix, clientX: number, clientY: number): { x: number; y: number } {
  return {
    x: matrix.a * clientX + matrix.c * clientY + matrix.e,
    y: matrix.b * clientX + matrix.d * clientY + matrix.f,
  };
}

function startGraphPhysics(anchorId = ""): void {
  if (graphPhysicsFrame) return;
  const started = performance.now();
  const tick = (now: number) => {
    const cache = graphDomCache;
    if (!cache) {
      graphPhysicsFrame = 0;
      return;
    }
    const activeDragId = anchorId && cache.nodes.has(anchorId) ? anchorId : "";
    const dirty = new Set<string>();
    const edges = Array.from(cache.edges.values());
    for (const node of cache.nodes.values()) {
      node.vx = (node.vx ?? 0) * 0.86;
      node.vy = (node.vy ?? 0) * 0.86;
    }
    for (const edge of edges) {
      const source = cache.nodes.get(edge.source);
      const targetNode = cache.nodes.get(edge.target);
      if (!source || !targetNode) continue;
      const dx = targetNode.x - source.x;
      const dy = targetNode.y - source.y;
      const dist = Math.hypot(dx, dy) || 1;
      const desired = edge.dir === "backward" ? 150 : 132;
      const force = (dist - desired) * 0.0078;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      if (source.id !== activeDragId) {
        source.vx = (source.vx ?? 0) + fx;
        source.vy = (source.vy ?? 0) + fy;
      }
      if (targetNode.id !== activeDragId) {
        targetNode.vx = (targetNode.vx ?? 0) - fx;
        targetNode.vy = (targetNode.vy ?? 0) - fy;
      }
    }
    const nodeList = Array.from(cache.nodes.values());
    if (nodeList.length <= 90) {
      for (let i = 0; i < nodeList.length; i += 1) {
        for (let j = i + 1; j < nodeList.length; j += 1) {
          const left = nodeList[i];
          const right = nodeList[j];
          const dx = right.x - left.x;
          const dy = right.y - left.y;
          const dist = Math.hypot(dx, dy) || 1;
          if (dist > 115) continue;
          const force = (115 - dist) * 0.0022;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          if (left.id !== activeDragId) {
            left.vx = (left.vx ?? 0) - fx;
            left.vy = (left.vy ?? 0) - fy;
          }
          if (right.id !== activeDragId) {
            right.vx = (right.vx ?? 0) + fx;
            right.vy = (right.vy ?? 0) + fy;
          }
        }
      }
    }
    let energy = 0;
    for (const node of cache.nodes.values()) {
      if (node.id === activeDragId) continue;
      const vx = clamp(node.vx ?? 0, -11, 11);
      const vy = clamp(node.vy ?? 0, -11, 11);
      node.vx = vx;
      node.vy = vy;
      node.x = clamp(node.x + vx, 38, 862);
      node.y = clamp(node.y + vy, 34, 526);
      energy += Math.abs(vx) + Math.abs(vy);
      dirty.add(node.id);
    }
    scheduleGraphDomUpdate(dirty);
    if (energy > 0.06 && now - started < 1800) {
      graphPhysicsFrame = requestAnimationFrame(tick);
    } else {
      graphPhysicsFrame = 0;
      scheduleInspectorRender();
    }
  };
  graphPhysicsFrame = requestAnimationFrame(tick);
}

function bindGraphDrag(target: HTMLElement): void {
  const svg = target.querySelector<SVGSVGElement>("svg");
  if (!svg) return;
  let drag:
    | {
        node: BeliefNode;
        pointerId: number;
        dx: number;
        dy: number;
        nodeEl: SVGGElement;
        influence: Map<string, number>;
        starts: Map<string, { x: number; y: number }>;
        pendingClientX: number;
        pendingClientY: number;
        screenMatrix: DOMMatrix;
        startClientX: number;
        startClientY: number;
        moved: boolean;
      }
    | null = null;
  const processGraphDragFrame = () => {
    graphDragFrame = 0;
    if (!drag || !graphDomCache) return;
    const primaryStart = drag.starts.get(drag.node.id) ?? { x: drag.node.x, y: drag.node.y };
    const pointer = graphClientPoint(drag.screenMatrix, drag.pendingClientX, drag.pendingClientY);
    const targetPoint = clampGraphPoint(
      svg,
      pointer.x + drag.dx,
      pointer.y + drag.dy,
    );
    const deltaX = targetPoint.x - primaryStart.x;
    const deltaY = targetPoint.y - primaryStart.y;
    for (const [id, strength] of drag.influence.entries()) {
      const node = graphDomCache.nodes.get(id);
      const start = drag.starts.get(id);
      if (!node || !start) continue;
      const next = clampGraphPoint(svg, start.x + deltaX * strength, start.y + deltaY * strength);
      const oldX = node.x;
      const oldY = node.y;
      const stiffness = id === drag.node.id ? 1 : 0.42 + strength * 0.3;
      node.x += (next.x - node.x) * stiffness;
      node.y += (next.y - node.y) * stiffness;
      node.vx = (node.vx ?? 0) * 0.25 + (node.x - oldX) * 0.75;
      node.vy = (node.vy ?? 0) * 0.25 + (node.y - oldY) * 0.75;
    }
    scheduleGraphDomUpdate(drag.influence.keys());
    startGraphPhysics(drag.node.id);
  };
  svg.addEventListener("pointerdown", (event) => {
    const nodeEl = (event.target as Element | null)?.closest<SVGGElement>(
      ".node",
    );
    if (!nodeEl) return;
    if (!graphDomCache) buildGraphDomCache();
    const node = graphDomCache?.nodes.get(nodeEl.dataset.nodeId ?? "");
    if (!node) return;
    if (graphPhysicsFrame) {
      cancelAnimationFrame(graphPhysicsFrame);
      graphPhysicsFrame = 0;
    }
    const screenMatrix = svg.getScreenCTM()?.inverse();
    if (!screenMatrix) return;
    const point = graphClientPoint(screenMatrix, event.clientX, event.clientY);
    const influence = graphDragInfluence(node.id);
    const starts = new Map<string, { x: number; y: number }>();
    for (const id of influence.keys()) {
      const influenced = graphDomCache?.nodes.get(id);
      if (influenced) starts.set(id, { x: influenced.x, y: influenced.y });
    }
    drag = {
      node,
      pointerId: event.pointerId,
      dx: node.x - point.x,
      dy: node.y - point.y,
      nodeEl,
      influence,
      starts,
      pendingClientX: event.clientX,
      pendingClientY: event.clientY,
      screenMatrix,
      startClientX: event.clientX,
      startClientY: event.clientY,
      moved: false,
    };
    state.selectedNodeId = node.id;
    state.selectedEdgeId = "";
    scheduleInspectorRender();
    nodeEl.classList.add("dragging");
    target.classList.add("graph-dragging");
    svg.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const samples = event.getCoalescedEvents?.() ?? [];
    const sample = samples[samples.length - 1] ?? event;
    if (Math.hypot(sample.clientX - drag.startClientX, sample.clientY - drag.startClientY) > 3) {
      drag.moved = true;
    }
    drag.pendingClientX = sample.clientX;
    drag.pendingClientY = sample.clientY;
    if (!graphDragFrame) graphDragFrame = requestAnimationFrame(processGraphDragFrame);
  });
  const endDrag = (event: PointerEvent) => {
    if (!drag) return;
    drag.nodeEl.classList.remove("dragging");
    target.classList.remove("graph-dragging");
    try {
      svg.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture can already be released by the browser.
    }
    if (graphDragFrame) {
      cancelAnimationFrame(graphDragFrame);
      graphDragFrame = 0;
    }
    if (graphPhysicsFrame) {
      cancelAnimationFrame(graphPhysicsFrame);
      graphPhysicsFrame = 0;
    }
    if (drag.moved) suppressGraphClickUntil = performance.now() + 260;
    startGraphPhysics();
    drag = null;
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
}

function renderInspector(): void {
  const target = document.querySelector<HTMLDivElement>("#inspector");
  if (!target) return;
  const edge = selectedEdge();
  if (edge) {
    const source = state.memory.nodes.find((item) => item.id === edge.source);
    const targetNode = state.memory.nodes.find((item) => item.id === edge.target);
    target.innerHTML = `
      <div class="detail-card">
        <h2>Relation</h2>
        <div class="edge-detail-rel">
          <span>${esc(edge.source)}</span>
          <span class="arrow">-&gt;</span>
          <span class="rel-pill type-${esc(edgeRelationLabel(edge))}">${esc(edgeRelationLabel(edge))}</span>
          <span class="arrow">-&gt;</span>
          <span>${esc(edge.target)}</span>
        </div>
        <h3>Note</h3>
        <p>${esc(edge.note || "No relation note available.")}</p>
        <div class="edge-detail-pair">
          <div class="side">
            <div class="side-label">from</div>
            <div class="side-text">${esc(source?.beliefText || source?.label || edge.source)}</div>
          </div>
          <div class="side">
            <div class="side-label">to</div>
            <div class="side-text">${esc(targetNode?.beliefText || targetNode?.label || edge.target)}</div>
          </div>
        </div>
      </div>
    `;
    return;
  }
  const node =
    state.memory.nodes.find((item) => item.id === state.selectedNodeId) ??
    state.memory.nodes[0];
  if (!node) {
    target.innerHTML = `<div class="empty">No node selected.</div>`;
    return;
  }
  state.selectedNodeId = node.id;
  const incident = state.memory.edges.filter(
    (edge) => edge.source === node.id || edge.target === node.id,
  );
  target.innerHTML = `
    <div class="detail-card">
    <h2>Belief ${esc(node.id)} . ${node.posterior === undefined ? "--" : node.posterior.toFixed(2)}</h2>
    <div class="detail-meta">
      <span class="badge source-${esc(sourceClass(node.sourceType))}">${esc(sourceLabel(node.sourceType))}</span>
      <span class="badge stance-${esc(node.stance || "asserted")}">${esc(node.stance || "asserted")}</span>
      <span class="badge layer">${esc(node.layer || node.type)}</span>
    </div>
    <p class="detail-belief-text">${esc(node.beliefText || node.label)}</p>
    <h3>Source</h3>
    <dl class="kv-list">
      <div><dt>ID</dt><dd>${esc(node.id)}</dd></div>
      <div><dt>Confidence</dt><dd>${node.posterior === undefined ? "-" : esc(pct(node.posterior))}</dd></div>
      <div><dt>Status</dt><dd>${esc(node.status || "-")}</dd></div>
      <div><dt>Degree</dt><dd>${incident.length}</dd></div>
    </dl>
    <pre>${esc(JSON.stringify(node.payload ?? {}, null, 2))}</pre>
    </div>
  `;
}

function renderStatus(): void {
  const target = document.querySelector<HTMLDivElement>("#status");
  if (!target) return;
  target.className = `status-line ${state.statusKind}`;
  target.textContent = state.status;
}

function syncLayoutButtons(): void {
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-layout]")) {
    button.classList.toggle("active", button.dataset.layout === state.layoutMode);
  }
}









function memoryToBcgGraph(memory: BeliefMemoryGraph): BCGGraph {
  return {
    nodes: memory.nodes.map((node) => ({
      uuid: node.id,
      name: node.label,
      probability: node.posterior ?? 1,
      payload: {
        ...(node.payload ?? {}),
        type: node.type,
        x: node.x,
        y: node.y,
      },
      metadata: {
        dashboard_type: node.type,
        gdb: memory.gdb,
      },
    })),
    edges: memory.edges.map((edge) => ({
      uuid: edge.id,
      source: edge.source,
      target: edge.target,
      weight: edge.weight ?? 1,
      payload: {
        ...(edge.payload ?? {}),
        type: edge.type,
        direction: edge.direction ?? 1,
      },
      metadata: {
        dashboard_type: edge.type,
        gdb: memory.gdb,
      },
    })),
  };
}







function selectedEdge(): BeliefEdge | undefined {
  return state.memory.edges.find((edge) => edge.id === state.selectedEdgeId);
}




function edgeCurve(source: BeliefNode, targetNode: BeliefNode, dir: EdgeDir): string {
  const midX = (source.x + targetNode.x) / 2;
  const midY = (source.y + targetNode.y) / 2;
  const sign = dir === "backward" ? -1 : 1;
  const bend = Math.max(26, Math.min(96, Math.abs(targetNode.x - source.x) * 0.28));
  return `M ${source.x},${source.y} Q ${midX},${midY - bend * sign} ${targetNode.x},${targetNode.y}`;
}

function edgeLabelPosition(
  source: BeliefNode,
  targetNode: BeliefNode,
): { x: number; y: number } {
  const dx = targetNode.x - source.x;
  const dy = targetNode.y - source.y;
  const length = Math.hypot(dx, dy) || 1;
  return {
    x: (source.x + targetNode.x) / 2 + (-dy / length) * 10,
    y: (source.y + targetNode.y) / 2 + (dx / length) * 10,
  };
}

function edgeRenderPoints(
  source: BeliefNode,
  targetNode: BeliefNode,
): { x1: number; y1: number; x2: number; y2: number } {
  const dx = targetNode.x - source.x;
  const dy = targetNode.y - source.y;
  const length = Math.hypot(dx, dy) || 1;
  const startOffset = 31;
  const endOffset = 32;
  if (length <= startOffset + endOffset + 4) {
    return { x1: source.x, y1: source.y, x2: targetNode.x, y2: targetNode.y };
  }
  return {
    x1: source.x + (dx / length) * startOffset,
    y1: source.y + (dy / length) * startOffset,
    x2: targetNode.x - (dx / length) * endOffset,
    y2: targetNode.y - (dy / length) * endOffset,
  };
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function setStatus(message: string, kind: "" | "ok" | "error" = ""): void {
  state.status = message;
  state.statusKind = kind;
}

function setText(selector: string, text: unknown): void {
  const target = document.querySelector<HTMLElement>(selector);
  if (target) target.textContent = String(text ?? "");
}

function sampleTrajectory(): Array<{
  index: number;
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  beliefIds: string[];
}> {
  return [
    {
      index: 0,
      role: "system",
      beliefIds: [],
      text: "You are a research assistant that answers questions by searching for relevant information.",
    },
    {
      index: 1,
      role: "user",
      beliefIds: ["01", "02", "03"],
      text:
        "What is the title of the 1980 ethnographic film that is part of a series on the !Kung people, directed by John Marshall in collaboration with Adrienne, and whose footage was reused in The Gods Must Be Crazy?",
    },
    {
      index: 2,
      role: "assistant",
      beliefIds: ["07", "09", "19"],
      text:
        "I recall Marshall collaborations involving the !Kung people, but the title needs verification. Search query: John Marshall Adrienne LaFrance !Kung 1980 ethnographic film title.",
    },
    {
      index: 3,
      role: "tool",
      beliefIds: ["21", "22", "26"],
      text:
        "N!ai, the Story of a !Kung Woman is a documentary by John Marshall. It was first broadcast in 1980 as part of PBS Odyssey and contains a scene from the filming of The Gods Must Be Crazy.",
    },
    {
      index: 4,
      role: "assistant",
      beliefIds: ["37"],
      text:
        "The retrieved evidence points to N!ai rather than the earlier guessed title The !Kung.",
    },
    {
      index: 6,
      role: "assistant",
      beliefIds: ["47"],
      text:
        "The answer is N!ai, the Story of a !Kung Woman.",
    },
  ];
}

function highlightMessage(text: string, beliefIds: string[]): string {
  if (!beliefIds.length) return esc(text);
  const chunks = text.split(/(?<=[.!?])\s+/);
  return chunks
    .map((chunk, index) => {
      const beliefId = beliefIds[Math.min(index, beliefIds.length - 1)];
      return `<span class="ev ${beliefId === state.selectedNodeId ? "active" : ""}" data-belief-id="${esc(beliefId)}">${esc(chunk)}</span>`;
    })
    .join("\n\n");
}

function countBy<T>(items: T[], keyFn: (item: T) => string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const item of items) {
    const key = keyFn(item);
    out[key] = (out[key] ?? 0) + 1;
  }
  return out;
}


function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function average(values: number[]): number | undefined {
  if (!values.length) return undefined;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function esc(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function truncate(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}...`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
