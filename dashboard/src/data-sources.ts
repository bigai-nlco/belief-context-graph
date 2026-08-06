import { normalizeAnyGraph, type NormalizeDefaults } from "./normalize.ts";
import type { BeliefMemoryGraph } from "./types.ts";

/**
 * Data-source adapters (step 12): three explicit sources feeding the
 * dashboard — the live BCG construct server (contracts/http.schema.json),
 * persisted artifact replay (memory.json / JSONL), and the bundled sample.
 * Every input passes through the same normalizer; errors are never masked
 * as sample success.
 */

export interface LiveSourceOptions {
  baseUrl: string;
  problemId?: string;
  fetchImpl?: typeof globalThis.fetch;
}

export async function resolveActiveProblemId(
  baseUrl: string,
  fetchImpl: typeof globalThis.fetch = globalThis.fetch,
): Promise<string | undefined> {
  const response = await fetchImpl(`${baseUrl}/health`);
  if (!response.ok) {
    throw new Error(`BCG server health check failed (HTTP ${response.status})`);
  }
  const body = (await response.json()) as { active?: string[]; all?: string[] };
  const active = Array.isArray(body?.active) ? body.active : [];
  return active[0];
}

export async function loadLiveGraph(
  options: LiveSourceOptions,
  defaults: NormalizeDefaults = {},
): Promise<BeliefMemoryGraph> {
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const problemId = options.problemId ?? (await resolveActiveProblemId(baseUrl, fetchImpl));
  if (!problemId) {
    throw new Error("No active graph session on the BCG server; start a construct session first.");
  }
  const response = await fetchImpl(`${baseUrl}/graph?problem_id=${encodeURIComponent(problemId)}`);
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`BCG server /graph failed (HTTP ${response.status}): ${detail}`);
  }
  const payload: unknown = await response.json();
  return normalizeAnyGraph(payload, "bcg-server", defaults);
}

export interface ArtifactReplayOptions {
  defaults?: NormalizeDefaults;
}

/**
 * Normalize a persisted artifact payload (memory document with
 * schema: bcg.memory.v2, or a raw graph object) for replay.
 */
export function loadArtifactReplay(payload: unknown, options: ArtifactReplayOptions = {}): BeliefMemoryGraph {
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch (error) {
      throw new Error(`Artifact is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return normalizeAnyGraph(payload, "artifact", options.defaults);
}

export function sampleMemory(defaults: NormalizeDefaults = {}): BeliefMemoryGraph {
  const memory = normalizeAnyGraph(
    {
    memory_key: "construct_beliefs:example",
    title: "Belief Graph",
    description: "construct_beliefs trajectory sample",
    mode: "adapter",
    gdb: "memgraph",
    nodes: [
      {
        id: "01",
        type: "Claim",
        label: "!Kung clue",
        posterior: 0.82,
        status: "observed",
        x: 125,
        y: 92,
        payload: {
          source_type: "user_input",
          stance: "asserted",
          layer: "question",
          trajectory_index: 1,
          belief: "The question asks for a 1980 ethnographic film about the !Kung people.",
        },
      },
      {
        id: "02",
        type: "Claim",
        label: "Marshall + Adrienne",
        posterior: 0.78,
        status: "active",
        x: 125,
        y: 148,
        payload: {
          source_type: "user_input",
          stance: "asserted",
          layer: "question",
          trajectory_index: 1,
          belief: "The target film is linked to John Marshall and a collaborator named Adrienne.",
        },
      },
      {
        id: "03",
        type: "Claim",
        label: "Reused footage",
        posterior: 0.76,
        status: "estimated",
        x: 125,
        y: 204,
        payload: {
          source_type: "user_input",
          stance: "asserted",
          layer: "question",
          trajectory_index: 1,
          belief: "The clue says related footage appeared in The Gods Must Be Crazy.",
        },
      },
      {
        id: "07",
        type: "BeliefVariable",
        label: "Recall collaborator",
        posterior: 0.56,
        status: "active",
        x: 255,
        y: 64,
        payload: {
          source_type: "llm_reasoning",
          stance: "recalled",
          layer: "hypothesis",
          trajectory_index: 2,
          belief: "John Marshall worked on films about the !Kung people, possibly with Adrienne LaFrance.",
        },
      },
      {
        id: "09",
        type: "BeliefVariable",
        label: "Early title guess",
        posterior: 0.43,
        status: "contradicted",
        x: 255,
        y: 138,
        payload: {
          source_type: "llm_reasoning",
          stance: "speculated",
          layer: "hypothesis",
          trajectory_index: 2,
          belief: "The title might be The !Kung, but this is only an early guess.",
        },
      },
      {
        id: "19",
        type: "Evidence",
        label: "Search query",
        posterior: 0.7,
        status: "issued",
        x: 255,
        y: 300,
        payload: {
          source_type: "tool_call",
          stance: "asserted",
          layer: "search",
          trajectory_index: 2,
          belief: "The assistant searches for John Marshall, Adrienne LaFrance, !Kung, and 1980 film title.",
        },
      },
      {
        id: "21",
        type: "Evidence",
        label: "N!ai documentary",
        posterior: 0.91,
        status: "retrieved",
        x: 385,
        y: 72,
        payload: {
          source_type: "tool_result",
          stance: "asserted",
          layer: "retrieval",
          trajectory_index: 3,
          belief: "N!ai, the Story of a !Kung Woman is a documentary film by ethnographic filmmaker John Marshall.",
        },
      },
      {
        id: "22",
        type: "Evidence",
        label: "1980 Odyssey",
        posterior: 0.88,
        status: "retrieved",
        x: 385,
        y: 132,
        payload: {
          source_type: "tool_result",
          stance: "asserted",
          layer: "retrieval",
          trajectory_index: 3,
          belief: "The film was first broadcast in 1980 as part of the PBS Odyssey series.",
        },
      },
      {
        id: "26",
        type: "Evidence",
        label: "Gods scene",
        posterior: 0.84,
        status: "retrieved",
        x: 385,
        y: 220,
        payload: {
          source_type: "tool_result",
          stance: "asserted",
          layer: "retrieval",
          trajectory_index: 3,
          belief: "The film contains a scene from the filming of The Gods Must Be Crazy.",
        },
      },
      {
        id: "37",
        type: "Claim",
        label: "Guess corrected",
        posterior: 0.77,
        status: "revised",
        x: 515,
        y: 230,
        payload: {
          source_type: "llm_reasoning",
          stance: "judged",
          layer: "revision",
          trajectory_index: 4,
          belief: "The retrieved evidence points to N!ai rather than the earlier guessed title.",
        },
      },
      {
        id: "47",
        type: "Decision",
        label: "Final title",
        posterior: 0.9,
        status: "accepted",
        x: 775,
        y: 190,
        payload: {
          source_type: "assistant_other",
          stance: "judged",
          layer: "answer",
          trajectory_index: 6,
          belief: "The answer is N!ai, the Story of a !Kung Woman.",
        },
      },
      {
        id: "55",
        type: "Evidence",
        label: "1980 release",
        posterior: 0.74,
        status: "retrieved",
        x: 645,
        y: 104,
        payload: {
          source_type: "tool_result",
          stance: "asserted",
          layer: "retrieval",
          trajectory_index: 5,
          belief: "The Gods Must Be Crazy was released in 1980.",
        },
      },
    ],
    edges: [
      {
        id: "e1",
        source: "01",
        target: "07",
        type: "informs",
        dir: "forward",
        note: "The !Kung clue prompted recall of Marshall collaborations.",
        weight: 0.82,
        direction: 1,
      },
      {
        id: "e2",
        source: "02",
        target: "07",
        type: "informs",
        dir: "forward",
        note: "The collaborator clue shaped the Adrienne LaFrance hypothesis.",
        weight: 0.74,
        direction: 1,
      },
      {
        id: "e3",
        source: "07",
        target: "09",
        type: "informs",
        dir: "forward",
        note: "The recalled collaboration led to the tentative title guess.",
        weight: 0.6,
        direction: 1,
      },
      {
        id: "e4",
        source: "01",
        target: "19",
        type: "informs",
        dir: "forward",
        note: "The !Kung clue was included in the search query.",
        weight: 0.76,
        direction: 1,
      },
      {
        id: "e5",
        source: "02",
        target: "19",
        type: "informs",
        dir: "forward",
        note: "The Marshall and Adrienne clue was included in the search query.",
        weight: 0.73,
        direction: 1,
      },
      {
        id: "e6",
        source: "19",
        target: "21",
        type: "informs",
        dir: "forward",
        note: "The search query returned the John Marshall documentary title.",
        weight: 0.9,
        direction: 1,
      },
      {
        id: "e7",
        source: "19",
        target: "22",
        type: "informs",
        dir: "forward",
        note: "The search query returned the 1980 Odyssey broadcast fact.",
        weight: 0.86,
        direction: 1,
      },
      {
        id: "e8",
        source: "19",
        target: "26",
        type: "informs",
        dir: "forward",
        note: "The search query returned the Gods Must Be Crazy connection.",
        weight: 0.82,
        direction: 1,
      },
      {
        id: "e9",
        source: "21",
        target: "37",
        type: "informs",
        dir: "forward",
        note: "The retrieved title corrected the early title guess.",
        weight: 0.84,
        direction: 1,
      },
      {
        id: "e10",
        source: "21",
        target: "47",
        type: "informs",
        dir: "forward",
        note: "The documentary title directly informed the final answer.",
        weight: 0.91,
        direction: 1,
      },
      {
        id: "e11",
        source: "22",
        target: "47",
        type: "informs",
        dir: "forward",
        note: "The 1980 broadcast date supported the final identification.",
        weight: 0.88,
        direction: 1,
      },
      {
        id: "e12",
        source: "26",
        target: "47",
        type: "informs",
        dir: "forward",
        note: "The Gods Must Be Crazy clue supported the final identification.",
        weight: 0.84,
        direction: 1,
      },
      {
        id: "e13",
        source: "37",
        target: "09",
        type: "contradicts",
        dir: "backward",
        note: "The later evidence identifies the title as N!ai rather than The !Kung.",
        weight: 0.77,
        direction: -1,
      },
      {
        id: "e14",
        source: "21",
        target: "09",
        type: "confirms",
        dir: "backward",
        note: "The retrieval confirms that the earlier search path was in the right film family.",
        weight: 0.62,
        direction: 1,
      },
      {
        id: "e15",
        source: "55",
        target: "26",
        type: "extends",
        dir: "backward",
        note: "The release date extends the reused-footage clue with a specific 1980 context.",
        weight: 0.7,
        direction: 1,
      },
    ],
    summary: {
      claims: 0,
      belief_variables: 0,
      evidence: 0,
      factors: 0,
      decisions: 0,
    },
    },
    "sample",
    defaults,
  );
  return memory;
}





