#!/usr/bin/env node

/** Compare the recorded compact renderer with relation-adjacent trail layout. */

import { readFileSync, readdirSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, join, basename } from "node:path";
import { formatCompactBcgDialogueContext } from "../agent-cli/dist/core/context/bcg-context.js";

function parseArgs(argv) {
	const values = {};
	for (let index = 0; index < argv.length; index += 2) {
		const key = argv[index];
		if (!key?.startsWith("--") || argv[index + 1] === undefined) {
			throw new Error("Usage: analyze_compact_graph_layout.mjs --run-dir PATH --graphs-dir PATH --output-dir PATH");
		}
		values[key.slice(2)] = argv[index + 1];
	}
	for (const key of ["run-dir", "graphs-dir", "output-dir"]) {
		if (!values[key]) throw new Error(`Missing --${key}`);
	}
	return values;
}

function readJsonl(path) {
	if (!existsSync(path)) return [];
	return readFileSync(path, "utf8")
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line));
}

function graphSessions(graphsDir) {
	const sessions = new Map();
	for (const name of readdirSync(graphsDir)) {
		const directory = join(graphsDir, name);
		const latest = join(directory, "belief_graph_latest.json");
		if (!existsSync(latest)) continue;
		const problemId = String(JSON.parse(readFileSync(latest, "utf8")).problem_id ?? "");
		const sessionId = problemId.split(":", 1)[0];
		if (sessionId) sessions.set(sessionId, directory);
	}
	return sessions;
}

const NODE_RE = /^- \[B(\d+)\] (?!depends_on|supplements|contradicts)(.*)$/gm;
const RELATION_RE = /^- \[B(\d+)\] (depends_on|supplements|contradicts) \[B(\d+)\]$/gm;

function renderedGraph(text) {
	const nodeIds = Array.from(text.matchAll(NODE_RE), (match) => Number(match[1]));
	const relations = Array.from(text.matchAll(RELATION_RE), (match) => ({
		from: Number(match[1]),
		type: match[2],
		to: Number(match[3]),
	}));
	return { nodeIds, relations };
}

function relationKey(relation) {
	return `${relation.from}:${relation.type}:${relation.to}`;
}

function mean(values) {
	return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function endpointDistances(graph) {
	const positions = new Map(graph.nodeIds.map((nodeId, index) => [nodeId, index]));
	return graph.relations.flatMap((relation) => {
		const from = positions.get(relation.from);
		const to = positions.get(relation.to);
		return from === undefined || to === undefined ? [] : [Math.abs(from - to)];
	});
}

function sameSet(left, right) {
	return left.size === right.size && Array.from(left).every((value) => right.has(value));
}

const args = parseArgs(process.argv.slice(2));
const runDir = resolve(args["run-dir"]);
const graphsDir = resolve(args["graphs-dir"]);
const outputDir = resolve(args["output-dir"]);
const modeDir = join(runDir, "browsecomp", "bcg");
const sessions = graphSessions(graphsDir);
const rows = [];

for (const taskName of readdirSync(join(modeDir, "tasks")).filter((name) => name.endsWith(".json")).sort()) {
	const taskId = basename(taskName, ".json");
	const trajectory = readJsonl(join(modeDir, "trajectories", `browsecomp-bcg-${taskId}.jsonl`));
	const sessionId = String(trajectory[0]?.id ?? "");
	const graphDir = sessions.get(sessionId);
	if (!graphDir) continue;
	const snapshots = new Map(
		readJsonl(join(graphDir, "belief_graph.jsonl")).map((snapshot) => [snapshot.stream_turn_index, snapshot]),
	);
	const traces = readJsonl(join(modeDir, "graph-contexts", `browsecomp-bcg-${taskId}.jsonl`));
	for (let index = 0; index < traces.length; index += 1) {
		const trace = traces[index];
		if (!trace.text || !Array.isArray(trace.selectedNodeIds)) continue;
		const snapshot = snapshots.get(trace.streamTurnIndex);
		if (!snapshot) continue;
		const candidateText = formatCompactBcgDialogueContext(
			snapshot,
			true,
			new Set(trace.selectedNodeIds),
			new Set(trace.selectedRelationIds ?? []),
			true,
		);
		const baseline = renderedGraph(trace.text);
		const candidate = renderedGraph(candidateText);
		const baselineDistances = endpointDistances(baseline);
		const candidateDistances = endpointDistances(candidate);
		const baselineRelations = new Set(baseline.relations.map(relationKey));
		const candidateRelations = new Set(candidate.relations.map(relationKey));
		rows.push({
			case_id: `${taskId}:graph-context:${index + 1}`,
			stream_turn_index: trace.streamTurnIndex,
			baseline_chars: trace.text.length,
			candidate_chars: candidateText.length,
			baseline_nodes: baseline.nodeIds.length,
			candidate_nodes: candidate.nodeIds.length,
			baseline_relations: baseline.relations.length,
			candidate_relations: candidate.relations.length,
			node_set_equal: sameSet(new Set(baseline.nodeIds), new Set(candidate.nodeIds)),
			relation_set_equal: sameSet(baselineRelations, candidateRelations),
			baseline_endpoint_distance_mean: mean(baselineDistances),
			candidate_endpoint_distance_mean: mean(candidateDistances),
			baseline_endpoint_distances: baselineDistances.length,
			candidate_endpoint_distances: candidateDistances.length,
		});
	}
}

const weightedDistance = (prefix) => {
	const total = rows.reduce((sum, row) => sum + row[`${prefix}_endpoint_distances`], 0);
	return total === 0
		? null
		: rows.reduce(
				(sum, row) =>
					sum + (row[`${prefix}_endpoint_distance_mean`] ?? 0) * row[`${prefix}_endpoint_distances`],
				0,
			) / total;
};
const summary = {
	run_dir: runDir,
	graphs_dir: graphsDir,
	snapshots: rows.length,
	node_set_equal: rows.filter((row) => row.node_set_equal).length,
	relation_set_equal: rows.filter((row) => row.relation_set_equal).length,
	baseline_chars_mean: mean(rows.map((row) => row.baseline_chars)),
	candidate_chars_mean: mean(rows.map((row) => row.candidate_chars)),
	baseline_endpoint_distance_mean: weightedDistance("baseline"),
	candidate_endpoint_distance_mean: weightedDistance("candidate"),
	baseline_relations_mean: mean(rows.map((row) => row.baseline_relations)),
	candidate_relations_mean: mean(rows.map((row) => row.candidate_relations)),
};

mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, "per_snapshot.jsonl"), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
writeFileSync(join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
