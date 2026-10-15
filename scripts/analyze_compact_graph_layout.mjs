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

function relationContinuity(relations) {
	if (relations.length < 2) return null;
	let connected = 0;
	for (let index = 1; index < relations.length; index += 1) {
		const previous = relations[index - 1];
		const current = relations[index];
		if (
			previous.from === current.from ||
			previous.from === current.to ||
			previous.to === current.from ||
			previous.to === current.to
		) {
			connected += 1;
		}
	}
	return connected / (relations.length - 1);
}

function answerRelationCoverage(relations, answerIds) {
	if (answerIds.size === 0) return null;
	return relations.some((relation) => answerIds.has(relation.from) || answerIds.has(relation.to)) ? 1 : 0;
}

function sameSet(left, right) {
	return left.size === right.size && Array.from(left).every((value) => right.has(value));
}

function normalizedAnswer(value) {
	return String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function answerNodeIds(snapshot, answer) {
	const normalized = normalizedAnswer(answer);
	if (!normalized) return new Set();
	const tokens = normalized.split(" ");
	return new Set(
		(snapshot.beliefs ?? [])
			.filter((node) => {
				const content = normalizedAnswer(node.belief ?? node.decision ?? "");
				return normalized.length > 0 &&
					(content.includes(normalized) ||
						(tokens.length > 1 && tokens.every((token) => content.split(" ").includes(token))));
			})
			.map((node) => node.id),
	);
}

function firstSelectedPosition(nodeIds, answerIds) {
	const positions = nodeIds.flatMap((nodeId, index) => (answerIds.has(nodeId) ? [index + 1] : []));
	return positions.length > 0 ? Math.min(...positions) : null;
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
	const task = JSON.parse(readFileSync(join(modeDir, "tasks", taskName), "utf8"));
	const answer = task.reference_answers?.[0] ?? "";
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
		const answerIds = answerNodeIds(snapshot, answer);
		const answerNodeMetadata = (snapshot.beliefs ?? [])
			.filter((node) => answerIds.has(node.id))
			.map((node) => ({
				id: node.id,
				confidence: node.confidence ?? null,
				stance: node.stance ?? null,
				node_type: node.node_type ?? null,
				extraction_method: node.extraction_method ?? null,
				source_turn: node.source?.turn_id ?? null,
				selected: trace.selectedNodeIds.includes(node.id),
			}));
		const baselineAnswerPosition = firstSelectedPosition(baseline.nodeIds, answerIds);
		const candidateAnswerPosition = firstSelectedPosition(candidate.nodeIds, answerIds);
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
			baseline_relation_continuity: relationContinuity(baseline.relations),
			candidate_relation_continuity: relationContinuity(candidate.relations),
			answer_nodes_available: answerIds.size,
			answer_node_metadata: answerNodeMetadata,
			baseline_answer_first_position: baselineAnswerPosition,
			candidate_answer_first_position: candidateAnswerPosition,
			baseline_answer_relation_coverage: answerRelationCoverage(baseline.relations, answerIds),
			candidate_answer_relation_coverage: answerRelationCoverage(candidate.relations, answerIds),
			baseline_answer_first_position_fraction:
				baselineAnswerPosition === null ? null : baselineAnswerPosition / baseline.nodeIds.length,
			candidate_answer_first_position_fraction:
				candidateAnswerPosition === null ? null : candidateAnswerPosition / candidate.nodeIds.length,
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
	baseline_relation_continuity_mean: mean(
		rows.flatMap((row) => row.baseline_relation_continuity === null ? [] : [row.baseline_relation_continuity]),
	),
	candidate_relation_continuity_mean: mean(
		rows.flatMap((row) => row.candidate_relation_continuity === null ? [] : [row.candidate_relation_continuity]),
	),
	answer_bearing_snapshots: rows.filter((row) => row.answer_nodes_available > 0).length,
	baseline_answer_first_position_mean: mean(
		rows.flatMap((row) =>
			row.baseline_answer_first_position === null ? [] : [row.baseline_answer_first_position],
		),
	),
	candidate_answer_first_position_mean: mean(
		rows.flatMap((row) =>
			row.candidate_answer_first_position === null ? [] : [row.candidate_answer_first_position],
		),
	),
	baseline_answer_relation_coverage_mean: mean(
		rows.flatMap((row) => row.baseline_answer_relation_coverage === null ? [] : [row.baseline_answer_relation_coverage]),
	),
	candidate_answer_relation_coverage_mean: mean(
		rows.flatMap((row) => row.candidate_answer_relation_coverage === null ? [] : [row.candidate_answer_relation_coverage]),
	),
	baseline_answer_first_position_fraction_mean: mean(
		rows.flatMap((row) =>
			row.baseline_answer_first_position_fraction === null
				? []
				: [row.baseline_answer_first_position_fraction],
		),
	),
	candidate_answer_first_position_fraction_mean: mean(
		rows.flatMap((row) =>
			row.candidate_answer_first_position_fraction === null
				? []
				: [row.candidate_answer_first_position_fraction],
		),
	),
};

mkdirSync(outputDir, { recursive: true });
writeFileSync(join(outputDir, "per_snapshot.jsonl"), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
writeFileSync(join(outputDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
