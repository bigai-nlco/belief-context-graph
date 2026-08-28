import type { AgentMessage } from "@bigai-nlco/bcg-agent-core";
import { BcgClient, type BcgTurnPayload } from "./bcg-client.ts";
import type { BcgSnapshot, BcgTurn } from "./bcg-contract.types.ts";
import type { BcgContextSelectionResponse } from "./bcg-contract.types.ts";
import type { BcgGraphSelection, BcgGraphView } from "../settings-manager.ts";
import { bashExecutionToText } from "../messages.ts";

const GRAPH_PREFIX =
	"The following belief graph captures the reasoning trajectory so far. " +
	"These are preliminary beliefs derived from prior turns and may contain errors or incomplete information. " +
	"Use them to guide the next action, but do not treat them as verified evidence.";

const GRAPH_DIALOGUE_CONTEXT_GUIDE =
	"<context_blocks_guide>\n" +
	"The dialogue-encoded context below holds beliefs derived from earlier turns. " +
	"A belief is a self-contained claim or reasoning unit, such as a fact, hypothesis, intermediate conclusion, or decision. " +
	"Those earlier turns have been omitted from the raw conversation context and are represented by this belief context instead. " +
	"The beliefs may contain errors or incomplete information. Use each belief's confidence to judge how trustworthy its content is, " +
	"and do not treat any belief as verified evidence solely because it appears here. " +
	"Relation direction is literal: `A depends_on B` means A requires B as a premise, evidence, input, constraint, or context; " +
	"`A supplements B` means A adds compatible detail or evidence to B without refuting it; " +
	"`A contradicts B` means A conflicts with, corrects, negates, or replaces B. " +
	"Use this context to avoid repeating searches that were already performed, and do not repeatedly search for the same information.\n" +
	"</context_blocks_guide>";

const COMPACT_GRAPH_DIALOGUE_CONTEXT_GUIDE =
	"<context_blocks_guide>\n" +
	"Earlier raw turns are omitted; the Candidate evidence, Search history, and Relation paths below are Graph memory, not verified evidence. " +
	"A belief is a self-contained fact, hypothesis, intermediate conclusion, or decision. " +
	"Confidence estimates reliability, not answer relevance. Factual beliefs are candidate evidence; search-action beliefs only record prior work. " +
	"A specific low-confidence belief that fills the requested value remains a candidate to verify; a generic high-confidence fact is not an answer unless it satisfies the question. " +
	"Relations record reasoning or provenance, not truth by themselves. Direction is literal: `A depends_on B` means A requires B as a premise, evidence, input, constraint, or context; " +
	"`A supplements B` adds compatible detail or evidence; `A contradicts B` conflicts with, corrects, negates, or replaces B. " +
	"Treat each connected relation path as an investigation branch: follow outgoing relations from a candidate to its premises and incoming relations to later checks or results. " +
	"Compare plausible candidates against every pivotal constraint in the original question using direct, source-grounded evidence. " +
	"Do not let recency, confidence alone, or many beliefs about the same candidate substitute for covering distinct constraints; missing edges or empty searches do not disprove a candidate. " +
	"Before searching, identify the unresolved candidate-constraint pair whose answer would most change the final choice. Search that gap with a discriminating query, batch independent gaps when useful, and avoid queries that only reconfirm an already-supported constraint. " +
	"If a concrete answer value is already supported across the pivotal constraints with no retained contradiction, answer without re-verifying every clue. Otherwise keep a plausible alternative until source-grounded evidence resolves the differentiating constraint. " +
	"Preserve the exact supported value; add no unsupported precision.\n" +
	"</context_blocks_guide>";

const DIALOGUE_BOS = "<｜begin▁of▁sentence｜>";
const DIALOGUE_EOS = "<｜end▁of▁sentence｜>";
const DIALOGUE_USER = "<｜User｜>";
const DIALOGUE_ASSISTANT = "<｜Assistant｜>";
const DIALOGUE_VALID_ROLES = new Set(["system", "user", "assistant"]);

export const BCG_TURN_LIMIT_MARKER = "BCG_TURN_LIMIT_EXCEEDED";

export class BcgTurnLimitError extends Error {
	readonly submittedTurns: number;
	readonly maxTurns: number;

	constructor(submittedTurns: number, maxTurns: number) {
		super(
			`${BCG_TURN_LIMIT_MARKER}: Graph message limit ${maxTurns} reached ` +
				`after ${submittedTurns} submitted messages; this task is a failure.`,
		);
		this.name = "BcgTurnLimitError";
		this.submittedTurns = submittedTurns;
		this.maxTurns = maxTurns;
	}
}


export interface BcgContextManagerOptions {
	baseUrl: string;
	problemId: string;
	recentTurns: number;
	maxTurns: number;
	timeoutMs: number;
	finalizationTimeoutMs?: number;
	includeRelations: boolean;
	graphView?: BcgGraphView;
	graphSelection?: BcgGraphSelection;
	getSystemPrompt: () => string;
	getInitialUserMessage?: () => AgentMessage | undefined;
	fetch?: typeof globalThis.fetch;
	onWarning?: (message: string) => void;
	onGraphContext?: (trace: BcgGraphContextTrace) => void;
}

export interface BcgGraphContextTrace {
	view: BcgGraphView;
	streamTurnIndex?: number;
	nTurnsIngested?: number;
	nNodes: number;
	nRelations: number;
	chars: number;
	text: string;
	selectionStrategy?: "ranked" | "connected" | "focused";
	selectionRetrieval?: string;
	selectedNodeIds?: number[];
	selectedRelationIds?: number[];
}

interface SerializedMessage {
	messages: AgentMessage[];
	payload: BcgTurnPayload | undefined;
}


function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function compactJson(value: unknown): string {
	if (value === null) {
		return "null";
	}
	if (Array.isArray(value)) {
		return `[${value.map((item) => compactJson(item)).join(", ")}]`;
	}
	if (isRecord(value)) {
		const fields = Object.entries(value)
			.filter(([, item]) => item !== undefined)
			.map(([key, item]) => `${JSON.stringify(key)}: ${compactJson(item)}`);
		return `{${fields.join(", ")}}`;
	}
	try {
		return JSON.stringify(value) ?? "null";
	} catch {
		return JSON.stringify(String(value));
	}
}

export function contextContentToText(content: unknown): string {
	if (typeof content === "string") {
		return content;
	}
	if (!Array.isArray(content)) {
		return "";
	}

	const parts: string[] = [];
	for (const block of content) {
		if (!isRecord(block) || typeof block.type !== "string") {
			continue;
		}
		switch (block.type) {
			case "text":
				if (typeof block.text === "string") {
					parts.push(block.text);
				}
				break;
			case "thinking":
				if (typeof block.thinking === "string" && block.thinking.trim()) {
					parts.push(`<thinking>\n${block.thinking.trim()}\n</thinking>`);
				}
				break;
			case "image":
				parts.push("[Image omitted from BCG text context]");
				break;
			case "toolCall": {
				const name = typeof block.name === "string" ? block.name : "unknown";
				const args = "arguments" in block ? block.arguments : {};
				const id = typeof block.id === "string" ? block.id : undefined;
				parts.push(
					`<tool_call>\n${compactJson({ id, name, arguments: args })}\n</tool_call>`,
				);
				break;
			}
		}
	}
	return parts.join("\n\n").trim();
}

export function contextToolResultsToText(messages: AgentMessage[]): string {
	return messages
		.filter((message) => message.role === "toolResult")
		.map((message) => {
			const content = contextContentToText(message.content);
			return (
				"<tool_result>\n" +
				compactJson({
					tool_call_id: message.toolCallId,
					name: message.toolName,
					is_error: message.isError,
					content,
				}) +
				"\n</tool_result>"
			);
		})
		.join("\n\n");
}

export function contextMessageText(message: AgentMessage): string {
	switch (message.role) {
		case "user":
		case "assistant":
			return contextContentToText(message.content);
		case "toolResult": {
			return contextToolResultsToText([message]);
		}
		case "bashExecution":
			return bashExecutionToText(message);
		case "custom":
			return contextContentToText(message.content);
		case "branchSummary":
			return message.summary;
		case "compactionSummary":
			return message.summary;
	}
}

function graphRole(message: AgentMessage): BcgTurnPayload["role"] | undefined {
	switch (message.role) {
		case "user":
		case "custom":
		case "branchSummary":
		case "compactionSummary":
			return "user";
		case "assistant":
			return "assistant";
		case "toolResult":
		case "bashExecution":
			return "tool";
	}
}

export function contextMessageKey(message: AgentMessage): string {
	const timestamp = "timestamp" in message ? String(message.timestamp) : "";
	return `${message.role}\u0000${timestamp}\u0000${contextMessageText(message)}`;
}

function hasAssistant(messages: AgentMessage[]): boolean {
	return messages.some((message) => message.role === "assistant");
}

/**
 * Split the transcript after the permanent initial user input into model turns.
 * A turn starts with an assistant message (or a later user input) and owns all
 * following tool results until the next assistant/user boundary.
 */
export function splitBcgTurns(messages: AgentMessage[]): AgentMessage[][] {
	const turns: AgentMessage[][] = [];
	let current: AgentMessage[] = [];

	for (const message of messages) {
		if (message.role === "user") {
			if (current.length > 0) {
				turns.push(current);
			}
			current = [message];
			continue;
		}

		if (message.role === "assistant" && hasAssistant(current)) {
			turns.push(current);
			current = [message];
			continue;
		}

		current.push(message);
	}

	if (current.length > 0) {
		turns.push(current);
	}
	return turns;
}

export function partitionContextTurns(
	turns: AgentMessage[][],
	recentTurns: number,
): {
	evicted: AgentMessage[][];
	retained: AgentMessage[][];
} {
	if (recentTurns < 0) {
		return { evicted: [], retained: turns };
	}

	let completedTurnsToKeep = recentTurns;
	let cut = turns.length;
	while (cut > 0) {
		const turn = turns[cut - 1];
		if (!hasAssistant(turn)) {
			cut -= 1;
			continue;
		}
		if (completedTurnsToKeep <= 0) {
			break;
		}
		completedTurnsToKeep -= 1;
		cut -= 1;
	}
	return {
		evicted: turns.slice(0, cut),
		retained: turns.slice(cut),
	};
}

export function formatBcgMarkdown(snapshot: BcgSnapshot, includeRelations = true): string {
	const beliefs = Array.isArray(snapshot.beliefs) ? snapshot.beliefs : [];
	if (beliefs.length === 0) {
		return "";
	}

	const lines = ["## Belief Graph", "", GRAPH_PREFIX, ""];
	for (const belief of beliefs) {
		lines.push(`- **[${belief.id ?? "?"}]** ${belief.belief ?? ""}`);
		if (belief.tool_name) {
			lines.push(`  - Tool: ${belief.tool_name ?? "tool"}`);
		}
		if (belief.query) {
			lines.push(`  - Query: ${belief.query}`);
		}
		if (belief.tool_arguments) {
			lines.push(`  - Arguments: ${compactJson(belief.tool_arguments)}`);
		}
	}

	const relations = includeRelations ? (snapshot.relations ?? []) : [];
	if (relations.length > 0) {
		lines.push("", "### Relations");
		for (const relation of relations) {
			const note = relation.note ? ` — ${relation.note}` : "";
			lines.push(
				`- [${relation.from_id ?? "?"}] → [${relation.to_id ?? "?"}] ` + `(${relation.type ?? "informs"})${note}`,
			);
		}
	}
	return lines.join("\n");
}

interface GraphDialogueRelation {
	direction: "outgoing";
	to: number;
	type: string;
	reason?: string;
}

interface GraphDialogueBeliefMessage {
	id: number;
	role: string;
	content: string;
	toolName: string | null;
	query: string | null;
	toolArguments: Record<string, unknown> | null;
	relations: GraphDialogueRelation[];
	confidence: number | null;
}

type CompactRelation = BcgSnapshot["relations"][number];

interface CompactTrailLine {
	text: string;
	kind: "node" | "tree_relation" | "cross_relation" | "separator";
	relationType?: CompactRelation["type"];
}

const COMPACT_GRAPH_CHAR_BUDGET = 8_000;
const COMPACT_FACT_CHAR_BUDGET = 4_900;
const COMPACT_SEARCH_CHAR_BUDGET = 1_700;
const COMPACT_RELATION_PATH_LIMIT = 3;

function compactConfidence(belief: BcgSnapshot["beliefs"][number]): string {
	return typeof belief.confidence === "number" ? ` (confidence ${belief.confidence.toFixed(2)})` : "";
}

function compactSourceTurn(belief: BcgSnapshot["beliefs"][number]): number {
	return isRecord(belief.source) && typeof belief.source.turn_id === "number"
		? belief.source.turn_id
		: -1;
}

function isCompactEmptySearchResultBelief(belief: BcgSnapshot["beliefs"][number]): boolean {
	return /^The (?:web_search|\S+ tool) (?:tool )?returned no results\.?$/i.test(
		belief.belief ?? "",
	);
}

function isCompactSearchHistoryBelief(belief: BcgSnapshot["beliefs"][number]): boolean {
	return (
		belief.extraction_method === "rule_tool_call" ||
		typeof belief.query === "string"
	);
}

function compactFactPriority(belief: BcgSnapshot["beliefs"][number]): number {
	const confidence = typeof belief.confidence === "number" ? belief.confidence : 0;
	const semanticBonus = belief.extraction_method === "compact_llm_tool_result" ? 100 : 0;
	const rawResultPenalty = belief.extraction_method === "rule_tool_result" ? -100 : 0;
	return confidence * 1_000 + semanticBonus + rawResultPenalty;
}

function addCompactBeliefsWithinBudget(
	ordered: BcgSnapshot["beliefs"],
	selected: Map<number, BcgSnapshot["beliefs"][number]>,
	lines: string[],
	charBudget: number,
	includeConfidence: boolean,
): void {
	let usedChars = 0;
	for (const belief of ordered) {
		if (selected.has(belief.id) || !belief.belief) continue;
		const confidence = includeConfidence ? compactConfidence(belief) : "";
		const line = `- [B${belief.id}] ${belief.belief}${confidence}`;
		if (usedChars + line.length + 1 > charBudget) continue;
		selected.set(belief.id, belief);
		lines.push(line);
		usedChars += line.length + 1;
	}
}

function compactBeliefLine(belief: BcgSnapshot["beliefs"][number]): string {
	const confidence = isCompactSearchHistoryBelief(belief) ? "" : compactConfidence(belief);
	return `- [B${belief.id}] ${belief.belief}${confidence}`;
}

function compactRelationLine(relation: CompactRelation): string {
	return `- [B${relation.from_id}] ${relation.type} [B${relation.to_id}]`;
}

/**
 * Arrange selected relations as short, endpoint-contiguous paths.
 *
 * Candidate belief order is intentionally handled separately so graph
 * traversal cannot bury answer evidence. This traversal changes relation order
 * only; no belief, relation, or endpoint is rewritten.
 */
function compactRelationPathLines(
	retained: BcgSnapshot["beliefs"],
	relations: CompactRelation[],
): CompactTrailLine[] {
	const byId = new Map(retained.map((belief) => [belief.id, belief]));
	const orderedIds = [
		...retained
			.filter((belief) => !isCompactSearchHistoryBelief(belief))
			.sort(
				(left, right) =>
					compactFactPriority(right) - compactFactPriority(left) ||
					compactSourceTurn(right) - compactSourceTurn(left) ||
					right.id - left.id,
			)
			.map((belief) => belief.id),
		...retained
			.filter(isCompactSearchHistoryBelief)
			.sort(
				(left, right) =>
					compactSourceTurn(right) - compactSourceTurn(left) || right.id - left.id,
			)
			.map((belief) => belief.id),
	];
	const priority = new Map(orderedIds.map((nodeId, index) => [nodeId, index]));
	const adjacency = new Map<
		number,
		Array<{ relation: CompactRelation; neighbor: number; incoming: boolean }>
	>();
	for (const relation of relations) {
		if (!byId.has(relation.from_id) || !byId.has(relation.to_id)) continue;
		const outgoing = adjacency.get(relation.from_id) ?? [];
		outgoing.push({ relation, neighbor: relation.to_id, incoming: false });
		adjacency.set(relation.from_id, outgoing);
		const incoming = adjacency.get(relation.to_id) ?? [];
		incoming.push({ relation, neighbor: relation.from_id, incoming: true });
		adjacency.set(relation.to_id, incoming);
	}

	const seenRelations = new Set<number>();
	const lines: CompactTrailLine[] = [];
	const edgePriority = (edge: {
		relation: CompactRelation;
		neighbor: number;
		incoming: boolean;
	}): [number, number, number, number] => [
		edge.relation.type === "contradicts" ? 0 : 1,
		edge.incoming ? 1 : 0,
		priority.get(edge.neighbor) ?? Number.MAX_SAFE_INTEGER,
		edge.relation.id,
	];
	const compareEdges = (
		left: { relation: CompactRelation; neighbor: number; incoming: boolean },
		right: { relation: CompactRelation; neighbor: number; incoming: boolean },
	): number => {
		const leftKey = edgePriority(left);
		const rightKey = edgePriority(right);
		for (let index = 0; index < leftKey.length; index += 1) {
			if (leftKey[index] !== rightKey[index]) return leftKey[index] - rightKey[index];
		}
		return 0;
	};

	const visit = (
		nodeId: number,
		trail: { edges: number; nodes: Set<number> },
	): void => {
		if (!byId.has(nodeId) || trail.edges >= COMPACT_RELATION_PATH_LIMIT) return;
		trail.nodes.add(nodeId);
		const edges = [...(adjacency.get(nodeId) ?? [])].sort(compareEdges);
		for (const edge of edges) {
			if (trail.edges >= COMPACT_RELATION_PATH_LIMIT) break;
			if (trail.nodes.has(edge.neighbor) || seenRelations.has(edge.relation.id)) continue;
			seenRelations.add(edge.relation.id);
			trail.edges += 1;
			lines.push({
				text: compactRelationLine(edge.relation),
				kind: "tree_relation",
				relationType: edge.relation.type,
			});
			visit(edge.neighbor, trail);
		}
	};

	for (const nodeId of orderedIds) {
		const before = lines.length;
		visit(nodeId, { edges: 0, nodes: new Set<number>() });
		if (before > 0 && lines.length > before) {
			lines.splice(before, 0, { text: "", kind: "separator" });
		}
	}

	// Defensive: retain any selected relation that was not reached because of
	// malformed duplicate IDs, but mark it as a lower-priority cross-link.
	for (const relation of relations) {
		if (seenRelations.has(relation.id)) continue;
		lines.push({
			text: compactRelationLine(relation),
			kind: "tree_relation",
			relationType: relation.type,
		});
	}
	return lines;
}

function trimCompactTrailRelationsToBudget(
	heading: string,
	sectionHeading: string,
	lines: CompactTrailLine[],
): CompactTrailLine[] {
	const result = [...lines];
	const renderedChars = (): number =>
		heading.length + 2 + sectionHeading.length + 1 + result.map((line) => line.text).join("\n").length;
	const removableKinds: CompactTrailLine["kind"][] = ["cross_relation", "tree_relation"];
	for (const kind of removableKinds) {
		while (renderedChars() > COMPACT_GRAPH_CHAR_BUDGET) {
			let removable = -1;
			for (let index = result.length - 1; index >= 0; index -= 1) {
				if (result[index].kind !== kind) continue;
				// Preserve answer-changing conflicts until every other relation type
				// of this class has been considered.
				if (result[index].relationType === "contradicts") continue;
				removable = index;
				break;
			}
			if (removable < 0) break;
			result.splice(removable, 1);
		}
	}
	while (renderedChars() > COMPACT_GRAPH_CHAR_BUDGET) {
		let removable = -1;
		for (let index = result.length - 1; index >= 0; index -= 1) {
			if (result[index].kind === "cross_relation" || result[index].kind === "tree_relation") {
				removable = index;
				break;
			}
		}
		if (removable < 0) break;
		result.splice(removable, 1);
	}
	return result;
}

/**
 * Render a bounded historical-dialogue memory view without mutating the graph.
 *
 * This renderer is selection-only: it may omit beliefs to stay within budget,
 * but never rewrites belief text, derives query/result mappings, summarizes
 * tool-result metadata, or emits relations under a new label. Every displayed
 * graph item is copied from an existing belief.
 */
export function formatCompactBcgDialogueContext(
	snapshot: BcgSnapshot,
	includeRelations = true,
	selectedNodeIds?: ReadonlySet<number>,
	selectedRelationIds?: ReadonlySet<number>,
	relationTrailLayout = false,
): string {
	const beliefs = Array.isArray(snapshot.beliefs) ? snapshot.beliefs : [];
	if (beliefs.length === 0) {
		return "";
	}

	const retained = beliefs.filter((belief) => {
		const sourceTurn = compactSourceTurn(belief);
		// The initial system/user seed remains verbatim in every Agent request.
		// Empty search observations remain in the source graph for auditability,
		// but add no useful evidence to the bounded Agent-facing view.
		return (sourceTurn < 0 || sourceTurn > 1) &&
			!isCompactEmptySearchResultBelief(belief) &&
			(!selectedNodeIds || selectedNodeIds.has(belief.id));
	});
	if (retained.length === 0) {
		return "";
	}
	if (relationTrailLayout && selectedNodeIds) {
		const retainedIds = new Set(retained.map((belief) => belief.id));
		const relations = includeRelations
			? (snapshot.relations ?? [])
					.filter(
						(relation) =>
							retainedIds.has(relation.from_id) && retainedIds.has(relation.to_id),
					)
					.filter((relation) => !selectedRelationIds || selectedRelationIds.has(relation.id))
			: [];
		const heading = "### Earlier investigation memory";
		const facts = retained
			.filter((belief) => !isCompactSearchHistoryBelief(belief))
			.sort(
				(left, right) =>
					compactFactPriority(right) - compactFactPriority(left) ||
					compactSourceTurn(right) - compactSourceTurn(left) ||
					right.id - left.id,
			);
		const searches = retained
			.filter(isCompactSearchHistoryBelief)
			.sort(
				(left, right) =>
					compactSourceTurn(right) - compactSourceTurn(left) || right.id - left.id,
			);
		const candidateLines: CompactTrailLine[] = [];
		if (facts.length > 0) {
			candidateLines.push({ text: "#### Candidate evidence", kind: "separator" });
			candidateLines.push(
				...facts.map((belief) => ({ text: compactBeliefLine(belief), kind: "node" as const })),
			);
		}
		if (searches.length > 0) {
			if (candidateLines.length > 0) candidateLines.push({ text: "", kind: "separator" });
			candidateLines.push({ text: "#### Search history", kind: "separator" });
			candidateLines.push(
				...searches.map((belief) => ({ text: compactBeliefLine(belief), kind: "node" as const })),
			);
		}
		const relationLines = compactRelationPathLines(retained, relations);
		if (relationLines.length > 0) {
			candidateLines.push(
				{ text: "", kind: "separator" },
				{ text: "#### Relation paths", kind: "separator" },
				...relationLines,
			);
		}
		const trailLines = trimCompactTrailRelationsToBudget(
			heading,
			"",
			candidateLines,
		);
		const payload = `${heading}\n\n${trailLines.map((line) => line.text).join("\n")}`;
		return DIALOGUE_BOS + DIALOGUE_USER + payload + DIALOGUE_ASSISTANT + DIALOGUE_EOS;
	}
	const facts = retained
		.filter((belief) => !isCompactSearchHistoryBelief(belief))
		.sort(
			(left, right) =>
				compactFactPriority(right) - compactFactPriority(left) ||
				compactSourceTurn(right) - compactSourceTurn(left) ||
				right.id - left.id,
		);
	const searchHistory = retained
		.filter(isCompactSearchHistoryBelief)
		.sort(
			(left, right) =>
				compactSourceTurn(right) - compactSourceTurn(left) || right.id - left.id,
		);

	const heading = "### Earlier investigation memory";
	const selected = new Map<number, BcgSnapshot["beliefs"][number]>();
	const lines: string[] = [];
	if (facts.length > 0) {
		lines.push("#### Factual beliefs");
		addCompactBeliefsWithinBudget(
			facts,
			selected,
			lines,
			selectedNodeIds ? Number.MAX_SAFE_INTEGER : COMPACT_FACT_CHAR_BUDGET,
			true,
		);
	}
	if (searchHistory.length > 0) {
		lines.push("", "#### Search-history beliefs");
		addCompactBeliefsWithinBudget(
			searchHistory,
			selected,
			lines,
			selectedNodeIds ? Number.MAX_SAFE_INTEGER : COMPACT_SEARCH_CHAR_BUDGET,
			false,
		);
	}
	if (selected.size === 0) {
		return "";
	}
	if (includeRelations) {
		const selectedIds = new Set(selected.keys());
		const relationLines = (snapshot.relations ?? [])
			.filter((relation) => selectedIds.has(relation.from_id) && selectedIds.has(relation.to_id))
			.filter((relation) => !selectedRelationIds || selectedRelationIds.has(relation.id))
			.sort((left, right) => left.id - right.id)
			.map((relation) => `- [B${relation.from_id}] ${relation.type} [B${relation.to_id}]`);
		if (relationLines.length > 0) {
			lines.push("", "#### Retained relations");
			let totalChars = heading.length + 1 + lines.join("\n").length;
			for (const line of relationLines) {
				if (totalChars + line.length + 1 > COMPACT_GRAPH_CHAR_BUDGET) break;
				lines.push(line);
				totalChars += line.length + 1;
			}
		}
	}
	const payload = `${heading}\n\n${lines.join("\n")}`;
	return (
		DIALOGUE_BOS +
		DIALOGUE_USER +
		payload +
		DIALOGUE_ASSISTANT +
		DIALOGUE_EOS
	);
}

function formatDialogueBeliefMarkdown(message: GraphDialogueBeliefMessage): string {
	const lines = [
		`### Belief ${message.id}`,
		`**Content:** ${message.content}`,
	];
	if (message.query !== null) {
		lines.push(`**Tool:** ${message.toolName ?? "tool"}`);
		lines.push(`**Query:** ${message.query}`);
	} else if (message.toolName !== null) {
		lines.push(`**Tool:** ${message.toolName}`);
	}
	if (message.toolArguments !== null) {
		lines.push(`**Arguments:** ${compactJson(message.toolArguments)}`);
	}
	lines.push("**Relations:**");
	if (message.relations.length === 0) {
		lines.push("- None");
	} else {
		for (const relation of message.relations) {
			const fields = Object.entries(relation)
				.filter(([, value]) => value !== undefined && value !== "")
				.map(([key, value]) => `${key}=${value}`);
			lines.push(`- ${fields.join("; ")}`);
		}
	}
	lines.push(`**Confidence:** ${message.confidence ?? ""}`);
	return lines.join("\n");
}

function beliefRole(belief: BcgSnapshot["beliefs"][number]): string {
	const sourceRole = isRecord(belief.source) && typeof belief.source.role === "string" ? belief.source.role : undefined;
	const role = belief.role || sourceRole || "assistant";
	return DIALOGUE_VALID_ROLES.has(role) ? role : "user";
}

/** Encode a BCG snapshot as compact, role-marked dialogue context. */
export function formatBcgDialogueContext(snapshot: BcgSnapshot, includeRelations = true): string {
	const beliefs = Array.isArray(snapshot.beliefs) ? snapshot.beliefs : [];
	if (beliefs.length === 0) {
		return "";
	}

	const messages = new Map<number, GraphDialogueBeliefMessage>();
	for (const belief of beliefs) {
		messages.set(belief.id, {
			id: belief.id,
			role: beliefRole(belief),
			content: belief.belief ?? "",
			toolName: belief.tool_name ?? null,
			query: belief.query ?? null,
			toolArguments: belief.tool_arguments ?? null,
			relations: [],
			confidence: belief.confidence ?? null,
		});
	}

	if (includeRelations) {
		for (const relation of snapshot.relations ?? []) {
			const outgoing = messages.get(relation.from_id);
			if (outgoing) {
				outgoing.relations.push({
					direction: "outgoing",
					to: relation.to_id,
					type: relation.type ?? "informs",
					...(relation.note ? { reason: relation.note } : {}),
				});
			}
		}
	}

	const parts = [DIALOGUE_BOS];
	for (const message of messages.values()) {
		const payload = formatDialogueBeliefMarkdown(message);
		if (message.role === "system") {
			parts.push(payload);
		} else if (message.role === "assistant") {
			parts.push(`${DIALOGUE_ASSISTANT}${payload}${DIALOGUE_EOS}`);
		} else {
			parts.push(`${DIALOGUE_USER}${payload}`);
		}
	}
	return parts.join("");
}

function compactSelectionQuery(initialUser: AgentMessage, retained: AgentMessage[]): string {
	const question = contextMessageText(initialUser);
	const parts = [question];
	for (const message of retained) {
		const value = contextMessageText(message);
		if (value && value !== question) parts.push(value);
	}
	const combined = parts.join("\n\n");
	if (combined.length <= 12_000) return combined;
	const remaining = Math.max(0, 12_000 - question.length - 2);
	return `${question}\n\n${combined.slice(-remaining)}`;
}

function compactFocusSelectionQuery(initialUser: AgentMessage, retained: AgentMessage[]): string {
	const question = contextMessageText(initialUser);
	const parts = [question];
	for (const message of retained) {
		// Raw Tool Results are evidence candidates inside the Graph, not the
		// retrieval intent. Echoing them here creates a last-result feedback loop.
		if (message.role === "toolResult" || message.role === "bashExecution") continue;
		const value = contextMessageText(message);
		if (value && value !== question) parts.push(value);
	}
	const combined = parts.join("\n\n");
	if (combined.length <= 6_000) return combined;
	const remaining = Math.max(0, 6_000 - question.length - 2);
	return `${question}\n\n${combined.slice(-remaining)}`;
}

export class BcgContextManager {
	private readonly baseUrl: string;
	private readonly problemId: string;
	private readonly recentTurns: number;
	private readonly maxTurns: number;
	private readonly timeoutMs: number;
	private readonly finalizationTimeoutMs: number;
	private readonly includeRelations: boolean;
	private readonly graphView: BcgGraphView;
	private readonly graphSelection: BcgGraphSelection;
	private readonly getSystemPrompt: () => string;
	private readonly getInitialUserMessage?: () => AgentMessage | undefined;
	private readonly client: BcgClient;
	private readonly onWarning: (message: string) => void;
	private readonly onGraphContext?: (trace: BcgGraphContextTrace) => void;
	private readonly sentMessages = new WeakSet<object>();
	private seeded = false;
	private released = false;
	private submittedTurns = 0;
	private warned = false;
	private graphText = "";
	private latestSnapshot: BcgSnapshot | undefined;
	private reportableTokenUsage: Record<string, unknown> | undefined;
	private requestReady = false;
	private initialUserMessage: AgentMessage | undefined;

	constructor(options: BcgContextManagerOptions) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.problemId = options.problemId;
		this.recentTurns = Math.max(-1, Math.trunc(options.recentTurns));
		this.maxTurns = Math.max(1, Math.trunc(options.maxTurns));
		this.timeoutMs = Math.max(1, Math.trunc(options.timeoutMs));
		this.finalizationTimeoutMs = Math.max(
			1,
			Math.trunc(options.finalizationTimeoutMs ?? 900000),
		);
		this.includeRelations = options.includeRelations;
		this.graphView = options.graphView ?? "full";
		this.graphSelection = options.graphSelection ?? "connected";
		this.getSystemPrompt = options.getSystemPrompt;
		this.getInitialUserMessage = options.getInitialUserMessage;
		this.client = new BcgClient({ baseUrl: options.baseUrl, problemId: options.problemId, timeoutMs: options.timeoutMs, fetch: options.fetch });
		this.onWarning = options.onWarning ?? ((message) => console.warn(message));
		this.onGraphContext = options.onGraphContext;
	}

	async transform(messages: AgentMessage[], signal?: AbortSignal): Promise<AgentMessage[]> {
		this.requestReady = false;
		try {
			const initialUser = this.resolveInitialUser(messages);
			if (!initialUser) {
				throw new Error("the session has no initial user input");
			}

			if (!this.seeded) {
				const snapshot = await this.postTurns(
					[this.turnPayload("system", this.getSystemPrompt()), this.turnPayload("user", contextMessageText(initialUser))],
					signal,
				);
				this.updateSnapshot(snapshot);
				this.seeded = true;
				this.sentMessages.add(initialUser);
			}

			const initialKey = contextMessageKey(initialUser);
			let removedInitial = false;
			const rest = messages.filter((message) => {
				if (!removedInitial && contextMessageKey(message) === initialKey) {
					removedInitial = true;
					return false;
				}
				return true;
			});
			const { evicted, retained } = partitionContextTurns(splitBcgTurns(rest), this.recentTurns);
			const unsent = this.serializePending(evicted.flat());
			const payloads = unsent.flatMap((group) => (group.payload ? [group.payload] : []));

			if (payloads.length > 0) {
				const snapshot = await this.postTurns(payloads, signal);
				this.updateSnapshot(snapshot);
			}
			for (const group of unsent) {
				for (const message of group.messages) this.sentMessages.add(message);
			}

			if (this.graphView === "compact" && this.graphSelection !== "ranked" && this.latestSnapshot) {
				const query = compactSelectionQuery(initialUser, retained.flat());
				const question = contextMessageText(initialUser);
				const focusQuery = compactFocusSelectionQuery(initialUser, retained.flat());
				try {
					const selection = await this.client.selectContext(
						query,
						{
							strategy: this.graphSelection,
							focusQuery,
							question,
						},
						signal,
					);
					this.graphText = formatCompactBcgDialogueContext(
						this.latestSnapshot,
						this.includeRelations,
						new Set(selection.node_ids),
						new Set(selection.relation_ids),
						this.graphSelection === "focused",
					);
					this.emitGraphTrace(this.latestSnapshot, selection);
				} catch (error) {
					const detail = error instanceof Error ? error.message : String(error);
					this.onWarning(`[BCG context selection] ${detail}; using ranked compact selection.`);
					this.emitGraphTrace(this.latestSnapshot);
				}
			}

			this.warned = false;
			this.requestReady = true;
			return [initialUser, ...retained.flat()];
		} catch (error) {
			if (error instanceof BcgTurnLimitError) {
				throw error;
			}
			if (!this.warned) {
				const detail = error instanceof Error ? error.message : String(error);
				this.onWarning(`[BCG context] ${detail}; using the complete raw context for this request.`);
				this.warned = true;
			}
			return messages;
		}
	}

	augmentSystemPrompt(systemPrompt: string | undefined): string | undefined {
		if (!this.requestReady || !this.graphText) {
			return systemPrompt;
		}
		const guide =
			this.graphView === "compact" ? COMPACT_GRAPH_DIALOGUE_CONTEXT_GUIDE : GRAPH_DIALOGUE_CONTEXT_GUIDE;
		return [systemPrompt, guide, this.graphText].filter(Boolean).join("\n\n");
	}

	async release(messages: AgentMessage[] = []): Promise<Record<string, unknown> | undefined> {
		if (!this.seeded || this.released) {
			return undefined;
		}
		this.released = true;
		// Only report construction work that could affect an Agent request. The
		// final unsent-message ingest below exists to complete the persisted Graph
		// (for example, by recording the final decision), but the Agent never sees
		// that update. Its model usage therefore must not be charged to benchmark
		// graph_usage totals.
		const tokenUsage = this.reportableTokenUsage;
		if (messages.length > 0) {
			try {
				const initialUser = this.resolveInitialUser(messages);
				const initialKey = initialUser ? contextMessageKey(initialUser) : undefined;
				let removedInitial = false;
				const pending = messages
					.filter((message) => {
						if (!removedInitial && initialKey !== undefined && contextMessageKey(message) === initialKey) {
							removedInitial = true;
							return false;
						}
						return true;
					});
				const unsent = this.serializePending(pending);
				const payloads = unsent.flatMap((group) => (group.payload ? [group.payload] : []));
				if (payloads.length > 0) {
					const snapshot = await this.postTurns(payloads, undefined, this.finalizationTimeoutMs);
					this.updateSnapshot(snapshot, false);
				}
				for (const group of unsent) {
					for (const message of group.messages) this.sentMessages.add(message);
				}
			} catch (error) {
				const detail = error instanceof Error ? error.message : String(error);
				this.onWarning(`[BCG finalization] failed to ingest final unsent messages: ${detail}`);
			}
		}
		try {
			const snapshot = await this.client.finalize(this.finalizationTimeoutMs);
			this.updateSnapshot(snapshot, false);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			this.onWarning(`[BCG finalization] failed to finalize Graph session: ${detail}`);
		}
		try {
			await this.client.release(this.finalizationTimeoutMs);
		} catch (error) {
			const detail = error instanceof Error ? error.message : String(error);
			this.onWarning(`[BCG finalization] failed to release Graph session: ${detail}`);
		}
		return tokenUsage;
	}

	private resolveInitialUser(messages: AgentMessage[]): AgentMessage | undefined {
		if (this.initialUserMessage) {
			return this.initialUserMessage;
		}
		const fromSession = this.getInitialUserMessage?.();
		if (fromSession?.role === "user") {
			this.initialUserMessage = fromSession;
			return fromSession;
		}
		const fromContext = messages.find((message) => message.role === "user");
		if (fromContext) {
			this.initialUserMessage = fromContext;
		}
		return this.initialUserMessage;
	}

	private serialize(message: AgentMessage): SerializedMessage {
		const role = graphRole(message);
		const content = contextMessageText(message);
		return {
			messages: [message],
			payload: role && content ? this.turnPayload(role, content) : undefined,
		};
	}

	private serializePending(messages: AgentMessage[]): SerializedMessage[] {
		const pending = messages.filter((message) => !this.sentMessages.has(message));
		const serialized: SerializedMessage[] = [];
		for (let index = 0; index < pending.length; index += 1) {
			const message = pending[index];
			if (message.role !== "toolResult") {
				serialized.push(this.serialize(message));
				continue;
			}
			const group: AgentMessage[] = [message];
			while (index + 1 < pending.length && pending[index + 1].role === "toolResult") {
				group.push(pending[index + 1]);
				index += 1;
			}
			serialized.push({
				messages: group,
				payload: this.turnPayload("tool", contextToolResultsToText(group)),
			});
		}
		return serialized;
	}

	private turnPayload(role: BcgTurnPayload["role"], content: string): BcgTurnPayload {
		return {
			problem_id: this.problemId,
			role,
			content,
			is_message_end: true,
			is_trajectory_end: false,
		};
	}

	private async postTurns(
		payloads: BcgTurnPayload[],
		signal?: AbortSignal,
		timeoutMs = this.timeoutMs,
	): Promise<BcgSnapshot> {
		if (this.submittedTurns + payloads.length > this.maxTurns) {
			throw new BcgTurnLimitError(this.submittedTurns, this.maxTurns);
		}
		const snapshot = await this.client.postTurns(payloads, signal, timeoutMs);
		this.submittedTurns += payloads.length;
		return snapshot;
	}

	private updateSnapshot(snapshot: BcgSnapshot, recordTokenUsage = true): void {
		this.latestSnapshot = snapshot;
		if (recordTokenUsage && snapshot.token_usage) {
			this.reportableTokenUsage = snapshot.token_usage;
		}
		this.graphText =
			this.graphView === "compact"
				? formatCompactBcgDialogueContext(snapshot, this.includeRelations)
				: formatBcgDialogueContext(snapshot, this.includeRelations);
		if (this.graphView !== "compact" || this.graphSelection === "ranked") {
			this.emitGraphTrace(snapshot);
		}
	}

	private emitGraphTrace(
		snapshot: BcgSnapshot,
		selection?: BcgContextSelectionResponse,
	): void {
		this.onGraphContext?.({
			view: this.graphView,
			streamTurnIndex: snapshot.stream_turn_index,
			nTurnsIngested: snapshot.n_turns_ingested,
			nNodes: snapshot.n_nodes,
			nRelations: snapshot.relations?.length ?? 0,
			chars: this.graphText.length,
			text: this.graphText,
			...(this.graphView === "compact"
				? { selectionStrategy: selection?.strategy ?? "ranked" }
				: {}),
			...(selection
				? {
						selectionRetrieval: selection.retrieval,
						selectedNodeIds: selection.node_ids,
						selectedRelationIds: selection.relation_ids,
					}
				: {}),
		});
	}
}
