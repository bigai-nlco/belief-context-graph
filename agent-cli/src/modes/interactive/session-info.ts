import type { SessionStats } from "../../core/agent-session.ts";
import { computeCacheWaste } from "../../core/cache-stats.ts";
import type { SessionEntry } from "../../core/session-manager.ts";
import { getUsageCostBreakdown } from "../../core/usage-totals.ts";
import type { ModelRuntime } from "../../core/model-runtime.ts";
import { formatTokens } from "./components/footer.ts";
import { theme } from "./theme/theme.ts";

/**
 * Session-info text assembly (step 11.3): extracted from
 * InteractiveMode.handleSessionCommand. Pure text building over injected
 * session data; the class only renders the resulting block.
 */
export interface SessionInfoDeps {
	getSessionStats(): SessionStats;
	getSessionName(): string | undefined;
	getEntries(): SessionEntry[];
	modelRuntime: ModelRuntime;
}

export function buildSessionInfoText(deps: SessionInfoDeps): string {
	const stats = deps.getSessionStats();
	const sessionName = deps.getSessionName();
	const entries = deps.getEntries();
	const cacheWaste = computeCacheWaste(entries, deps.modelRuntime);

	// Cost/token totals per provider/model actually used (e.g. OpenRouter `auto`
	// resolves to a concrete responseModel). Usage without model attribution is
	// grouped separately so the breakdown reconciles with the session total.
	const usageBreakdown = getUsageCostBreakdown(entries);

	let info = `${theme.bold("Session Info")}\n\n`;
	if (sessionName) {
		info += `${theme.fg("dim", "Name:")} ${sessionName}\n`;
	}
	info += `${theme.fg("dim", "File:")} ${stats.sessionFile ?? "In-memory"}\n`;
	info += `${theme.fg("dim", "ID:")} ${stats.sessionId}\n\n`;
	info += `${theme.bold("Messages")}\n`;
	info += `${theme.fg("dim", "Total:")} ${stats.totalMessages}\n`;
	info += `${theme.fg("dim", "User:")} ${stats.userMessages}\n`;
	info += `${theme.fg("dim", "Assistant:")} ${stats.assistantMessages}\n`;
	info += `${theme.fg("dim", "Tools:")} ${stats.toolCalls} calls, ${stats.toolResults} results\n\n`;
	info += `${theme.bold("Tokens")}\n`;
	// "Input" is the full prompt volume. With cache activity, split it into
	// cached (served from cache) vs uncached (everything else) - the only
	// provider-independent split. Cache writes, where reported, are a detail
	// of the uncached portion.
	const { input, cacheRead, cacheWrite } = stats.tokens;
	const promptTokens = input + cacheRead + cacheWrite;
	info += `${theme.fg("dim", "Input:")} ${promptTokens.toLocaleString()}\n`;
	if (promptTokens > 0 && (cacheRead > 0 || cacheWrite > 0)) {
		const hitRate = theme.fg("dim", `(${((cacheRead / promptTokens) * 100).toFixed(1)}%)`);
		info += `  ${theme.fg("dim", "Cached:")} ${cacheRead.toLocaleString()} ${hitRate}\n`;
		const written =
			cacheWrite > 0 ? ` ${theme.fg("dim", `(${cacheWrite.toLocaleString()} written to cache)`)}` : "";
		info += `  ${theme.fg("dim", "Uncached:")} ${(input + cacheWrite).toLocaleString()}${written}\n`;
	}
	info += `${theme.fg("dim", "Output:")} ${stats.tokens.output.toLocaleString()}\n`;
	info += `${theme.fg("dim", "Total:")} ${stats.tokens.total.toLocaleString()}\n`;

	if (stats.cost > 0 || cacheWaste.missedTokens > 0) {
		info += `\n${theme.bold("Cost")}\n`;
		info += `${theme.fg("dim", "Total:")} $${stats.cost.toFixed(3)}`;
		if (usageBreakdown.length > 1) {
			for (const entry of usageBreakdown) {
				info += `\n  ${theme.fg("dim", `${entry.key}:`)} $${entry.cost.toFixed(3)} ${theme.fg("dim", `(${formatTokens(entry.tokens)} tokens)`)}`;
			}
		}
		if (cacheWaste.missedTokens > 0) {
			const missLabel = cacheWaste.missCount === 1 ? "1 miss" : `${cacheWaste.missCount} misses`;
			const detail = `${cacheWaste.missedTokens.toLocaleString()} tokens, ${missLabel}`;
			info +=
				cacheWaste.missedCost >= 0.0001
					? `\n${theme.fg("dim", "Cache Re-billed:")} $${cacheWaste.missedCost.toFixed(3)} ${theme.fg("dim", `(${detail})`)}`
					: `\n${theme.fg("dim", "Cache Re-billed:")} ${detail}`;
		}
	}

	return info;
}
