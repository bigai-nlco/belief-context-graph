import type { SessionManager } from "../session-manager.ts";
import type { ContextManagementProvider } from "../settings-manager.ts";

export const CONTEXT_MODE_ENTRY_TYPE = "bcg.context_mode";

export function usesBoundedContext(provider: ContextManagementProvider | undefined): boolean {
	return provider === "bcg" || provider === "summary" || provider === "recent-only" || provider === "rag";
}

function isContextManagementProvider(value: unknown): value is ContextManagementProvider {
	return (
		value === "default" ||
		value === "bcg" ||
		value === "summary" ||
		value === "recent-only" ||
		value === "rag"
	);
}

export function getSessionContextMode(sessionManager: SessionManager): ContextManagementProvider | undefined {
	const branch = sessionManager.getBranch();
	for (let index = branch.length - 1; index >= 0; index -= 1) {
		const entry = branch[index];
		if (entry.type !== "custom" || entry.customType !== CONTEXT_MODE_ENTRY_TYPE) {
			continue;
		}
		const data = entry.data;
		if (typeof data === "object" && data !== null && "provider" in data) {
			const provider = (data as { provider?: unknown }).provider;
			if (isContextManagementProvider(provider)) {
				return provider;
			}
		}
	}
	return undefined;
}

export function hasSessionConversationStarted(sessionManager: SessionManager): boolean {
	return sessionManager
		.getBranch()
		.some((entry) => entry.type === "message" && entry.message.role === "user");
}

export function setSessionContextMode(
	sessionManager: SessionManager,
	provider: ContextManagementProvider,
): ContextManagementProvider {
	sessionManager.appendCustomEntry(CONTEXT_MODE_ENTRY_TYPE, { provider });
	return provider;
}

export function ensureSessionContextMode(
	sessionManager: SessionManager,
	fallback: ContextManagementProvider,
): ContextManagementProvider {
	const existing = getSessionContextMode(sessionManager);
	if (existing) {
		return existing;
	}

	// Sessions created before context modes were persisted were BCG-only.
	const provider = hasSessionConversationStarted(sessionManager) ? "bcg" : fallback;
	return setSessionContextMode(sessionManager, provider);
}
