import type { Component } from "@bigai-nlco/bcg-tui";
import type { ProjectTrustStoreEntry, ProjectTrustUpdate } from "../../core/trust-manager.ts";
import { TrustSelectorComponent } from "./components/trust-selector.ts";
import { UserMessageSelectorComponent } from "./components/user-message-selector.ts";

/**
 * Trust and fork-message selector flows (step 11.3): extracted from
 * InteractiveMode. Backends (trust store, fork) are injected.
 */

export interface TrustSelectorDeps {
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	requestRender(): void;
	showStatus(message: string): void;
	getCwd(): string;
	getSavedDecision(cwd: string): ProjectTrustStoreEntry | null;
	isProjectTrusted(): boolean;
	saveTrust(updates: ProjectTrustUpdate[]): void;
}

export function showTrustSelector(deps: TrustSelectorDeps): void {
	const cwd = deps.getCwd();
	const savedDecision = deps.getSavedDecision(cwd);
	deps.showSelector((done) => {
		const selector = new TrustSelectorComponent({
			cwd,
			savedDecision,
			projectTrusted: deps.isProjectTrusted(),
			onSelect: (selection) => {
				deps.saveTrust(selection.updates);
				done();
				deps.showStatus(
					`Saved trust decision: ${selection.trusted ? "trusted" : "untrusted"}. Restart BCG for this to take effect.`,
				);
			},
			onCancel: () => {
				done();
				deps.requestRender();
			},
		});
		return { component: selector, focus: selector };
	});
}

export interface UserMessageSelectorDeps {
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	requestRender(): void;
	showStatus(message: string): void;
	showError(message: string): void;
	getUserMessagesForForking(): Array<{ entryId: string; text: string }>;
	fork(entryId: string): Promise<{ cancelled: boolean; selectedText?: string }>;
	setEditorText(text: string): void;
}

export function showUserMessageSelector(deps: UserMessageSelectorDeps): void {
	const userMessages = deps.getUserMessagesForForking();

	if (userMessages.length === 0) {
		deps.showStatus("No messages to fork from");
		return;
	}

	const initialSelectedId = userMessages[userMessages.length - 1]?.entryId;

	deps.showSelector((done) => {
		const selector = new UserMessageSelectorComponent(
			userMessages.map((m) => ({ id: m.entryId, text: m.text })),
			async (entryId) => {
				done();
				try {
					const result = await deps.fork(entryId);
					if (result.cancelled) {
						deps.requestRender();
						return;
					}

					deps.setEditorText(result.selectedText ?? "");
					deps.showStatus("Forked to new session");
				} catch (error: unknown) {
					deps.showError(error instanceof Error ? error.message : String(error));
				}
			},
			() => {
				done();
				deps.requestRender();
			},
			initialSelectedId,
		);
		return { component: selector, focus: selector.getMessageList() };
	});
}
