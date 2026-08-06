import type { Component } from "@bigai-nlco/bcg-tui";
import type { SessionTreeNode } from "../../core/session-manager.ts";
import { TreeSelectorComponent, type FilterMode } from "./components/tree-selector.ts";

/**
 * Session-tree navigation flow (step 11.3): the /tree picker with branch
 * summarization extracted from InteractiveMode. All session, UI and
 * clipboard backends are injected.
 */

export interface TreeNavigationResult {
	aborted?: boolean;
	cancelled?: boolean;
	editorText?: string;
}

export interface TreeSelectorDeps {
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	requestRender(): void;
	showStatus(message: string): void;
	showError(message: string): void;
	getTree(): SessionTreeNode[];
	getLeafId(): string | null;
	getTerminalRows(): number;
	getTreeFilterMode(): FilterMode;
	getBranchSummarySkipPrompt(): boolean;
	appendLabelChange(entryId: string, label: string | undefined): void;
	/** Returns undefined when the user cancels the choice. */
	promptForSummaryChoice(title: string, options: string[]): Promise<string | undefined>;
	/** Returns undefined when the user cancels custom instructions. */
	promptForCustomInstructions(title: string): Promise<string | undefined>;
	abortBranchSummary(): void;
	navigateTree(entryId: string, options: { summarize: boolean; customInstructions?: string }): Promise<TreeNavigationResult>;
	setEditorTextIfEmpty(text: string): void;
	flushCompactionQueue(): void;
	/**
	 * Install a temporary editor escape handler; the returned function
	 * restores the previous handler.
	 */
	withSummaryEscapeHandler(handler: () => void): () => void;
	showSummaryIndicator(): void;
	clearSummaryIndicator(): void;
	renderInitialMessages(): void;
	clearChat(): void;
	copyToClipboard(text: string): Promise<void>;
}

export function showTreeSelector(deps: TreeSelectorDeps, initialSelectedId?: string): void {
	const tree = deps.getTree();
	const realLeafId = deps.getLeafId();
	const initialFilterMode = deps.getTreeFilterMode();

	if (tree.length === 0) {
		deps.showStatus("No entries in session");
		return;
	}

	deps.showSelector((done) => {
		const selector = new TreeSelectorComponent(
			tree as never,
			realLeafId,
			deps.getTerminalRows(),
			async (entryId) => {
				// Selecting the current leaf is a no-op (already there)
				if (entryId === realLeafId) {
					done();
					deps.showStatus("Already at this point");
					return;
				}

				// Ask about summarization
				done(); // Close selector first

				// Loop until user makes a complete choice or cancels to tree
				let wantsSummary = false;
				let customInstructions: string | undefined;

				// Check if we should skip the prompt (user preference to always default to no summary)
				if (!deps.getBranchSummarySkipPrompt()) {
					while (true) {
						const summaryChoice = await deps.promptForSummaryChoice("Summarize branch?", [
							"No summary",
							"Summarize",
							"Summarize with custom prompt",
						]);

						if (summaryChoice === undefined) {
							// User pressed escape - re-show tree selector with same selection
							showTreeSelector(deps, entryId);
							return;
						}

						wantsSummary = summaryChoice !== "No summary";

						if (summaryChoice === "Summarize with custom prompt") {
							customInstructions = await deps.promptForCustomInstructions(
								"Custom summarization instructions",
							);
							if (customInstructions === undefined) {
								// User cancelled - loop back to summary selector
								continue;
							}
						}

						// User made a complete choice
						break;
					}
				}

				// Set up escape handler and status indicator if summarizing
				let showingSummaryIndicator = false;
				let restoreEscape: (() => void) | undefined;

				if (wantsSummary) {
					const restoreEscape = deps.withSummaryEscapeHandler(() => {
						deps.abortBranchSummary();
					});
					deps.showSummaryIndicator();
					showingSummaryIndicator = true;
					deps.requestRender();
				}

				try {
					const result = await deps.navigateTree(entryId, {
						summarize: wantsSummary,
						customInstructions,
					});

					if (result.aborted) {
						// Summarization aborted - re-show tree selector with same selection
						deps.showStatus("Branch summarization cancelled");
						showTreeSelector(deps, entryId);
						return;
					}
					if (result.cancelled) {
						deps.showStatus("Navigation cancelled");
						return;
					}

					// Update UI
					deps.clearChat();
					deps.renderInitialMessages();
					if (result.editorText) {
						deps.setEditorTextIfEmpty(result.editorText);
					}
					deps.showStatus("Navigated to selected point");
					deps.flushCompactionQueue();
				} catch (error) {
					deps.showError(error instanceof Error ? error.message : String(error));
				} finally {
					if (showingSummaryIndicator) {
						deps.clearSummaryIndicator();
						restoreEscape?.();
					}
				}
			},
			() => {
				done();
				deps.requestRender();
			},
			(entryId, label) => {
				deps.appendLabelChange(entryId, label);
				deps.requestRender();
			},
			initialSelectedId,
			initialFilterMode,
		);
		selector.onCopy = async (text) => {
			if (!text) {
				deps.showError("Selected entry has no text to copy");
				return;
			}
			try {
				await deps.copyToClipboard(text);
				deps.showStatus("Copied selected message to clipboard");
			} catch (error) {
				deps.showError(error instanceof Error ? error.message : String(error));
			}
		};
		return { component: selector, focus: selector };
	});
}
