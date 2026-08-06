import type { Component } from "@bigai-nlco/bcg-tui";
import type { Model } from "@bigai-nlco/bcg-ai/compat";
import type { ModelRuntime } from "../../core/model-runtime.ts";
import { resolveModelScope, resolveModelScopeWithDiagnostics } from "../../core/model-resolver.ts";
import type { ThinkingLevel } from "@bigai-nlco/bcg-agent-core";
import type { SettingsManager } from "../../core/settings-manager.ts";
import type { TUI } from "@bigai-nlco/bcg-tui";
import { ModelSelectorComponent } from "./components/model-selector.ts";
import { ScopedModelsSelectorComponent } from "./components/scoped-models-selector.ts";

/**
 * Model selector flows (step 11.3): the /model and /models pickers extracted
 * from InteractiveMode. UI wiring and model/settings backends are injected.
 */
export interface ScopedModelItem {
	model: Model<any>;
	thinkingLevel?: ThinkingLevel;
}

export interface ModelSelectorDeps {
	ui: TUI;
	settingsManager: SettingsManager;
	modelRuntime: ModelRuntime;
	getCurrentModel(): Model<any> | undefined;
	scopedModels: ReadonlyArray<ScopedModelItem>;
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	showStatus(message: string): void;
	showError(message: string): void;
	requestRender(): void;
	setModel(model: Model<any>): Promise<void>;
	/** Session-side post-set effects (footer, editor border, subscription warning). */
	onModelSetSuccess(model: Model<any>): void;
	refreshProviderCount(): Promise<void>;
	setScopedModels(models: Array<{ model: Model<any>; thinkingLevel?: ThinkingLevel }>): void;
}

export function showModelSelector(deps: ModelSelectorDeps, initialSearchInput?: string): void {
	deps.showSelector((done) => {
		const selector = new ModelSelectorComponent(
			deps.ui,
			deps.getCurrentModel(),
			deps.settingsManager,
			deps.modelRuntime,
			deps.scopedModels,
			async (model) => {
				try {
					await deps.setModel(model);
					deps.onModelSetSuccess(model);
					done();
					deps.showStatus(`Model: ${model.id}`);
				} catch (error) {
					done();
					deps.showError(error instanceof Error ? error.message : String(error));
				}
			},
			() => {
				done();
				deps.requestRender();
			},
			initialSearchInput,
		);
		return { component: selector, focus: selector };
	});
}

export async function showModelsSelector(deps: ModelSelectorDeps): Promise<void> {
	// Get all available models
	await deps.modelRuntime.refresh();
	const allModels = [...(await deps.modelRuntime.getAvailable())];
	const allModelIds = new Set(allModels.map((model) => `${model.provider}/${model.id}`));
	const configuredPatterns = deps.settingsManager.getEnabledModels();
	const sessionScopedModels = deps.scopedModels;

	if (allModels.length === 0 && !configuredPatterns?.length && sessionScopedModels.length === 0) {
		deps.showStatus("No models available");
		return;
	}

	const configuredScope = configuredPatterns?.length
		? await resolveModelScopeWithDiagnostics(configuredPatterns, deps.modelRuntime)
		: undefined;

	// Check if session has scoped models (from previous session-only changes or CLI --models)
	const hasSessionScope = sessionScopedModels.length > 0;

	// Build enabled model IDs from session state or settings
	let currentEnabledIds: string[] | null = null;

	if (hasSessionScope) {
		// Use current session's scoped models
		currentEnabledIds = sessionScopedModels.map((scoped) => `${scoped.model.provider}/${scoped.model.id}`);
	} else if (configuredScope) {
		currentEnabledIds = configuredScope.scopedModels.map(
			(scoped) => `${scoped.model.provider}/${scoped.model.id}`,
		);
	}

	for (const diagnostic of configuredScope?.diagnostics ?? []) {
		if (diagnostic.code !== "no-match") continue;
		currentEnabledIds ??= [];
		if (!currentEnabledIds.includes(diagnostic.pattern)) currentEnabledIds.push(diagnostic.pattern);
	}

	// Helper to update session's scoped models (session-only, no persist)
	const updateSessionModels = async (enabledIds: string[] | null) => {
		currentEnabledIds = enabledIds === null ? null : [...enabledIds];
		const hasEnabledAvailableModel = enabledIds?.some((id) => allModelIds.has(id)) ?? false;
		const allAvailableModelsEnabled =
			enabledIds !== null && [...allModelIds].every((id) => enabledIds.includes(id));
		if (enabledIds && hasEnabledAvailableModel && !allAvailableModelsEnabled) {
			const newScopedModels = await resolveModelScope(enabledIds, deps.modelRuntime);
			deps.setScopedModels(
				newScopedModels.map((sm) => ({
					model: sm.model,
					thinkingLevel: sm.thinkingLevel,
				})),
			);
		} else {
			// All enabled or none enabled = no filter
			deps.setScopedModels([]);
		}
		await deps.refreshProviderCount();
		deps.requestRender();
	};

	deps.showSelector((done) => {
		const selector = new ScopedModelsSelectorComponent(
			{
				allModels,
				enabledModelIds: currentEnabledIds,
			},
			{
				onChange: async (enabledIds) => {
					await updateSessionModels(enabledIds);
				},
				onPersist: (enabledIds) => {
					// Persist to settings
					const allEnabled =
						enabledIds !== null &&
						enabledIds.length === allModels.length &&
						enabledIds.every((id) => allModelIds.has(id));
					const newPatterns = enabledIds === null || allEnabled ? undefined : enabledIds;
					deps.settingsManager.setEnabledModels(newPatterns ? [...newPatterns] : undefined);
					deps.showStatus("Model selection saved to settings");
				},
				onCancel: () => {
					done();
					deps.requestRender();
				},
			},
		);
		return { component: selector, focus: selector };
	});
}
