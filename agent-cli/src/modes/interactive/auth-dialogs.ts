import type { Component, TUI } from "@bigai-nlco/bcg-tui";
import type { AuthEvent, AuthPrompt } from "@bigai-nlco/bcg-ai";
import type { Model } from "@bigai-nlco/bcg-ai/compat";
import type { ModelRuntime } from "../../core/model-runtime.ts";
import { defaultModelPerProvider } from "../../core/model-resolver.ts";
import { getAuthPath, getReadmePath } from "../../config.ts";
import { ExtensionSelectorComponent } from "./components/extension-selector.ts";
import { LoginDialogComponent } from "./components/login-dialog.ts";
import type { AuthSelectorProvider } from "./components/oauth-selector.ts";
import { theme } from "./theme/theme.ts";

/**
 * Auth dialog flows (step 11.3): login/logout dialogs and the
 * post-authentication model selection, extracted from InteractiveMode.
 */
export interface AuthDialogDeps {
	ui: TUI;
	modelRuntime: ModelRuntime;
	getCurrentModel(): Model<any> | undefined;
	setModel(model: Model<any>): Promise<void>;
	showStatus(message: string): void;
	showError(message: string): void;
	requestRender(): void;
	/** Replace the editor area with a dialog/selector and focus it. */
	swapEditor(component: Component): void;
	/** Restore the editor area. */
	restoreEditor(): void;
	refreshProviderCount(): Promise<void>;
	/** Post-auth UI effects (footer invalidation, editor border). */
	onAuthUiEffects(): void;
	/** Optional subscription warning after a successful login. */
	onAuthWarning(model?: Model<any>): void;
}

function isUnknownModel(model: Model<any> | undefined): boolean {
	return !!model && model.provider === "unknown" && model.id === "unknown" && model.api === "unknown";
}

function hasDefaultModelProvider(providerId: string): providerId is keyof typeof defaultModelPerProvider {
	return providerId in defaultModelPerProvider;
}

export function notifyAuthDialog(dialog: LoginDialogComponent, event: AuthEvent): void {
	if (event.type === "auth_url") {
		dialog.showAuth(event.url, event.instructions);
	} else if (event.type === "device_code") {
		dialog.showDeviceCode(event);
		dialog.showWaiting("Waiting for authentication...");
	} else if (event.type === "info") {
		dialog.showInfo(event.message, event.links);
	} else {
		dialog.showProgress(event.message);
	}
}

export async function completeProviderAuthentication(
	deps: AuthDialogDeps,
	providerId: string,
	providerName: string,
	authType: "oauth" | "api_key",
	previousModel: Model<any> | undefined,
): Promise<void> {
	await deps.modelRuntime.getAvailable();

	const actionLabel = authType === "oauth" ? `Logged in to ${providerName}` : `Saved API key for ${providerName}`;

	let selectedModel: Model<any> | undefined;
	let selectionError: string | undefined;
	if (isUnknownModel(previousModel)) {
		const availableModels = await deps.modelRuntime.getAvailable();
		const providerModels = availableModels.filter((model) => model.provider === providerId);
		if (!hasDefaultModelProvider(providerId)) {
			selectionError = `${actionLabel}, but no default model is configured for provider "${providerId}". Use /model to select a model.`;
		} else if (providerModels.length === 0) {
			selectionError = `${actionLabel}, but no models are available for that provider. Use /model to select a model.`;
		} else {
			const defaultModelId = defaultModelPerProvider[providerId];
			selectedModel = providerModels.find((model) => model.id === defaultModelId);
			if (!selectedModel) {
				selectionError = `${actionLabel}, but its default model "${defaultModelId}" is not available. Use /model to select a model.`;
			} else {
				try {
					await deps.setModel(selectedModel);
				} catch (error: unknown) {
					selectedModel = undefined;
					const errorMessage = error instanceof Error ? error.message : String(error);
					selectionError = `${actionLabel}, but selecting its default model failed: ${errorMessage}. Use /model to select a model.`;
				}
			}
		}
	}

	await deps.refreshProviderCount();
	deps.onAuthUiEffects();
	if (selectedModel) {
		deps.showStatus(`${actionLabel}. Selected ${selectedModel.id}. Credentials saved to ${getAuthPath()}`);
		deps.onAuthWarning(selectedModel);
	} else {
		deps.showStatus(`${actionLabel}. Credentials saved to ${getAuthPath()}`);
		if (selectionError) {
			deps.showError(selectionError);
		} else {
			deps.onAuthWarning();
		}
	}
}

export function showAmbientAuthDialog(deps: AuthDialogDeps, providerOption: AuthSelectorProvider): void {
	const dialog = new LoginDialogComponent(
		deps.ui,
		providerOption.id,
		() => deps.restoreEditor(),
		providerOption.name,
		`${providerOption.name} setup`,
	);
	dialog.showInfo(`${providerOption.method?.name ?? "Authentication"} is configured outside BCG.`, [], true);

	deps.swapEditor(dialog);
}

async function loginProvider(
	deps: AuthDialogDeps,
	dialog: LoginDialogComponent,
	providerId: string,
	method: "api_key" | "oauth",
): Promise<void> {
	await deps.modelRuntime.login(providerId, method, {
		signal: dialog.signal,
		prompt: (prompt) => showAuthPrompt(deps, dialog, prompt),
		notify: (event) => notifyAuthDialog(dialog, event),
	});
}

export async function showApiKeyLoginDialog(
	deps: AuthDialogDeps,
	providerId: string,
	providerName: string,
): Promise<void> {
	const previousModel = deps.getCurrentModel();

	const dialog = new LoginDialogComponent(
		deps.ui,
		providerId,
		(_success, _message) => {
			// Completion handled below
		},
		providerName,
	);

	if (providerId === "amazon-bedrock") {
		dialog.showDetails([
			theme.fg("text", "You can also use an AWS profile, IAM keys, or role-based credentials."),
			theme.fg("muted", "See:"),
			theme.fg("accent", `  ${getReadmePath()}`),
		]);
	}

	deps.swapEditor(dialog);

	try {
		await loginProvider(deps, dialog, providerId, "api_key");
		deps.restoreEditor();
		await completeProviderAuthentication(deps, providerId, providerName, "api_key", previousModel);
	} catch (error: unknown) {
		deps.restoreEditor();
		const errorMsg = error instanceof Error ? error.message : String(error);
		if (errorMsg !== "Login cancelled") {
			deps.showError(`Failed to save API key for ${providerName}: ${errorMsg}`);
		}
	}
}

export async function showLoginDialog(
	deps: AuthDialogDeps,
	providerId: string,
	providerName: string,
): Promise<void> {
	const previousModel = deps.getCurrentModel();
	const dialog = new LoginDialogComponent(deps.ui, providerId, (_success, _message) => {}, providerName);

	deps.swapEditor(dialog);

	try {
		await loginProvider(deps, dialog, providerId, "oauth");
		deps.restoreEditor();
		await completeProviderAuthentication(deps, providerId, providerName, "oauth", previousModel);
	} catch (error: unknown) {
		deps.restoreEditor();
		const errorMsg = error instanceof Error ? error.message : String(error);
		if (errorMsg !== "Login cancelled") {
			deps.showError(`Failed to login to ${providerName}: ${errorMsg}`);
		}
	}
}

function showAuthSelect(
	deps: AuthDialogDeps,
	dialog: LoginDialogComponent,
	prompt: Extract<AuthPrompt, { type: "select" }>,
): Promise<string> {
	return new Promise((resolve, reject) => {
		const restoreDialog = () => {
			deps.swapEditor(dialog);
		};
		const labels = prompt.options.map((option) => option.label);
		const selector = new ExtensionSelectorComponent(
			prompt.message,
			labels,
			(optionLabel) => {
				restoreDialog();
				const id = prompt.options.find((option) => option.label === optionLabel)?.id;
				if (id) resolve(id);
				else reject(new Error("Login cancelled"));
			},
			() => {
				restoreDialog();
				reject(new Error("Login cancelled"));
			},
		);
		deps.swapEditor(selector);
	});
}

export async function showAuthPrompt(
	deps: AuthDialogDeps,
	dialog: LoginDialogComponent,
	prompt: AuthPrompt,
): Promise<string> {
	let response: Promise<string>;
	if (prompt.type === "select") {
		response = showAuthSelect(deps, dialog, prompt);
	} else if (prompt.type === "manual_code") {
		response = dialog.showManualInput(prompt.message);
	} else {
		response = dialog.showPrompt(prompt.message, prompt.placeholder);
	}
	if (!prompt.signal) return response;
	if (prompt.signal.aborted) throw new Error("Login cancelled");
	const signal = prompt.signal;
	let onAbort: (() => void) | undefined;
	const aborted = new Promise<string>((_resolve, reject) => {
		onAbort = () => reject(new Error("Login cancelled"));
		signal.addEventListener("abort", onAbort, { once: true });
	});
	try {
		return await Promise.race([response, aborted]);
	} finally {
		if (onAbort) signal.removeEventListener("abort", onAbort);
	}
}
