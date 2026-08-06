import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);

vi.mock("../src/modes/interactive/components/login-dialog.ts", () => ({
	LoginDialogComponent: class {
		signal = new AbortController().signal;
		showAuth = vi.fn();
		showDeviceCode = vi.fn();
		showWaiting = vi.fn();
		showInfo = vi.fn();
		showProgress = vi.fn();
		showManualInput = vi.fn(() => Promise.resolve(""));
		showPrompt = vi.fn(() => Promise.resolve(""));
		showDetails = vi.fn();
		constructor(...args: unknown[]) {
			constructed.push({ name: "login-dialog", args });
		}
	},
}));

vi.mock("../src/modes/interactive/components/extension-selector.ts", () => ({
	ExtensionSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "extension", args });
		}
	},
}));

import {
	completeProviderAuthentication,
	notifyAuthDialog,
	showApiKeyLoginDialog,
	showAuthPrompt,
	type AuthDialogDeps,
} from "../src/modes/interactive/auth-dialogs.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function fakeDeps(overrides: Partial<AuthDialogDeps> = {}): AuthDialogDeps & {
	status: string[];
	errors: string[];
	set: unknown[];
	swaps: number;
	restores: number;
} {
	const result = {
		status: [] as string[],
		errors: [] as string[],
		set: [] as unknown[],
		swaps: 0,
		restores: 0,
		ui: {} as AuthDialogDeps["ui"],
		modelRuntime: {
			getAvailable: async () => [],
			login: async () => {},
		} as unknown as AuthDialogDeps["modelRuntime"],
		getCurrentModel: () => ({ id: "unknown", provider: "unknown", api: "unknown" }),
		setModel: async (model: unknown) => {
			result.set.push(model);
		},
		showStatus: (m: string) => result.status.push(m),
		showError: (m: string) => result.errors.push(m),
		requestRender: () => {},
		swapEditor: () => {
			result.swaps += 1;
		},
		restoreEditor: () => {
			result.restores += 1;
		},
		refreshProviderCount: async () => {},
		onAuthUiEffects: () => {},
		onAuthWarning: () => {},
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("auth-dialogs (step 11.3)", () => {
	it("dispatches auth events to the dialog", () => {
		const dialog = constructed.length ? (constructed.at(-1)!.args[0] as never) : null;
		// construct a fake dialog directly
		const fake = { showAuth: vi.fn(), showDeviceCode: vi.fn(), showWaiting: vi.fn(), showInfo: vi.fn(), showProgress: vi.fn() };
		notifyAuthDialog(fake as never, { type: "auth_url", url: "https://x", instructions: "open" } as never);
		expect(fake.showAuth).toHaveBeenCalledWith("https://x", "open");
		notifyAuthDialog(fake as never, { type: "device_code" } as never);
		expect(fake.showWaiting).toHaveBeenCalled();
		notifyAuthDialog(fake as never, { type: "info", message: "hi", links: [] } as never);
		expect(fake.showInfo).toHaveBeenCalled();
		notifyAuthDialog(fake as never, { type: "progress", message: "working" } as never);
		expect(fake.showProgress).toHaveBeenCalled();
	});

	it("selects the provider default model after login when the previous model is unknown", async () => {
		const model = { id: "claude-opus-4-8", provider: "anthropic", api: "test" };
		const deps = fakeDeps({
			modelRuntime: {
				getAvailable: async () => [model],
			} as unknown as AuthDialogDeps["modelRuntime"],
		});
		await completeProviderAuthentication(
			deps,
			"anthropic",
			"Anthropic",
			"oauth",
			{ id: "unknown", provider: "unknown", api: "unknown" },
		);
		expect(deps.set).toEqual([model]);
		expect(deps.status[0]).toContain("Logged in to Anthropic");
	});

	it("reports a selection error when the provider has no models", async () => {
		const deps = fakeDeps({
			modelRuntime: {
				getAvailable: async () => [],
			} as unknown as AuthDialogDeps["modelRuntime"],
		});
		await completeProviderAuthentication(
			deps,
			"anthropic",
			"Anthropic",
			"api_key",
			{ id: "unknown", provider: "unknown", api: "unknown" },
		);
		expect(deps.errors[0]).toContain("no models are available");
		expect(deps.status[0]).toContain("Saved API key for Anthropic");
	});

	it("runs the api-key dialog flow through login and completion", async () => {
		const deps = fakeDeps({
			modelRuntime: {
				getAvailable: async () => [],
				login: async (_id, _method, opts: { notify?: (e: unknown) => void }) => {
					opts.notify?.({ type: "progress", message: "waiting" });
				},
			} as unknown as AuthDialogDeps["modelRuntime"],
		});
		await showApiKeyLoginDialog(deps, "openai", "OpenAI");
		expect(deps.swaps).toBeGreaterThan(0);
		expect(deps.restores).toBeGreaterThan(0);
		expect(deps.status[0]).toContain("Saved API key for OpenAI");
	});

	it("rejects aborted auth prompts as Login cancelled", async () => {
		const controller = new AbortController();
		controller.abort();
		const deps = fakeDeps();
		const dialog = { showPrompt: vi.fn(async () => "value"), showManualInput: vi.fn() } as never;
		await expect(
			showAuthPrompt(deps, dialog, { type: "text", message: "m", signal: controller.signal } as never),
		).rejects.toThrow("Login cancelled");
	});
});
