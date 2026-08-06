import { beforeAll, describe, expect, it, vi } from "vitest";

const constructed = vi.hoisted(() => [] as Array<{ name: string; args: unknown[] }>);

vi.mock("../src/modes/interactive/components/extension-selector.ts", () => ({
	ExtensionSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "extension", args });
		}
	},
}));

vi.mock("../src/modes/interactive/components/oauth-selector.ts", () => ({
	OAuthSelectorComponent: class {
		constructor(...args: unknown[]) {
			constructed.push({ name: "oauth", args });
		}
	},
	formatAuthSelectorProviderType: (t: string) => t,
}));

import {
	showLoginAuthTypeSelector,
	showLoginProviderSelector,
	showOAuthSelector,
	type AuthSelectorsDeps,
} from "../src/modes/interactive/auth-selectors.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function fakeDeps(overrides: Partial<AuthSelectorsDeps> = {}): AuthSelectorsDeps & {
	status: string[];
	errors: string[];
	started: string[];
	shownSelectors: number;
	loggedOut: string[];
} {
	const result: AuthSelectorsDeps & {
		status: string[];
		errors: string[];
		started: string[];
		shownSelectors: number;
		loggedOut: string[];
	} = {
		status: [],
		errors: [],
		started: [],
		shownSelectors: 0,
		loggedOut: [],
		showStatus: (message) => result.status.push(message),
		showError: (message) => result.errors.push(message),
		requestRender: () => {},
		showSelector: (create) => {
			result.shownSelectors += 1;
			create(() => {});
		},
		getLoginProviderOptions: () => [],
		getLogoutProviderOptions: async () => [],
		startProviderLogin: (option) => {
			result.started.push(`${option.id}:${option.authType}`);
		},
		logout: async (providerId) => {
			result.loggedOut.push(providerId);
		},
		...overrides,
	};
	return result;
}

beforeAll(() => {
	initTheme("default", false);
});

describe("auth-selectors (step 11.3)", () => {
	it("shows a status when a filtered provider list has no options", () => {
		const deps = fakeDeps();
		showLoginAuthTypeSelector(deps, []);
		expect(deps.status).toEqual(["No login methods available."]);
	});

	it("logs in directly when a single provider has a single auth type", () => {
		const deps = fakeDeps();
		showLoginAuthTypeSelector(deps, [{ id: "openai", name: "OpenAI", authType: "api_key" }]);
		expect(deps.started).toEqual(["openai:api_key"]);
	});

	it("shows the auth-type selector when no providerOptions are given", () => {
		const deps = fakeDeps();
		showLoginAuthTypeSelector(deps);
		expect(deps.shownSelectors).toBe(1);
		expect(constructed.at(-1)?.name).toBe("extension");
		expect(constructed.at(-1)?.args[0]).toContain("Select authentication method:");
	});

	it("reports an empty provider list from showLoginProviderSelector", () => {
		const deps = fakeDeps({ getLoginProviderOptions: () => [] });
		showLoginProviderSelector(deps, "oauth");
		expect(deps.status).toEqual(["No subscription providers available."]);
	});

	it("builds the oauth provider selector for login", () => {
		const deps = fakeDeps({
			getLoginProviderOptions: () => [{ id: "openai", name: "OpenAI", authType: "api_key" }],
		});
		showLoginProviderSelector(deps, "api_key");
		const last = constructed.at(-1)!;
		expect(last.name).toBe("oauth");
		expect(last.args[0]).toBe("login");
		expect((last.args[1] as Array<{ id: string }>)[0]!.id).toBe("openai");
	});

	it("logout flow reports status and handles errors when the callback runs", async () => {
		const deps = fakeDeps({
			getLogoutProviderOptions: async () => [
				{ id: "anthropic", name: "Anthropic", authType: "oauth" },
			],
			showSelector: (create) => {
				create(() => {});
				// simulate the user selecting the provider
				const selector = constructed.at(-1)!;
				void (selector.args[2] as (providerId: string) => Promise<void>)("anthropic");
			},
			logout: async (providerId) => {
				deps.loggedOut.push(providerId);
			},
		});
		await showOAuthSelector(deps, "logout");
		expect(deps.loggedOut).toEqual(["anthropic"]);
		expect(deps.status).toContain("Logged out of Anthropic");
	});

	it("surfaces logout errors", async () => {
		const deps = fakeDeps({
			getLogoutProviderOptions: async () => [
				{ id: "openai", name: "OpenAI", authType: "api_key" },
			],
			showSelector: (create) => {
				create(() => {});
				const selector = constructed.at(-1)!;
				void (selector.args[2] as (providerId: string) => Promise<void>)("openai");
			},
			logout: async () => {
				throw new Error("boom");
			},
		});
		await showOAuthSelector(deps, "logout");
		expect(deps.errors[0]).toContain("Logout failed: boom");
	});
});
