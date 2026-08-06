import { describe, expect, it } from "vitest";
import {
	buildLoginProviderOptions,
	buildLogoutProviderOptions,
	findLoginProviderOptions,
	type ModelRuntimeAuthView,
} from "../src/modes/interactive/login-provider-options.ts";

function fakeRuntime(overrides: Partial<ModelRuntimeAuthView> = {}): ModelRuntimeAuthView {
	return {
		getProviders: () => [
			{
				id: "anthropic",
				name: "Anthropic",
				auth: {
					oauth: { clientId: "x" },
					apiKey: { env: "ANTHROPIC_API_KEY" },
				},
			},
			{
				id: "openai",
				name: "OpenAI",
				auth: { apiKey: { env: "OPENAI_API_KEY" } },
			},
		],
		getProviderAuthStatus: () => ({ configured: false }),
		isUsingOAuth: () => false,
		getProvider: (id) => ({ id, name: id }),
		listCredentials: async () => [
			{ providerId: "anthropic", type: "oauth" },
			{ providerId: "openai", type: "api_key" },
		],
		...overrides,
	};
}

describe("login-provider-options (step 11.3)", () => {
	it("builds one option per supported auth type, sorted by name", () => {
		const options = buildLoginProviderOptions(fakeRuntime());
		expect(options).toHaveLength(3); // anthropic oauth+api_key, openai api_key
		expect(options[0]!.name).toBe("Anthropic");
		expect(options.map((o) => o.authType).sort()).toEqual(["api_key", "api_key", "oauth"]);
	});

	it("filters by auth type", () => {
		const oauth = buildLoginProviderOptions(fakeRuntime(), "oauth");
		expect(oauth).toHaveLength(1);
		expect(oauth[0]!.id).toBe("anthropic");

		const apiKey = buildLoginProviderOptions(fakeRuntime(), "api_key");
		expect(apiKey).toHaveLength(2);
	});

	it("marks configured providers with their status", () => {
		const runtime = fakeRuntime({
			getProviderAuthStatus: (id) =>
				id === "anthropic"
					? { configured: true, label: "stored", source: "env" }
					: { configured: false },
			isUsingOAuth: (id) => id === "anthropic",
		});
		const options = buildLoginProviderOptions(runtime);
		const anthropicOAuth = options.find((o) => o.id === "anthropic" && o.authType === "oauth")!;
		expect(anthropicOAuth.status).toEqual({ type: "oauth", source: "stored" });
		const openai = options.find((o) => o.id === "openai")!;
		expect(openai.status).toBeUndefined();
	});

	it("findLoginProviderOptions matches by id or name, case-insensitive", () => {
		const runtime = fakeRuntime();
		expect(findLoginProviderOptions(runtime, "ANTHROPIC")).toHaveLength(2);
		expect(findLoginProviderOptions(runtime, "openai")).toHaveLength(1);
		expect(findLoginProviderOptions(runtime, "  ")).toHaveLength(0);
		expect(findLoginProviderOptions(runtime, "nope")).toHaveLength(0);
	});

	it("buildLogoutProviderOptions lists stored credentials", async () => {
		const options = await buildLogoutProviderOptions(fakeRuntime());
		expect(options).toHaveLength(2);
		expect(options[0]!.status?.source).toBe("stored credential");
	});
});
