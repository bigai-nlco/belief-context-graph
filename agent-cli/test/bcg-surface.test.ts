import { describe, expect, it } from "vitest";
import { APP_NAME, CONFIG_DIR_NAME, PACKAGE_NAME } from "../src/config.ts";
import { shouldIncludeBuiltinProvider } from "../src/core/model-runtime.ts";
import { BUILTIN_SLASH_COMMANDS } from "../src/core/slash-commands.ts";

describe("BCG terminal surface", () => {
	it("exposes only the compact BCG command set", () => {
		expect(BUILTIN_SLASH_COMMANDS.map((command) => command.name)).toEqual([
			"help",
			"model",
			"mode",
			"login",
			"logout",
			"new",
			"resume",
			"graph",
			"exit",
		]);
	});

	it("uses BCG package and configuration names", () => {
		expect(PACKAGE_NAME).toBe("@bigai-nlco/bcg-agent");
		expect(APP_NAME).toBe("bcg");
		expect(CONFIG_DIR_NAME).toBe(".bcg");
	});

	it("hides official OpenAI models when a custom OpenAI base URL is configured", () => {
		const customEndpoint = { OPENAI_BASE_URL: "https://example.test/v1" };

		expect(shouldIncludeBuiltinProvider("openai", customEndpoint)).toBe(false);
		expect(shouldIncludeBuiltinProvider("openai-codex", customEndpoint)).toBe(true);
		expect(shouldIncludeBuiltinProvider("anthropic", customEndpoint)).toBe(true);
		expect(shouldIncludeBuiltinProvider("openai", {})).toBe(true);
	});
});
