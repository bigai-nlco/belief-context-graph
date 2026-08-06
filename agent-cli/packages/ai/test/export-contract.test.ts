import { describe, expect, it } from "vitest";
import * as ai from "@bigai-nlco/bcg-ai";
import * as compat from "@bigai-nlco/bcg-ai/compat";

// Public export contract for @bigai-nlco/bcg-ai (step 11).
// Independently publishable: no dependency on agent-core, tui or the shell.
describe("bcg-ai public exports", () => {
	it("exposes the side-effect-free core surface", () => {
		expect(ai.Type).toBeDefined();
		expect(typeof ai.contentText).toBe("function");
		expect(typeof ai.createModels).toBe("function");
	});

	it("exposes the compat entry with lazy per-API factories", () => {
		expect(typeof compat.anthropicMessagesApi).toBe("function");
		expect(typeof compat.openAICompletionsApi).toBe("function");
	});

	it("does not leak generated catalogs into the root entry", () => {
		expect("imageModels" in ai).toBe(false);
	});
});
