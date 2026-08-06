import { describe, expect, it } from "vitest";
import * as agentCore from "@bigai-nlco/bcg-agent-core";

// Public export contract for @bigai-nlco/bcg-agent-core (step 11).
// This package is independently publishable: it has no dependency on the
// agent-cli shell, only on bcg-ai. Key symbols below are load-bearing for
// consumers (agent-cli and external harness users).
describe("agent-core public exports", () => {
	it("exposes the core agent loop surface", () => {
		expect(typeof agentCore.Agent).toBe("function");
		expect(typeof agentCore.agentLoop).toBe("function");
		expect(typeof agentCore.agentLoopContinue).toBe("function");
	});

	it("exposes harness/session storage primitives", () => {
		expect(typeof agentCore.JsonlSessionStorage).toBe("function");
		expect(typeof agentCore.Session).toBe("function");
		expect(typeof agentCore.buildSessionContext).toBe("function");
	});
});
