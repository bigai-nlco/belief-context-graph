import { describe, expect, it } from "vitest";
import { getUserMessageText } from "../src/modes/interactive/message-utils.ts";
import type { Message } from "@bigai-nlco/bcg-ai/compat";

describe("message-utils (step 11.3)", () => {
	it("returns empty text for non-user messages", () => {
		expect(getUserMessageText({ role: "assistant", content: "hi" } as Message)).toBe("");
	});

	it("returns string content verbatim", () => {
		expect(getUserMessageText({ role: "user", content: "hello" } as Message)).toBe("hello");
	});

	it("joins text blocks and skips non-text blocks", () => {
		const message = {
			role: "user",
			content: [
				{ type: "text", text: "a" },
				{ type: "image", image: {} },
				{ type: "text", text: "b" },
			],
		} as Message;
		expect(getUserMessageText(message)).toBe("ab");
	});
});
