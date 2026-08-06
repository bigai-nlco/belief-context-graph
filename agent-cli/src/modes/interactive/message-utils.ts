import type { Message } from "@bigai-nlco/bcg-ai/compat";

/**
 * Message conversion utilities (step 11.3): extracted from InteractiveMode.
 */

export function getUserMessageText(message: Message): string {
	if (message.role !== "user") return "";
	const textBlocks =
		typeof message.content === "string"
			? [{ type: "text", text: message.content }]
			: message.content.filter((c: { type: string }) => c.type === "text");
	return textBlocks.map((c) => (c as { text: string }).text).join("");
}
