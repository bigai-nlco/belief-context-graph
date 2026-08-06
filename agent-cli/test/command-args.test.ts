import { describe, expect, it } from "vitest";
import { getCommandNameArgument, getPathCommandArgument } from "../src/modes/interactive/command-args.ts";

describe("command-args (step 11.3)", () => {
	it("returns undefined for bare commands", () => {
		expect(getPathCommandArgument("/export", "/export")).toBeUndefined();
		expect(getPathCommandArgument("/import", "/import")).toBeUndefined();
	});

	it("returns undefined without the command prefix", () => {
		expect(getPathCommandArgument("export /a/b.jsonl", "/export")).toBeUndefined();
	});

	it("parses plain paths up to the first whitespace", () => {
		expect(getPathCommandArgument("/export /a/b.jsonl", "/export")).toBe("/a/b.jsonl");
		expect(getPathCommandArgument("/export /a/b.jsonl extra", "/export")).toBe("/a/b.jsonl");
	});

	it("parses quoted paths with spaces", () => {
		expect(getPathCommandArgument('/export "/a/my session.jsonl"', "/export")).toBe("/a/my session.jsonl");
		expect(getPathCommandArgument("/import '/a/single.jsonl'", "/import")).toBe("/a/single.jsonl");
	});

	it("rejects unterminated quotes", () => {
		expect(getPathCommandArgument('/export "/a/b.jsonl', "/export")).toBeUndefined();
	});

	it("trims empty argument tails", () => {
		expect(getPathCommandArgument("/import   ", "/import")).toBeUndefined();
	});

	it("extracts /name arguments", () => {
		expect(getCommandNameArgument("/name my session", "/name")).toBe("my session");
		expect(getCommandNameArgument("/name", "/name")).toBe("");
	});
});
