import { describe, expect, it } from "vitest";
import {
	getAutocompleteSourceTag,
	getBuiltInCommandConflictDiagnostics,
	prefixAutocompleteDescription,
} from "../src/modes/interactive/autocomplete-source.ts";
import type { SourceInfo } from "../src/core/source-info.ts";

function source(overrides: Partial<SourceInfo>): SourceInfo {
	return { source: "local", scope: "project", ...overrides } as SourceInfo;
}

describe("autocomplete-source (step 11.3)", () => {
	describe("getAutocompleteSourceTag", () => {
		it("returns the scope prefix for local/auto/cli sources", () => {
			expect(getAutocompleteSourceTag(source({ scope: "user" }))).toBe("u");
			expect(getAutocompleteSourceTag(source({ scope: "project" }))).toBe("p");
			expect(getAutocompleteSourceTag(source({ scope: "temporary" }))).toBe("t");
			expect(getAutocompleteSourceTag(source({ source: "auto" }))).toBe("p");
		});

		it("tags npm sources", () => {
			expect(getAutocompleteSourceTag(source({ source: "npm:@scope/pkg" }))).toBe(
				"p:npm:@scope/pkg",
			);
		});

		it("tags git sources with ref", () => {
			expect(
				getAutocompleteSourceTag(source({ source: "git:github.com/acme/repo@main" })),
			).toContain("git:github.com/acme/repo@main");
		});

		it("returns undefined without source info", () => {
			expect(getAutocompleteSourceTag(undefined)).toBeUndefined();
		});
	});

	describe("prefixAutocompleteDescription", () => {
		it("prefixes descriptions with the tag", () => {
			expect(prefixAutocompleteDescription("run tests", source({ scope: "user" }))).toBe(
				"[u] run tests",
			);
		});

		it("returns the description unchanged without source info", () => {
			expect(prefixAutocompleteDescription("run tests", undefined)).toBe("run tests");
		});
	});

	describe("getBuiltInCommandConflictDiagnostics", () => {
		it("flags extension commands shadowing built-ins", () => {
			const diagnostics = getBuiltInCommandConflictDiagnostics(
				{
					getRegisteredCommands: () => [
						{ name: "help", invocationName: "help", sourceInfo: { path: "/ext/help.ts" } },
						{ name: "custom", invocationName: "custom", sourceInfo: { path: "/ext/custom.ts" } },
					],
				},
				[{ name: "help" }],
			);
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0]!.path).toBe("/ext/help.ts");
			expect(diagnostics[0]!.message).toContain("conflicts with built-in interactive");
		});
	});
});
