import { beforeAll, describe, expect, it } from "vitest";
import {
	buildResourceSections,
	type LoadedResourcesData,
} from "../src/modes/interactive/resources-sections.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function emptyData(): LoadedResourcesData {
	return {
		contextFiles: [],
		skills: [],
		skillDiagnostics: [],
		promptDiagnostics: [],
		templates: [],
		extensions: [],
		extensionDiagnostics: [],
		themes: [],
		themeDiagnostics: [],
		sourceInfos: new Map(),
	};
}

describe("resources-sections (step 11.3)", () => {
	beforeAll(() => {
		initTheme("default", false);
	});

	it("builds no sections from empty data", () => {
		const { sections, diagnostics } = buildResourceSections(emptyData(), {
			expansion: false,
			includeDiagnostics: true,
		});
		expect(sections).toHaveLength(0);
		expect(diagnostics).toHaveLength(0);
	});

	it("assembles a context section with compact and expanded bodies", () => {
		const data = emptyData();
		data.contextFiles = [{ path: "/proj/a.md" }, { path: "/proj/b.md" }];
		const { sections } = buildResourceSections(data, {
			expansion: false,
			includeDiagnostics: false,
		});
		expect(sections).toHaveLength(1);
		const context = sections[0]!;
		expect(context.name).toBe("Context");
		expect(context.collapsed).toContain("/proj/a.md");
		expect(context.expanded).toContain("a.md");
	});

	it("assembles skills, extensions and themes sections with grouped bodies", () => {
		const data = emptyData();
		data.skills = [
			{ name: "summarize", filePath: "/proj/skills/summarize.md", sourceInfo: undefined },
		];
		data.extensions = [{ path: "/proj/ext/one.ts", sourceInfo: undefined }];
		data.themes = [
			{ name: "custom", sourcePath: "/proj/themes/custom.json", sourceInfo: undefined },
		];
		const { sections } = buildResourceSections(data, {
			expansion: true,
			includeDiagnostics: false,
		});
		const names = sections.map((s) => s.name);
		expect(names).toEqual(expect.arrayContaining(["Skills", "Extensions", "Themes"]));
		const skills = sections.find((s) => s.name === "Skills")!;
		expect(skills.collapsed).toContain("summarize");
		expect(skills.expanded).toContain("summarize.md");
	});

	it("emits diagnostic blocks only when included and non-empty", () => {
		const data = emptyData();
		data.skillDiagnostics = [
			{
				type: "collision",
				message: "dup",
				collision: { name: "x", winnerPath: "/a.ts", loserPath: "/b.ts" },
			} as never,
		];
		const without = buildResourceSections(data, {
			expansion: false,
			includeDiagnostics: false,
		});
		expect(without.diagnostics).toHaveLength(0);

		const withDiag = buildResourceSections(data, {
			expansion: false,
			includeDiagnostics: true,
		});
		expect(withDiag.diagnostics).toHaveLength(1);
		expect(withDiag.diagnostics[0]!.title).toBe("Skill conflicts");
		expect(withDiag.diagnostics[0]!.body).toContain("collision");
	});

	it("renders prompt templates from the templates source", () => {
		const data = emptyData();
		data.templates = [
			{ name: "code-review", filePath: "/proj/prompts/code-review.md", sourceInfo: undefined },
		];
		const { sections } = buildResourceSections(data, {
			expansion: false,
			includeDiagnostics: false,
		});
		expect(sections).toHaveLength(1);
		expect(sections[0]!.name).toBe("Prompts");
		expect(sections[0]!.collapsed).toContain("/code-review");
	});
});
