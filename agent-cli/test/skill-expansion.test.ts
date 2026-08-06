import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	expandSkillCommand,
	type SkillExpansionDeps,
	type SkillInfo,
} from "../src/core/skill-expansion.ts";

const cleanups: string[] = [];
afterEach(() => {
	for (const dir of cleanups.splice(0)) {
		rmSync(dir, { recursive: true, force: true });
	}
});

function skillFile(body: string): { dir: string; filePath: string } {
	const dir = mkdtempSync(path.join(tmpdir(), "bcg-skill-"));
	cleanups.push(dir);
	const filePath = path.join(dir, "skill.md");
	writeFileSync(filePath, body, "utf8");
	return { dir, filePath };
}

function deps(skills: SkillInfo[]): {
	deps: SkillExpansionDeps;
	emitted: Array<{ extensionPath: string; event: string; error: string }>;
} {
	const emitted: Array<{ extensionPath: string; event: string; error: string }> = [];
	return {
		deps: {
			getSkills: () => skills,
			emitError: (params) => emitted.push(params),
		},
		emitted,
	};
}

describe("skill-expansion (step 11)", () => {
	it("passes through non-skill text", () => {
		const { deps: d } = deps([]);
		expect(expandSkillCommand("just some text", d)).toBe("just some text");
	});

	it("expands a known skill into a block", () => {
		const { dir, filePath } = skillFile("---\ntitle: t\n---\n# Body\ncontent here");
		const { deps: d } = deps([{ name: "summarize", filePath, baseDir: dir }]);

		const expanded = expandSkillCommand("/skill:summarize", d);
		expect(expanded).toContain('<skill name="summarize"');
		expect(expanded).toContain("References are relative to");
		expect(expanded).toContain("content here");
		expect(expanded).not.toContain("---");
	});

	it("appends arguments after the skill block", () => {
		const { dir, filePath } = skillFile("# Body\nplain");
		const { deps: d } = deps([{ name: "summarize", filePath, baseDir: dir }]);

		const expanded = expandSkillCommand("/skill:summarize focus on risks", d);
		expect(expanded).toContain("</skill>");
		expect(expanded).toContain("\n\nfocus on risks");
	});

	it("passes unknown skills through", () => {
		const { deps: d } = deps([]);
		expect(expandSkillCommand("/skill:missing thing", d)).toBe("/skill:missing thing");
	});

	it("reports unreadable skill files and passes the text through", () => {
		const { deps: d, emitted } = deps([
			{ name: "broken", filePath: "/nonexistent/skill.md", baseDir: "/tmp" },
		]);

		const result = expandSkillCommand("/skill:broken", d);
		expect(result).toBe("/skill:broken");
		expect(emitted).toHaveLength(1);
		expect(emitted[0].event).toBe("skill_expansion");
	});
});
