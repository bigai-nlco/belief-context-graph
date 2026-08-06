import { readFileSync } from "node:fs";
import { stripFrontmatter } from "../utils/frontmatter.ts";

/**
 * Skill-command text expansion (step 11): extracted from AgentSession so the
 * expansion logic is unit-testable without a session instance. The session
 * injects its skill lookup and error reporting.
 */
export interface SkillInfo {
	name: string;
	filePath: string;
	baseDir: string;
}

export interface SkillExpansionDeps {
	getSkills: () => SkillInfo[];
	emitError: (params: {
		extensionPath: string;
		event: string;
		error: string;
	}) => void;
}

const SKILL_PREFIX = "/skill:";

/**
 * Expand a `/skill:<name> [args]` command into a skill block plus arguments.
 * Unknown skills and unreadable files pass the original text through
 * (unreadable files additionally report an error through `emitError`).
 */
export function expandSkillCommand(text: string, deps: SkillExpansionDeps): string {
	if (!text.startsWith(SKILL_PREFIX)) {
		return text;
	}

	const spaceIndex = text.indexOf(" ");
	const skillName =
		spaceIndex === -1 ? text.slice(SKILL_PREFIX.length) : text.slice(SKILL_PREFIX.length, spaceIndex);
	const args = spaceIndex === -1 ? "" : text.slice(spaceIndex + 1).trim();

	const skill = deps.getSkills().find((s) => s.name === skillName);
	if (!skill) {
		return text; // Unknown skill, pass through
	}

	try {
		const content = readFileSync(skill.filePath, "utf-8");
		const body = stripFrontmatter(content).trim();
		const skillBlock = `<skill name="${skill.name}" location="${skill.filePath}">\nReferences are relative to ${skill.baseDir}.\n\n${body}\n</skill>`;
		return args ? `${skillBlock}\n\n${args}` : skillBlock;
	} catch (err) {
		deps.emitError({
			extensionPath: skill.filePath,
			event: "skill_expansion",
			error: err instanceof Error ? err.message : String(err),
		});
		return text; // Return original on error
	}
}
