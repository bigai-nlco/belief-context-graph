import type { ResourceDiagnostic } from "../../core/resource-loader.ts";
import type { SourceInfo } from "../../core/source-info.ts";
import { parseGitUrl } from "../../utils/git.ts";

/**
 * Autocomplete source tagging (step 11.3): extracted from InteractiveMode.
 * Pure derivation of source tags for autocomplete descriptions and conflict
 * diagnostics against built-in slash commands.
 */

export function getAutocompleteSourceTag(sourceInfo?: SourceInfo): string | undefined {
	if (!sourceInfo) {
		return undefined;
	}

	const scopePrefix = sourceInfo.scope === "user" ? "u" : sourceInfo.scope === "project" ? "p" : "t";
	const source = sourceInfo.source.trim();

	if (source === "auto" || source === "local" || source === "cli") {
		return scopePrefix;
	}

	if (source.startsWith("npm:")) {
		return `${scopePrefix}:${source}`;
	}

	const gitSource = parseGitUrl(source);
	if (gitSource) {
		const ref = gitSource.ref ? `@${gitSource.ref}` : "";
		return `${scopePrefix}:git:${gitSource.host}/${gitSource.path}${ref}`;
	}

	return scopePrefix;
}

export function prefixAutocompleteDescription(
	description: string | undefined,
	sourceInfo?: SourceInfo,
): string | undefined {
	const sourceTag = getAutocompleteSourceTag(sourceInfo);
	if (!sourceTag) {
		return description;
	}
	return description ? `[${sourceTag}] ${description}` : `[${sourceTag}]`;
}

export interface BuiltinCommandView {
	name: string;
}

export interface ExtensionCommandView {
	name: string;
	invocationName: string;
	sourceInfo: { path: string };
}

export function getBuiltInCommandConflictDiagnostics(
	extensionRunner: {
		getRegisteredCommands(): ReadonlyArray<ExtensionCommandView>;
	},
	builtinCommands: ReadonlyArray<BuiltinCommandView>,
): ResourceDiagnostic[] {
	const builtinNames = new Set(builtinCommands.map((command) => command.name));
	return extensionRunner
		.getRegisteredCommands()
		.filter((command) => builtinNames.has(command.name))
		.map((command) => ({
			type: "warning" as const,
			message:
				command.invocationName === command.name
					? `Extension command '/${command.name}' conflicts with built-in interactive command. Skipping in autocomplete.`
					: `Extension command '/${command.name}' conflicts with built-in interactive command. Available as '/${command.invocationName}'.`,
			path: command.sourceInfo.path,
		}));
}
