import type { ResourceDiagnostic } from "../../core/resource-loader.ts";
import type { SourceInfo } from "../../core/source-info.ts";
import {
	buildScopeGroups,
	formatDiagnostics,
	formatDisplayPath,
	formatExtensionDisplayPath,
	formatScopeGroups,
	getCompactExtensionLabels,
	getCompactPathLabel,
	getShortPath,
} from "./display-format.ts";
import { theme, type ThemeColor } from "./theme/theme.ts";

/**
 * Loaded-resources section assembly (step 11.3): extracted from
 * InteractiveMode.showLoadedResources. Pure content building — the caller
 * owns the container and rendering; this module decides what is shown.
 */

export interface LoadedResourceItem {
	path: string;
	sourceInfo?: SourceInfo;
}

export interface LoadedResourcesData {
	contextFiles: Array<{ path: string }>;
	skills: ReadonlyArray<{ name: string; filePath: string; sourceInfo?: SourceInfo }>;
	skillDiagnostics: readonly ResourceDiagnostic[];
	promptDiagnostics: readonly ResourceDiagnostic[];
	templates: ReadonlyArray<{ name: string; filePath: string; sourceInfo?: SourceInfo }>;
	extensions: ReadonlyArray<LoadedResourceItem>;
	extensionDiagnostics: readonly ResourceDiagnostic[];
	themes: ReadonlyArray<{ name?: string; sourcePath?: string; sourceInfo?: SourceInfo }>;
	themeDiagnostics: readonly ResourceDiagnostic[];
	sourceInfos: Map<string, SourceInfo>;
}

export interface ResourceSection {
	name: string;
	color: ThemeColor;
	collapsed: string;
	expanded: string;
}

export interface DiagnosticBlock {
	title: string;
	body: string;
}

function formatCompactList(items: string[], options?: { sort?: boolean }): string {
	const labels = items.map((item) => item.trim()).filter((item) => item.length > 0);
	if (options?.sort !== false) {
		labels.sort((a, b) => a.localeCompare(b));
	}
	return theme.fg("dim", `  ${labels.join(", ")}`);
}

export function buildResourceSections(
	data: LoadedResourcesData,
	options: { expansion: boolean; includeDiagnostics: boolean },
): { sections: ResourceSection[]; diagnostics: DiagnosticBlock[] } {
	const sections: ResourceSection[] = [];
	const diagnostics: DiagnosticBlock[] = [];

	const sectionHeader = (name: string, color: ThemeColor = "mdHeading") =>
		theme.fg(color, `[${name}]`);

	const addSection = (
		name: string,
		collapsedBody: string,
		expandedBody = collapsedBody,
		color: ThemeColor = "mdHeading",
	): void => {
		sections.push({
			name,
			color,
			collapsed: `${sectionHeader(name, color)}\n${collapsedBody}`,
			expanded: `${sectionHeader(name, color)}\n${expandedBody}`,
		});
	};

	if (data.contextFiles.length > 0) {
		const contextList = data.contextFiles
			.map((f) => theme.fg("dim", `  ${formatDisplayPath(f.path)}`))
			.join("\n");
		const contextCompactList = formatCompactList(
			data.contextFiles.map((f) => f.path),
			{ sort: false },
		);
		addSection("Context", contextCompactList, contextList);
	}

	if (data.skills.length > 0) {
		const groups = buildScopeGroups(
			data.skills.map((skill) => ({ path: skill.filePath, sourceInfo: skill.sourceInfo })),
		);
		const skillList = formatScopeGroups(groups, {
			formatPath: (item) => formatDisplayPath(item.path),
			formatPackagePath: (item) => getShortPath(item.path, item.sourceInfo),
		});
		const skillCompactList = formatCompactList(data.skills.map((skill) => skill.name));
		addSection("Skills", skillCompactList, skillList);
	}

	if (data.templates.length > 0) {
		const groups = buildScopeGroups(
			data.templates.map((template) => ({ path: template.filePath, sourceInfo: template.sourceInfo })),
		);
		const templateByPath = new Map(data.templates.map((t) => [t.filePath, t]));
		const templateList = formatScopeGroups(groups, {
			formatPath: (item) => {
				const template = templateByPath.get(item.path);
				return template ? `/${template.name}` : formatDisplayPath(item.path);
			},
			formatPackagePath: (item) => {
				const template = templateByPath.get(item.path);
				return template ? `/${template.name}` : formatDisplayPath(item.path);
			},
		});
		const templateCompactList = formatCompactList(data.templates.map((template) => `/${template.name}`));
		addSection("Prompts", templateCompactList, templateList);
	}

	if (data.extensions.length > 0) {
		const groups = buildScopeGroups([...data.extensions]);
		const extList = formatScopeGroups(groups, {
			formatPath: (item) => formatExtensionDisplayPath(item.path),
			formatPackagePath: (item) => formatExtensionDisplayPath(getShortPath(item.path, item.sourceInfo)),
		});
		const extensionCompactList = formatCompactList(getCompactExtensionLabels([...data.extensions]));
		addSection("Extensions", extensionCompactList, extList, "mdHeading");
	}

	const customThemes = data.themes.filter((t) => t.sourcePath);
	if (customThemes.length > 0) {
		const groups = buildScopeGroups(
			customThemes.map((loadedTheme) => ({
				path: loadedTheme.sourcePath!,
				sourceInfo: loadedTheme.sourceInfo,
			})),
		);
		const themeList = formatScopeGroups(groups, {
			formatPath: (item) => formatDisplayPath(item.path),
			formatPackagePath: (item) => getShortPath(item.path, item.sourceInfo),
		});
		const themeCompactList = formatCompactList(
			customThemes.map(
				(loadedTheme) =>
					loadedTheme.name ?? getCompactPathLabel(loadedTheme.sourcePath!, loadedTheme.sourceInfo),
			),
		);
		addSection("Themes", themeCompactList, themeList);
	}

	if (options.includeDiagnostics) {
		const diag = (title: string, list: readonly ResourceDiagnostic[]): void => {
			if (list.length > 0) {
				diagnostics.push({
					title,
					body: formatDiagnostics(list, data.sourceInfos),
				});
			}
		};
		diag("Skill conflicts", data.skillDiagnostics);
		diag("Prompt conflicts", data.promptDiagnostics);
		diag("Extension issues", data.extensionDiagnostics);
		diag("Theme conflicts", data.themeDiagnostics);
	}

	return { sections, diagnostics };
}
