import { beforeAll, describe, expect, it } from "vitest";
import {
	buildScopeGroups,
	formatDiagnostics,
	formatDisplayPath,
	getCompactExtensionLabel,
	getCompactExtensionLabels,
	getCompactPathLabel,
	getDisplaySourceInfo,
	getScopeGroup,
	getShortPath,
	isPackageSource,
} from "../src/modes/interactive/display-format.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import type { SourceInfo } from "../src/core/source-info.ts";

function npmSource(baseDir: string, source = "npm:@scope/pkg"): SourceInfo {
	return { source, baseDir, scope: "project" } as SourceInfo;
}

describe("display-format (step 11.3)", () => {
	beforeAll(() => {
		initTheme("default", false);
	});
	describe("formatDisplayPath", () => {
		it("replaces the home prefix with ~", () => {
			const home = process.env.HOME!;
			expect(formatDisplayPath(`${home}/project/file.ts`)).toBe("~/project/file.ts");
		});

		it("keeps non-home paths unchanged", () => {
			expect(formatDisplayPath("/tmp/x.ts")).toBe("/tmp/x.ts");
		});
	});

	describe("getShortPath", () => {
		it("shortens paths inside a node_modules package to package-relative", () => {
			const baseDir = "/proj/node_modules/@scope/pkg";
			const full = "/proj/node_modules/@scope/pkg/src/index.ts";
			expect(getShortPath(full, npmSource(baseDir))).toBe("src/index.ts");
		});

		it("returns the file after node_modules for npm sources", () => {
			const sourceInfo = npmSource("/x/node_modules/a/b");
			const full = "/x/node_modules/a/b/dist/main.js";
			expect(getShortPath(full, sourceInfo)).toBe("dist/main.js");
		});

		it("falls back to display path for local sources", () => {
			expect(getShortPath("/home/u/proj/a/b.ts", { source: "local", scope: "project" })).toContain(
				"a/b.ts",
			);
		});
	});

	describe("labels and grouping", () => {
		it("getCompactPathLabel returns the last path segment", () => {
			expect(getCompactPathLabel("/a/b/c.ts", { source: "local", scope: "project" })).toBe("c.ts");
		});

		it("getCompactExtensionLabel formats package sources as source:path", () => {
			const info = npmSource("/proj/node_modules/@scope/pkg", "npm:@scope/pkg");
			const label = getCompactExtensionLabel("/proj/node_modules/@scope/pkg/src/x.ts", info);
			expect(label).toBe("@scope/pkg:src/x.ts");
		});

		it("getCompactExtensionLabels disambiguates shared segment names", () => {
			const local = { source: "local", scope: "project" };
			const labels = getCompactExtensionLabels([
				{ path: "/a/one/util.ts", sourceInfo: local },
				{ path: "/b/one/util.ts", sourceInfo: local },
			]);
			expect(labels[0]).not.toBe(labels[1]);
			expect(labels[0]).toContain("util.ts");
		});

		it("isPackageSource detects npm/git sources", () => {
			expect(isPackageSource(npmSource("/x"))).toBe(true);
			expect(isPackageSource({ source: "local", scope: "project" })).toBe(false);
		});

		it("getScopeGroup maps source/scope combinations", () => {
			expect(getScopeGroup({ source: "local", scope: "user" })).toBe("user");
			expect(getScopeGroup({ source: "local", scope: "project" })).toBe("project");
			expect(getScopeGroup({ source: "local", scope: "temporary" })).toBe("path");
			expect(getScopeGroup({ source: "cli", scope: "project" })).toBe("path");
		});

		it("getDisplaySourceInfo labels sources", () => {
			expect(getDisplaySourceInfo({ source: "local", scope: "user" })).toEqual({
				label: "user",
				color: "muted",
			});
			expect(getDisplaySourceInfo(npmSource("/x"))).toMatchObject({ label: "npm:@scope/pkg" });
		});
	});

	describe("buildScopeGroups and diagnostics", () => {
		it("groups items by scope and packages by source", () => {
			const groups = buildScopeGroups([
				{ path: "/a", sourceInfo: { source: "local", scope: "project" } },
				{ path: "/b", sourceInfo: { source: "local", scope: "user" } },
				{ path: "/pkg/x", sourceInfo: npmSource("/n/m/pkg", "npm:pkg") },
				{ path: "/pkg/y", sourceInfo: npmSource("/n/m/pkg", "npm:pkg") },
			]);
			// npm packages resolve to the project scope; path group stays empty
			expect(groups).toHaveLength(2);
			const project = groups.find((g) => g.scope === "project")!;
			expect(project.paths.map((p) => p.path)).toEqual(["/a"]);
			expect(project.packages.get("npm:pkg")).toHaveLength(2);
			expect(groups.find((g) => g.scope === "user")).toBeDefined();
		});

		it("formatDiagnostics renders collision winners and losers", () => {
			const diagnostics = [
				{
					type: "collision",
					message: "duplicate",
					collision: {
						name: "my-tool",
						winnerPath: "/a/one.ts",
						loserPath: "/b/one.ts",
					},
				} as never,
			];
			const output = formatDiagnostics(diagnostics, new Map());
			expect(output).toContain('"my-tool" collision:');
			expect(output).toContain("✓");
			expect(output).toContain("(skipped)");
		});
	});
});
