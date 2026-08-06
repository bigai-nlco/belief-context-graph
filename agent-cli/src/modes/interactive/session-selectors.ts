import type { Component } from "@bigai-nlco/bcg-tui";
import type { KeybindingsManager } from "../../core/keybindings.ts";
import type { SessionInfo, SessionListProgress } from "../../core/session-manager.ts";
import { SessionSelectorComponent } from "./components/session-selector.ts";

/**
 * Session selector flow (step 11.3): the /sessions picker orchestration
 * extracted from InteractiveMode. Listing, resume and rename operations are
 * injected; the flow wires the component and its callbacks.
 */
export interface SessionSelectorDeps {
	showSelector(create: (done: () => void) => { component: Component; focus: Component }): void;
	requestRender(): void;
	resumeSession(sessionPath: string): Promise<void>;
	shutdown(): void;
	keybindings: KeybindingsManager;
	listSessions(
		cwd: string,
		sessionDir: string,
		onProgress?: SessionListProgress,
	): Promise<SessionInfo[]>;
	listAllSessions(
		sessionDir: string | undefined,
		onProgress?: SessionListProgress,
	): Promise<SessionInfo[]>;
	usesDefaultSessionDir(): boolean;
	getCwd(): string;
	getSessionDir(): string;
	getSessionFile(): string | undefined;
	renameSession(sessionFilePath: string, nextName: string | undefined): void;
}

export function showSessionSelector(deps: SessionSelectorDeps): void {
	deps.showSelector((done) => {
		const selector = new SessionSelectorComponent(
			(onProgress) => deps.listSessions(deps.getCwd(), deps.getSessionDir(), onProgress),
			(onProgress) =>
				deps.usesDefaultSessionDir()
					? deps.listAllSessions(undefined, onProgress)
					: deps.listAllSessions(deps.getSessionDir(), onProgress),
			async (sessionPath) => {
				done();
				await deps.resumeSession(sessionPath);
			},
			() => {
				done();
				deps.requestRender();
			},
			() => {
				deps.shutdown();
			},
			() => deps.requestRender(),
			{
				renameSession: async (sessionFilePath: string, nextName: string | undefined) => {
					const next = (nextName ?? "").trim();
					if (!next) return;
					deps.renameSession(sessionFilePath, next);
				},
				showRenameHint: true,
				keybindings: deps.keybindings,
			},
			deps.getSessionFile(),
		);
		return { component: selector, focus: selector };
	});
}
