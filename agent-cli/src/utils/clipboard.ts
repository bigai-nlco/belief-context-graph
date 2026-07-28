import { execFileSync, execSync, spawn } from "child_process";
import { platform } from "os";
import { isWaylandSession } from "./clipboard-image.ts";

type NativeClipboardExecOptions = {
	input: string;
	timeout: number;
	stdio: ["pipe", "ignore", "ignore"];
};

function copyToX11Clipboard(options: NativeClipboardExecOptions): void {
	try {
		execSync("xclip -selection clipboard", options);
	} catch {
		execSync("xsel --clipboard --input", options);
	}
}

const MAX_OSC52_ENCODED_LENGTH = 100_000;

function isRemoteSession(env: NodeJS.ProcessEnv = process.env): boolean {
	return Boolean(env.SSH_CONNECTION || env.SSH_CLIENT || env.MOSH_CONNECTION);
}

function emitOsc52(text: string): boolean {
	const encoded = Buffer.from(text).toString("base64");
	if (encoded.length > MAX_OSC52_ENCODED_LENGTH) {
		return false;
	}
	process.stdout.write(`\x1b]52;c;${encoded}\x07`);
	return true;
}

/** Read plain text from the system clipboard using platform commands. */
export async function readClipboardText(): Promise<string | null> {
	try {
		const p = platform();
		let text: string;
		if (p === "darwin") {
			text = execFileSync("pbpaste", [], { encoding: "utf8", timeout: 5000 });
		} else if (p === "win32") {
			text = execFileSync(
				"powershell.exe",
				["-NoProfile", "-Command", "Get-Clipboard -Raw"],
				{ encoding: "utf8", timeout: 5000 },
			);
		} else if (process.env.TERMUX_VERSION) {
			text = execFileSync("termux-clipboard-get", [], { encoding: "utf8", timeout: 5000 });
		} else if (isWaylandSession() && process.env.WAYLAND_DISPLAY) {
			text = execFileSync("wl-paste", ["--no-newline"], { encoding: "utf8", timeout: 5000 });
		} else {
			try {
				text = execFileSync("xclip", ["-selection", "clipboard", "-o"], { encoding: "utf8", timeout: 5000 });
			} catch {
				text = execFileSync("xsel", ["--clipboard", "--output"], { encoding: "utf8", timeout: 5000 });
			}
		}
		return text || null;
	} catch {
		return null;
	}
}

export async function copyToClipboard(text: string): Promise<void> {
	let copied = false;

	const p = platform();

	const remote = isRemoteSession();
	const options: NativeClipboardExecOptions = { input: text, timeout: 5000, stdio: ["pipe", "ignore", "ignore"] };

	if (!copied) {
		try {
			if (p === "darwin") {
				execSync("pbcopy", options);
				copied = true;
			} else if (p === "win32") {
				execSync("clip", options);
				copied = true;
			} else {
				// Linux. Try Termux, Wayland, or X11 clipboard tools.
				if (process.env.TERMUX_VERSION) {
					try {
						execSync("termux-clipboard-set", options);
						copied = true;
					} catch {
						// Fall back to Wayland or X11 tools.
					}
				}

				if (!copied) {
					const hasWaylandDisplay = Boolean(process.env.WAYLAND_DISPLAY);
					const hasX11Display = Boolean(process.env.DISPLAY);
					const isWayland = isWaylandSession();
					if (isWayland && hasWaylandDisplay) {
						try {
							// Verify wl-copy exists (spawn errors are async and won't be caught)
							execSync("which wl-copy", { stdio: "ignore" });
							// wl-copy with execSync hangs due to fork behavior; use spawn instead.
							// Await the exit code and only claim success on a clean exit, so a
							// failed wl-copy falls through to the xclip/OSC 52 fallbacks.
							const wlCopyExit = await new Promise<number>((resolve) => {
								const proc = spawn("wl-copy", [], { stdio: ["pipe", "ignore", "ignore"] });
								proc.on("error", () => resolve(1));
								proc.on("close", (code) => resolve(code ?? 1));
								proc.stdin.on("error", () => {
									// Ignore EPIPE errors if wl-copy exits early
								});
								proc.stdin.write(text);
								proc.stdin.end();
							});
							if (wlCopyExit === 0) {
								copied = true;
							} else if (hasX11Display) {
								copyToX11Clipboard(options);
								copied = true;
							}
						} catch {
							if (hasX11Display) {
								copyToX11Clipboard(options);
								copied = true;
							}
						}
					} else if (hasX11Display) {
						copyToX11Clipboard(options);
						copied = true;
					}
				}
			}
		} catch {
			// Fall through to OSC 52 fallback.
		}
	}

	if (remote || !copied) {
		const osc52Copied = emitOsc52(text);
		copied = copied || osc52Copied;
	}

	if (!copied) {
		throw new Error("Failed to copy to clipboard");
	}
}
