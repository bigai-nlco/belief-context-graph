import { APP_NAME } from "../config.ts";
import type { SourceInfo } from "./source-info.ts";

export type SlashCommandSource = "extension" | "prompt" | "skill";

export interface SlashCommandInfo {
	name: string;
	description?: string;
	source: SlashCommandSource;
	sourceInfo: SourceInfo;
}

export interface BuiltinSlashCommand {
	name: string;
	description: string;
	argumentHint?: string;
}

export const BUILTIN_SLASH_COMMANDS: ReadonlyArray<BuiltinSlashCommand> = [
	{ name: "help", description: "Show BCG commands and keyboard controls" },
	{ name: "model", description: "Select the inference model", argumentHint: "<provider/model>" },
	{
		name: "mode",
		description: "Choose Default, BCG, or Summary context",
		argumentHint: "<default|bcg|summary>",
	},
	{ name: "login", description: "Configure the model API key", argumentHint: "<provider>" },
	{ name: "logout", description: "Remove a saved model API key" },
	{ name: "new", description: "Start a fresh BCG session" },
	{ name: "resume", description: "Resume a previous BCG session" },
	{ name: "graph", description: "Show Graph server and context status" },
	{ name: "exit", description: `Exit ${APP_NAME}` },
];
