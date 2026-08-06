/**
 * Slash-command argument parsing (step 11.3): extracted from
 * InteractiveMode.getPathCommandArgument. Pure and unit-testable.
 */

export type PathCommandName = "/export" | "/import";

export function getPathCommandArgument(text: string, command: PathCommandName): string | undefined {
	if (text === command) {
		return undefined;
	}
	if (!text.startsWith(`${command} `)) {
		return undefined;
	}

	const argsString = text.slice(command.length + 1).trimStart();
	if (!argsString) {
		return undefined;
	}

	const firstChar = argsString[0];
	if (firstChar === '"' || firstChar === "'") {
		const closingQuoteIndex = argsString.indexOf(firstChar, 1);
		if (closingQuoteIndex < 0) {
			return undefined;
		}
		return argsString.slice(1, closingQuoteIndex);
	}

	const firstWhitespaceIndex = argsString.search(/\s/);
	if (firstWhitespaceIndex < 0) {
		return argsString;
	}
	return argsString.slice(0, firstWhitespaceIndex);
}

export function getCommandNameArgument(text: string, command: string): string {
	return text.replace(new RegExp(`^${command}\\s*`), "").trim();
}
