"""Rich help rendering for legacy argparse-backed leaf commands."""

from __future__ import annotations

import argparse
import shutil
import sys
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichArgumentParser(argparse.ArgumentParser):
    """Keep argparse parsing semantics while matching Typer's Rich help UI."""

    def format_help(self) -> str:
        output = StringIO()
        is_terminal = bool(getattr(sys.stdout, "isatty", lambda: False)())
        width = max(60, shutil.get_terminal_size(fallback=(100, 24)).columns)
        console = Console(
            file=output,
            force_terminal=is_terminal,
            color_system="auto" if is_terminal else None,
            width=width,
        )

        usage = self._compact_usage()
        usage_line = Text(" Usage: ", style="bold")
        usage_line.append(usage)
        console.print()
        console.print(usage_line)

        if self.description:
            console.print()
            console.print(Text(str(self.description)))

        for group in self._action_groups:
            actions = [
                action
                for action in group._group_actions
                if action.help != argparse.SUPPRESS
            ]
            if not actions:
                continue

            table = Table.grid(expand=True, padding=(0, 2))
            table.add_column(style="bold cyan", ratio=2, overflow="fold")
            table.add_column(ratio=5, overflow="fold")
            formatter = self._get_formatter()
            for action in actions:
                invocation = self._action_invocation(action)
                try:
                    help_text = formatter._expand_help(action)
                except (KeyError, TypeError, ValueError):
                    help_text = "" if action.help is None else str(action.help)
                if action.choices:
                    choices = ", ".join(str(choice) for choice in action.choices)
                    help_text = f"{help_text} Choices: {choices}.".strip()
                if action.required and action.option_strings:
                    help_text = f"{help_text} Required.".strip()
                table.add_row(invocation, help_text)

            title = self._group_title(group.title)
            console.print()
            console.print(
                Panel(
                    table,
                    title=title,
                    title_align="left",
                    border_style="dim",
                    padding=(0, 1),
                )
            )

        if self.epilog:
            console.print()
            console.print(Text(str(self.epilog)))
        console.print()
        return output.getvalue()

    def _compact_usage(self) -> str:
        parts = [self.prog]
        if any(action.option_strings for action in self._actions):
            parts.append("[OPTIONS]")
        for action in self._actions:
            if action.option_strings or action.help == argparse.SUPPRESS:
                continue
            parts.append(self._positional_metavar(action))
        return " ".join(parts)

    def _action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return self._positional_metavar(action)
        invocation = ", ".join(action.option_strings)
        if action.nargs != 0:
            metavar = str(action.metavar or action.dest.upper())
            invocation = f"{invocation} {metavar}"
        return invocation

    @staticmethod
    def _positional_metavar(action: argparse.Action) -> str:
        metavar = str(action.metavar or action.dest.upper())
        if action.nargs == "?":
            return f"[{metavar}]"
        if action.nargs == "*":
            return f"[{metavar}]..."
        if action.nargs in {"+", argparse.REMAINDER}:
            return f"{metavar}..."
        if isinstance(action.nargs, int) and action.nargs > 1:
            return " ".join([metavar] * action.nargs)
        return metavar

    @staticmethod
    def _group_title(title: str | None) -> str:
        normalized = (title or "Options").strip().lower()
        if normalized == "positional arguments":
            return "Arguments"
        if normalized == "optional arguments":
            return "Options"
        return (title or "Options").strip().title()
