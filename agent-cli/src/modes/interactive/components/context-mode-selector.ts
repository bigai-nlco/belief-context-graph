import { Container, type SelectItem, SelectList, type SelectListLayoutOptions } from "@bigai-nlco/bcg-tui";
import type { ContextManagementProvider } from "../../../core/settings-manager.ts";
import { getSelectListTheme } from "../theme/theme.ts";
import { DynamicBorder } from "./dynamic-border.ts";

const CONTEXT_MODE_SELECT_LIST_LAYOUT: SelectListLayoutOptions = {
	minPrimaryColumnWidth: 12,
	maxPrimaryColumnWidth: 32,
};

export class ContextModeSelectorComponent extends Container {
	private readonly selectList: SelectList;

	constructor(
		currentMode: ContextManagementProvider,
		onSelect: (mode: ContextManagementProvider) => void,
		onCancel: () => void,
	) {
		super();

		const items: SelectItem[] = [
			{
				value: "bcg",
				label: "BCG",
				description: "Graph memory · initial request + configured recent turns",
			},
			{
				value: "summary",
				label: "Summary",
				description: "Rolling LLM summary · initial request + recent turns",
			},
			{
				value: "rag",
				label: "RAG",
				description: "Retrieved local history · initial request + recent turns",
			},
			{
				value: "recent-only",
				label: "Recent-Only",
				description: "Initial request + recent turns · no long-term memory",
			},
			{
				value: "default",
				label: "Default",
				description: "Full agent context · automatic compaction",
			},
		];

		this.addChild(new DynamicBorder());
		this.selectList = new SelectList(items, 7, getSelectListTheme(), CONTEXT_MODE_SELECT_LIST_LAYOUT);
		const selectedIndex = items.findIndex((item) => item.value === currentMode);
		this.selectList.setSelectedIndex(selectedIndex >= 0 ? selectedIndex : items.length - 1);
		this.selectList.onSelect = (item) => {
			onSelect(item.value as ContextManagementProvider);
		};
		this.selectList.onCancel = onCancel;
		this.addChild(this.selectList);
		this.addChild(new DynamicBorder());
	}

	getSelectList(): SelectList {
		return this.selectList;
	}
}
