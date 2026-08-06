import { describe, expect, it } from "vitest";
import * as tui from "@bigai-nlco/bcg-tui";

// Public export contract for @bigai-nlco/bcg-tui (step 11).
// Independently publishable UI component package; no dependency on the
// agent shell or model layer.
describe("bcg-tui public exports", () => {
	it("exposes layout and text primitives", () => {
		expect(typeof tui.Box).toBe("function");
		expect(typeof tui.Text).toBe("function");
		expect(typeof tui.Spacer).toBe("function");
	});

	it("exposes interactive components", () => {
		expect(typeof tui.Input).toBe("function");
		expect(typeof tui.Markdown).toBe("function");
		expect(typeof tui.SettingsList).toBe("function");
		expect(typeof tui.Loader).toBe("function");
	});
});
