const beliefState = {
  query: {
    label: "Selected belief · B17",
    text: "The assistant is using a targeted search strategy to resolve the remaining identity clue.",
    confidence: "0.62",
    meter: "62%",
    relation: "adds context → B25",
    source: "source: tool call",
  },
  evidence: {
    label: "Selected belief · B25",
    text: "Two independent records identify the same person and location.",
    confidence: "0.78",
    meter: "78%",
    relation: "supports → D04",
    source: "source: search result",
  },
  conflict: {
    label: "Selected belief · B38",
    text: "An earlier snippet suggests a different age and should not drive the final answer.",
    confidence: "0.31",
    meter: "31%",
    relation: "contradicts → B25",
    source: "source: earlier result",
  },
  decision: {
    label: "Selected decision · D04",
    text: "Answer with the identity supported by the higher-confidence records.",
    confidence: "0.84",
    meter: "84%",
    relation: "depends on → B25",
    source: "source: final reasoning",
  },
};

const header = document.querySelector("[data-header]");
const navToggle = document.querySelector("[data-nav-toggle]");
const nav = document.querySelector("[data-nav]");

const updateHeader = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 12);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const isOpen = nav?.classList.toggle("is-open") ?? false;
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const graphStage = document.querySelector("[data-graph-stage]");
const graphNodes = graphStage?.querySelectorAll("[data-belief]") ?? [];
const readoutLabel = graphStage?.querySelector("[data-readout-label]");
const readoutText = graphStage?.querySelector("[data-readout-text]");
const readoutConfidence = graphStage?.querySelector("[data-readout-confidence]");
const readoutMeter = graphStage?.querySelector("[data-readout-meter]");
const readoutRelation = graphStage?.querySelector("[data-readout-relation]");
const readoutSource = graphStage?.querySelector("[data-readout-source]");

graphNodes.forEach((node) => {
  node.addEventListener("click", () => {
    const belief = beliefState[node.dataset.belief];
    if (!belief) return;

    graphNodes.forEach((item) => {
      const selected = item === node;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-pressed", String(selected));
    });

    if (readoutLabel) readoutLabel.textContent = belief.label;
    if (readoutText) readoutText.textContent = belief.text;
    if (readoutConfidence) readoutConfidence.textContent = belief.confidence;
    if (readoutMeter) readoutMeter.style.width = belief.meter;
    if (readoutRelation) readoutRelation.textContent = belief.relation;
    if (readoutSource) readoutSource.textContent = belief.source;
  });
});

const codeTabs = document.querySelectorAll("[data-code-tab]");
const codePanels = document.querySelectorAll("[data-code-panel]");

const activateCodeTab = (name) => {
  codeTabs.forEach((tab) => {
    const active = tab.dataset.codeTab === name;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });

  codePanels.forEach((panel) => {
    const active = panel.dataset.codePanel === name;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
};

codeTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateCodeTab(tab.dataset.codeTab));
  tab.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + codeTabs.length) % codeTabs.length;
    codeTabs[nextIndex].focus();
    activateCodeTab(codeTabs[nextIndex].dataset.codeTab);
  });
});

const writeClipboard = async (text) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
};

const showCopied = (button) => {
  const original = button.textContent;
  button.textContent = "Copied";
  window.setTimeout(() => {
    button.textContent = original;
  }, 1400);
};

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    await writeClipboard(target.innerText);
    showCopied(button);
  });
});

document.querySelectorAll("[data-copy-install]").forEach((button) => {
  button.addEventListener("click", async () => {
    const command = button.closest(".install-command")?.querySelector("code")?.textContent;
    if (!command) return;
    await writeClipboard(command);
    showCopied(button);
  });
});

const year = document.querySelector("[data-year]");
if (year) year.textContent = String(new Date().getFullYear());
