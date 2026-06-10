const SESSIONS_KEY = "apodex.ui.sessions.v1";
const ACTIVE_KEY = "apodex.ui.activeSession.v1";
const API_BASE_KEY = "apodex.apiBase";
let API_BASE = resolveInitialApiBase();
const STREAM_PERSIST_INTERVAL_MS = 500;
const HEARTBEAT_RENDER_INTERVAL_MS = 1000;
const INQUIRY_WALL_TIME_S = 900;

const curatedExamples = [
  {
    id: "01",
    title: "Will the Fed cut rates before Q3 2026, and what signals support that view?",
    category: "MACROECONOMICS"
  },
  {
    id: "02",
    title: "What's the current evidence on GLP-1's long-term cardiovascular effects?",
    category: "MEDICINE"
  },
  {
    id: "03",
    title: "Trace former SGM contracts where the winner eventually underperformed.",
    category: "PROCUREMENT"
  },
  {
    id: "04",
    title: "Which public AI labs disclosed eval changes after a model launch?",
    category: "AI GOVERNANCE"
  },
  {
    id: "05",
    title: "What evidence distinguishes one-time revenue from durable demand?",
    category: "EQUITIES"
  }
];

const state = {
  sessions: readSessions(),
  activeSessionId: localStorage.getItem(ACTIVE_KEY) || null,
  query: "",
  followupQuery: "",
  submitError: "",
  modelLabel: "OPENAI_MODEL",
  defaultProfile: "default",
  supportedModels: [],
  userMenuOpen: false,
  followupOpen: false,
  expandedProcesses: {}
};

let streamRenderFrame = 0;
let lastStreamPersistAt = 0;
let heartbeatRenderTimer = 0;
let chatScrollFrame = 0;
const activeInquiryControllers = new Map();

const app = document.getElementById("app");
render({ autoScroll: Boolean(state.activeSessionId) });
void loadAppConfig();

function resolveInitialApiBase() {
  const stored = localStorage.getItem(API_BASE_KEY);
  const defaultBase = defaultApiBase();
  if (stored && !(isNonLoopbackHost(window.location.hostname) && isLoopbackApiBase(stored))) {
    return stored.replace(/\/+$/, "");
  }
  return defaultBase;
}

function defaultApiBase() {
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  const host = window.location.hostname || "127.0.0.1";
  return `${protocol}//${host}:8000`;
}

function isNonLoopbackHost(hostname) {
  return Boolean(hostname && hostname !== "localhost" && hostname !== "127.0.0.1" && hostname !== "::1");
}

function isLoopbackApiBase(value) {
  try {
    const url = new URL(value);
    return url.hostname === "localhost" || url.hostname === "127.0.0.1" || url.hostname === "::1";
  } catch {
    return false;
  }
}

function render(options = {}) {
  const autoScroll = Boolean(options.autoScroll);
  const activeSession = state.sessions.find((session) => session.id === state.activeSessionId);
  const scrollSnapshot = captureChatScroll();
  app.innerHTML = `
    <div class="app-shell">
      ${renderSidebar()}
      <main class="main-pane">
        ${activeSession ? renderSessionView(activeSession) : renderLanding()}
      </main>
    </div>
  `;
  bindEvents();
  syncHeartbeatRenderTimer();
  if (activeSession) {
    if (autoScroll) {
      scrollActiveChatToBottom();
    } else {
      restoreChatScroll(scrollSnapshot);
    }
    if (autoScroll && activeSession.status === "running") {
      scheduleActiveChatScrollToBottom();
    }
  }
}

function captureChatScroll() {
  const scroller = document.querySelector(".chat-scroll");
  if (!scroller) {
    return null;
  }
  return {
    scrollTop: scroller.scrollTop,
    windowY: window.scrollY
  };
}

function restoreChatScroll(snapshot) {
  if (!snapshot) {
    return;
  }
  const scroller = document.querySelector(".chat-scroll");
  if (scroller) {
    const maxTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    scroller.scrollTop = Math.min(snapshot.scrollTop, maxTop);
  }
  const root = document.documentElement;
  if (root.scrollHeight > root.clientHeight) {
    const maxY = Math.max(0, root.scrollHeight - root.clientHeight);
    window.scrollTo(0, Math.min(snapshot.windowY, maxY));
  }
}

function scheduleActiveChatScrollToBottom() {
  if (chatScrollFrame) {
    cancelAnimationFrame(chatScrollFrame);
  }
  chatScrollFrame = requestAnimationFrame(() => {
    chatScrollFrame = 0;
    scrollActiveChatToBottom();
  });
}

function scrollActiveChatToBottom() {
  const scroller = document.querySelector(".chat-scroll");
  if (!scroller) {
    return;
  }
  scroller.scrollTop = scroller.scrollHeight;
  const root = document.documentElement;
  if (root.scrollHeight > root.clientHeight) {
    window.scrollTo(0, root.scrollHeight);
  }
}

function renderSidebar() {
  const chats = state.sessions.length
    ? `
      <section class="chat-list" aria-label="Local chats">
        <div class="chat-list-header">
          <span>Chats</span>
          <span>${state.sessions.length}</span>
        </div>
        <div class="chat-items">
          ${state.sessions.map(renderChatItem).join("")}
        </div>
      </section>
    `
    : "";

  return `
    <aside class="sidebar" aria-label="Apodex navigation">
      <div class="sidebar-scroll">
        <button class="brand-button" type="button" data-action="new" aria-label="Apodex home">
          <span class="brand-lockup" aria-hidden="true">
            <svg viewBox="0 0 48 40">
              <path d="M4 34 23 6l21 28" />
              <path d="M4 34 18 18l6 7" />
              <path d="m25 3 1.5 4.5L31 9l-4.5 1.5L25 15l-1.5-4.5L19 9l4.5-1.5Z" />
            </svg>
            <span>
              <strong>Apodex</strong>
              <em>self-evolving</em>
            </span>
          </span>
        </button>
        <button class="sidebar-new-button" type="button" data-action="new">
          ${icon("plus", 17)}
          <span>New inquiry</span>
        </button>
        ${chats}
      </div>
      <div class="sidebar-footer">
        <div class="user-menu-wrap">
          ${
            state.userMenuOpen
              ? `<div class="user-menu" role="menu">
                  <div class="user-menu-row" role="menuitem">
                    ${icon("settings", 17)}
                    <span>Settings</span>
                  </div>
                  <div class="user-menu-meta">
                    <span>Model</span>
                    <strong>${escapeHtml(state.modelLabel)}</strong>
                  </div>
                </div>`
              : ""
          }
          <button class="user-button" type="button" data-action="toggle-user-menu" aria-expanded="${state.userMenuOpen ? "true" : "false"}">
            <span class="user-avatar">U</span>
            <span class="user-name">Placeholder user</span>
            ${icon("chevron-down", 15)}
          </button>
        </div>
      </div>
    </aside>
  `;
}

function renderChatItem(session) {
  const selected = session.id === state.activeSessionId ? " selected" : "";
  return `
    <button class="chat-item${selected}" type="button" data-session-id="${session.id}">
      <span>${escapeHtml(session.title)}</span>
      <span class="status-dot ${session.status}" aria-label="${session.status}"></span>
    </button>
  `;
}

function renderLanding() {
  const isSubmitting = state.sessions.some((session) => session.status === "running");
  const disabled = !state.query.trim() || isSubmitting ? "disabled" : "";
  const buttonText = isSubmitting
    ? `${icon("loader", 18, "spin")} Running`
    : "Begin inquiry";

  return `
    <div class="landing-wrap">
      <section class="hero" aria-labelledby="hero-title">
        <h1 id="hero-title">Ask the question that matters.</h1>
        <p>Apodex reasons through it step by step — verifying every conclusion before moving to the next. Not a chat reply. A verified brief.</p>
      </section>
      <form class="composer" id="inquiry-form">
        <label for="inquiry-input">ASK</label>
        <textarea id="inquiry-input" rows="5" placeholder="Pose a question, or pick one of the curated examples below...">${escapeHtml(state.query)}</textarea>
        <div class="composer-footer">
          <button class="model-select" type="button" aria-label="Selected model">
            ${icon("zap", 18)}
            <span>${escapeHtml(state.modelLabel)}</span>
            ${icon("chevron-down", 16)}
          </button>
          <div class="composer-actions">
            <button class="ghost-tool" type="button" aria-label="Attach file">${icon("paperclip", 22)}</button>
            <button class="ghost-tool" type="button" aria-label="Voice input">${icon("mic", 22)}</button>
            <button class="begin-button" type="button" ${disabled}>${buttonText}</button>
          </div>
        </div>
      </form>
      ${state.submitError ? `<p class="submit-error">${escapeHtml(state.submitError)}</p>` : ""}
      ${renderCuratedExamples()}
    </div>
  `;
}

function renderCuratedExamples() {
  return `
    <section class="curated" aria-label="Curated examples">
      <div class="curated-header">
        <div>
          ${icon("lock", 18)}
          <span>PICK ONE OF THESE</span>
        </div>
        <span>5 · CURATED</span>
      </div>
      <div class="curated-list">
        ${curatedExamples
          .map(
            (example, index) => `
              <button class="curated-row" type="button" data-preset-index="${index}">
                <span class="curated-number">${example.id}</span>
                <span class="curated-copy">
                  <strong>${escapeHtml(example.title)}</strong>
                  <small>${escapeHtml(example.category)}</small>
                </span>
                ${icon("plus", 18)}
              </button>
            `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderSessionView(session) {
  const liveTimeline = buildTimelineForRender(session);
  const researchStats = getResearchStats(session, liveTimeline);
  const isRunning = session.status === "running";
  const processExpanded = isRunning || Boolean(state.expandedProcesses[session.id]);
  const followupOpen = state.followupOpen && state.activeSessionId === session.id;

  return `
    <div class="chat-screen${followupOpen ? " modal-open" : ""}">
      <header class="chat-header">
        <h1>${escapeHtml(captionFromQuery(session.query))}</h1>
        <div class="chat-header-actions">
          <button class="header-icon-button" type="button" data-action="export-trace" aria-label="Export raw trace" title="Export raw trace">${icon("file-output", 18)}</button>
          <button class="header-followup-button" type="button" data-action="open-followup" ${isRunning ? "disabled" : ""}>
            ${icon("message-plus", 17)}
            Ask follow-up
          </button>
        </div>
      </header>
      <div class="chat-scroll">
        <article class="chat-canvas">
          <section class="question-turn" aria-label="Question">
            <div class="question-bubble">${escapeHtml(session.query)}</div>
            <time>${formatShortTime(session.createdAt)}</time>
          </section>
          ${renderResearchProcess(session, liveTimeline, researchStats, processExpanded)}
          ${renderReportPanel(session)}
        </article>
        <p class="content-advisory">The content is generated by Apodex. Critical review is advised.</p>
      </div>
      ${renderBottomAction(session)}
      ${followupOpen ? renderFollowupDialog(session) : ""}
    </div>
  `;
}

function renderResearchProcess(session, timeline, researchStats, expanded) {
  const isRunning = session.status === "running";
  const statusClass = isRunning ? " running" : "";
  const chevron = expanded ? "chevron-up" : "chevron-down";
  const summaryLabel = isRunning ? "Research process" : "Research process";
  const countLabel = `${researchStats.stepCount || 0} ${researchStats.stepCount === 1 ? "step" : "steps"}`;
  const toolLabel = `${researchStats.toolCount || 0} ${researchStats.toolCount === 1 ? "tool call" : "tool calls"}`;

  return `
    <section class="research-process${statusClass}" aria-label="Research process">
      <button class="process-summary" type="button" data-action="toggle-process" ${isRunning ? "disabled" : ""}>
        <span class="process-summary-left">
          ${isRunning ? icon("loader", 16, "spin") : icon("check", 16)}
          <strong>${summaryLabel}</strong>
          <span class="process-pill">${countLabel}</span>
          <span class="process-pill tool-count">${toolLabel}</span>
        </span>
        ${icon(chevron, 16)}
      </button>
      ${
        expanded
          ? `<div class="process-body">
              ${
                timeline.length
                  ? `<ol class="timeline-list">${timeline.map((item, index) => renderTimelineItem(item, index, researchStats)).join("")}</ol>`
                  : session.status === "running"
                    ? renderRunningTrace(session.streamStatus)
                    : `<div class="empty-trace">No research events were returned.</div>`
              }
              ${renderHeartbeat(session)}
            </div>`
          : ""
      }
    </section>
  `;
}

function renderTimelineItem(item, index, researchStats) {
  const displayIndex = index + 1;
  if (item.type === "reasoning") {
    const content = sanitizeReasoningMarkdown(item.content || "");
    if (!content && !item.live) {
      return "";
    }
    const stepNumber = getStepNumberForTimelineItem(item, researchStats);
    const stepLabel = `Step ${stepNumber || displayIndex}`;
    const stepBadge = item.live
      ? `<span class="live-pill">${escapeHtml(stepLabel)}</span>`
      : `<span>${escapeHtml(stepLabel)}</span>`;
    return `
      <li class="timeline-item thinking-item">
        <div class="timeline-rail">${icon("brain", 16)}</div>
        <div class="timeline-body">
          <div class="timeline-title-row">
            <h3>Thinking Process</h3>
            ${stepBadge}
          </div>
          <div class="markdown-body thinking-markdown">${renderMarkdown(content)}</div>
        </div>
      </li>
    `;
  }

  return renderToolTimelineItem(item.step || item, displayIndex);
}

function renderToolTimelineItem(step, displayIndex = step.index) {
  if (step?.tool_name === "web_search") {
    return renderSearchToolStep(step);
  }
  if (step?.tool_name === "web_fetch") {
    return renderReadingToolStep(step);
  }
  return renderGenericToolStep(step, displayIndex);
}

function renderSearchToolStep(step) {
  const results = parseSearchResults(step.observation);
  const queries = normalizeListArg(getToolArg(step, "q") || getToolArg(step, "query"));
  const queryText = queries.length ? queries.map((query) => `"${query}"`).join(", ") : "relevant evidence";
  const status = step.status === "running" ? "Searching" : `Found ${results.length} ${results.length === 1 ? "result" : "results"}`;

  return `
    <li class="timeline-item tool-item search-item ${step.status === "running" ? "running" : ""}">
      <div class="timeline-rail">${icon("search", 15)}</div>
      <div class="timeline-body">
        <div class="tool-query">Searching for ${escapeHtml(queryText)}</div>
        <div class="tool-status">${icon("list", 15)}<span>${escapeHtml(status)}</span></div>
        ${
          results.length
            ? `<ol class="search-results">${results.map(renderSearchResult).join("")}</ol>`
            : step.status === "running"
              ? `<div class="tool-pending">Waiting for search results...</div>`
              : renderToolFallback(step)
        }
      </div>
    </li>
  `;
}

function renderSearchResult(result) {
  const url = result.url || "";
  const host = domainFromUrl(url);
  return `
    <li>
      ${renderFavicon(url)}
      <a class="result-title" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(result.title || host || url)}</a>
      ${host ? `<a class="result-domain" href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(host)}</a>` : ""}
    </li>
  `;
}

function renderReadingToolStep(step) {
  const urls = uniqueStrings([
    ...normalizeListArg(getToolArg(step, "url")),
    ...extractUrls(step.observation)
  ]);
  const status = step.status === "running" ? "Reading" : "Read";

  return `
    <li class="timeline-item tool-item reading-item ${step.status === "running" ? "running" : ""}">
      <div class="timeline-rail">${icon("globe", 15)}</div>
      <div class="timeline-body">
        <div class="timeline-title-row">
          <h3>${status}</h3>
          ${step.status === "running" ? `<span class="live-pill">Running</span>` : step.duration_ms ? `<span>${step.duration_ms}ms</span>` : ""}
        </div>
        ${
          urls.length
            ? `<ul class="reading-list">${urls.map(renderReadingUrl).join("")}</ul>`
            : `<div class="tool-pending">Waiting for source URLs...</div>`
        }
        ${step.status === "error" ? renderToolFallback(step) : ""}
      </div>
    </li>
  `;
}

function renderReadingUrl(url) {
  const host = domainFromUrl(url) || url;
  return `
    <li>
      ${renderFavicon(url)}
      <a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a>
      <span>${escapeHtml(host)}</span>
    </li>
  `;
}

function renderGenericToolStep(step, displayIndex = step.index) {
  const isRunning = step.status === "running";
  const duration = isRunning ? `<span class="live-pill">Running</span>` : step.duration_ms ? `<span>${step.duration_ms}ms</span>` : "";
  return `
    <li class="timeline-item tool-item ${step.status === "error" ? "error" : ""} ${isRunning ? "running" : ""}">
      <div class="timeline-rail">${icon("tool", 15)}</div>
      <div class="timeline-body">
        <div class="timeline-title-row">
          <h3>${escapeHtml(step.title || `Tool ${displayIndex}`)}</h3>
          ${duration}
        </div>
        <p class="tool-summary">${escapeHtml(step.summary || "Used an available tool.")}</p>
        ${renderToolFallback(step)}
      </div>
    </li>
  `;
}

function renderToolFallback(step) {
  if (!step?.observation) {
    return "";
  }
  return `<pre class="tool-observation">${escapeHtml(step.observation)}</pre>`;
}

function sanitizeReasoningMarkdown(value) {
  let text = String(value || "");
  text = text.replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/gi, "");
  text = text.replace(/<function=[\s\S]*?<\/function>/gi, "");

  const danglingToolCall = text.search(/<tool_call\b/i);
  if (danglingToolCall !== -1) {
    text = text.slice(0, danglingToolCall);
  }

  const danglingFunction = text.search(/<function=/i);
  if (danglingFunction !== -1) {
    text = text.slice(0, danglingFunction);
  }

  return text.trim();
}

function renderHeartbeat(session) {
  if (session.status !== "running") {
    return "";
  }
  const value = session.lastHeartbeatAt || session.updatedAt || session.createdAt;
  if (elapsedMs(value) <= 1000) {
    return "";
  }
  return `
    <div class="heartbeat">
      <span></span>
      <strong>Last heartbeat ${formatRelativeAge(value)} ago</strong>
    </div>
  `;
}

function elapsedMs(value) {
  const then = new Date(value).getTime();
  if (!Number.isFinite(then)) {
    return 0;
  }
  return Math.max(0, Date.now() - then);
}

function renderReportPanel(session) {
  const response = session.response;
  const isRunning = session.status === "running";
  const answer = isRunning
    ? session.liveAnswer || response?.final_answer || ""
    : response?.final_answer || session.liveAnswer || "";
  const cleanAnswer = String(answer || "").trim();
  const duration = response?.duration_seconds ? `<span>${response.duration_seconds}s</span>` : "";

  if (session.status === "stopped" && !cleanAnswer) {
    return `<section class="report-panel stopped"><p>Stopped by user.</p></section>`;
  }

  if (response?.error && !cleanAnswer) {
    return `<section class="report-panel error"><p>${escapeHtml(response.error)}</p></section>`;
  }

  if (isRunning && !cleanAnswer) {
    return "";
  }

  return `
    <section class="report-panel ${isRunning ? "streaming" : ""}" aria-label="Final answer">
      <div class="report-heading">
        <span>Report</span>
        ${duration}
      </div>
      ${
        cleanAnswer
          ? renderReportContent(cleanAnswer, isRunning)
          : `<p class="answer-placeholder">No final answer was returned.</p>`
      }
    </section>
  `;
}

function renderReportContent(markdown, isLive) {
  const normalizedMarkdown = normalizeMarkdownReferenceNumbers(markdown);
  const { body, references } = splitReferences(normalizedMarkdown);
  return `
    <div class="report-content">
      <div class="markdown-body report-markdown">${renderMarkdown(body)}${isLive ? `<span class="stream-caret"></span>` : ""}</div>
      ${references.length ? renderReferences(references) : ""}
    </div>
  `;
}

function normalizeMarkdownReferenceNumbers(markdown) {
  const text = String(markdown || "");
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const referenceIndex = lines.findIndex((line) => /^\s*#{1,6}\s+references\s*:?\s*$/i.test(line));
  if (referenceIndex === -1) {
    return text;
  }

  const body = lines.slice(0, referenceIndex).join("\n");
  const heading = lines[referenceIndex];
  const referenceLines = lines.slice(referenceIndex + 1);
  const oldToNew = new Map();
  let nextIndex = 1;
  const normalizedReferenceLines = referenceLines.map((line) => {
    const match = line.match(/^(\s*)(?:[-*]\s*)?\[(\d+)](\s+.*)$/);
    if (!match) {
      return line;
    }
    const oldLabel = match[2];
    if (!oldToNew.has(oldLabel)) {
      oldToNew.set(oldLabel, String(nextIndex));
      nextIndex += 1;
    }
    return `${match[1]}[${oldToNew.get(oldLabel)}]${match[3]}`;
  });

  if (!oldToNew.size) {
    return text;
  }

  const normalizedBody = body.replace(/\[(\d+)]/g, (match, label) =>
    oldToNew.has(label) ? `[${oldToNew.get(label)}]` : match
  );
  return [normalizedBody, heading, ...normalizedReferenceLines].join("\n").trim();
}

function renderReferences(references) {
  return `
    <section class="references-section" aria-label="References">
      <h3>References</h3>
      <ol class="references-list">
        ${references.map((reference, index) => renderReference(reference, index + 1)).join("")}
      </ol>
    </section>
  `;
}

function renderReference(reference, index) {
  const host = domainFromUrl(reference.url || "");
  return `
    <li>
      <span class="reference-index">${index}</span>
      <div>
        ${
          reference.url
            ? `<a class="reference-title" href="${escapeHtml(reference.url)}" target="_blank" rel="noreferrer">${escapeHtml(reference.title || host || reference.url)}</a>`
            : `<strong class="reference-title">${escapeHtml(reference.title || "Reference")}</strong>`
        }
        ${host ? `<span class="reference-domain">${renderFavicon(reference.url)}${escapeHtml(host)}</span>` : ""}
      </div>
    </li>
  `;
}

function renderBottomAction(session) {
  if (session.status === "running") {
    return `
      <div class="bottom-action">
        <button class="floating-action cancel-action" type="button" data-stop-session-id="${escapeHtml(session.id)}">
          ${icon("ban", 18)}
          Cancel
        </button>
      </div>
    `;
  }

  return `
    <div class="bottom-action">
      <button class="floating-action followup-action" type="button" data-action="open-followup">
        ${icon("message-plus", 18)}
        Ask follow-up
      </button>
    </div>
  `;
}

function renderFollowupDialog(session) {
  const canSubmit = state.followupQuery.trim() ? "" : "disabled";
  return `
    <div class="modal-scrim" data-action="close-followup">
      <form class="followup-dialog" id="followup-form" aria-label="Follow-up question">
        <div class="followup-header">
          <span>${icon("corner-down-left", 15)} Follow-up question</span>
          <button class="header-icon-button" type="button" data-action="close-followup" aria-label="Close follow-up">${icon("x", 18)}</button>
        </div>
        <textarea id="followup-input" rows="5" placeholder="What about this brief is unresolved or worth a deeper look?">${escapeHtml(state.followupQuery)}</textarea>
        <div class="followup-footer">
          <button class="model-select" type="button" aria-label="Selected model">
            ${icon("zap", 17)}
            <span>${escapeHtml(state.modelLabel)}</span>
            ${icon("chevron-down", 15)}
          </button>
          <div class="composer-actions">
            <button class="ghost-tool" type="button" aria-label="Attach file">${icon("paperclip", 20)}</button>
            <button class="ghost-tool" type="button" aria-label="Voice input">${icon("mic", 20)}</button>
            <button class="begin-button" type="button" ${canSubmit}>Append chapter</button>
          </div>
        </div>
      </form>
    </div>
  `;
}

function renderRunningTrace(statusText = "Starting agent loop") {
  return `
    <div class="running-trace">
      <p class="stream-status">${escapeHtml(statusText || "Starting agent loop")}</p>
      ${[0, 1, 2]
        .map(
          () => `
            <div class="trace-skeleton">
              <span></span>
              <div><i></i><b></b></div>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function bindEvents() {
  document.querySelectorAll("[data-action='new']").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeSessionId = null;
      persistActiveSession();
      state.userMenuOpen = false;
      state.followupOpen = false;
      state.followupQuery = "";
      render();
    });
  });

  document.querySelectorAll("[data-session-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeSessionId = button.getAttribute("data-session-id");
      persistActiveSession();
      state.userMenuOpen = false;
      state.followupOpen = false;
      state.followupQuery = "";
      render({ autoScroll: true });
    });
  });

  document.querySelectorAll("[data-action='toggle-user-menu']").forEach((button) => {
    button.addEventListener("click", () => {
      state.userMenuOpen = !state.userMenuOpen;
      render();
    });
  });

  document.querySelectorAll("[data-action='toggle-process']").forEach((button) => {
    button.addEventListener("click", () => {
      const activeSession = state.sessions.find((session) => session.id === state.activeSessionId);
      if (!activeSession || activeSession.status === "running") {
        return;
      }
      state.expandedProcesses = {
        ...state.expandedProcesses,
        [activeSession.id]: !state.expandedProcesses[activeSession.id]
      };
      render();
    });
  });

  document.querySelectorAll("[data-action='export-trace']").forEach((button) => {
    button.addEventListener("click", () => {
      exportActiveSessionTrace();
    });
  });

  document.querySelectorAll("[data-action='open-followup']").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.hasAttribute("disabled")) {
        return;
      }
      state.followupOpen = true;
      state.followupQuery = "";
      render();
      const followupInput = document.getElementById("followup-input");
      if (followupInput) {
        followupInput.focus();
      }
    });
  });

  document.querySelectorAll("[data-action='close-followup']").forEach((element) => {
    element.addEventListener("click", () => {
      state.followupOpen = false;
      state.followupQuery = "";
      render();
    });
  });

  const followupDialog = document.querySelector(".followup-dialog");
  if (followupDialog) {
    followupDialog.addEventListener("click", (event) => {
      event.stopPropagation();
    });
  }

  document.querySelectorAll("[data-stop-session-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const sessionId = button.getAttribute("data-stop-session-id");
      if (sessionId) {
        void stopInquiry(sessionId);
      }
    });
  });

  const input = document.getElementById("inquiry-input");
  if (input) {
    input.addEventListener("input", (event) => {
      state.query = event.target.value;
      syncSubmitButton();
    });
  }

  const beginButton = document.querySelector("#inquiry-form .begin-button");
  if (beginButton) {
    beginButton.addEventListener("click", () => {
      state.query = getInquiryInputValue();
      void startInquiry(state.query).catch(handleInquiryStartError);
    });
  }

  const form = document.getElementById("inquiry-form");
  if (form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      state.query = getInquiryInputValue();
      void startInquiry(state.query).catch(handleInquiryStartError);
    });
  }

  const followupInput = document.getElementById("followup-input");
  if (followupInput) {
    followupInput.addEventListener("input", (event) => {
      state.followupQuery = event.target.value;
      const button = document.querySelector("#followup-form .begin-button");
      if (button) {
        button.disabled = !state.followupQuery.trim();
      }
    });
  }

  const followupSubmitButton = document.querySelector("#followup-form .begin-button");
  if (followupSubmitButton) {
    followupSubmitButton.addEventListener("click", () => {
      const query = getFollowupInputValue();
      state.followupOpen = false;
      state.followupQuery = "";
      void startInquiry(query).catch(handleInquiryStartError);
    });
  }

  const followupForm = document.getElementById("followup-form");
  if (followupForm) {
    followupForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const query = getFollowupInputValue();
      state.followupOpen = false;
      state.followupQuery = "";
      void startInquiry(query).catch(handleInquiryStartError);
    });
  }

  document.querySelectorAll("[data-preset-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.getAttribute("data-preset-index"));
      selectPresetQuery(index);
    });
  });
}

function getInquiryInputValue() {
  const input = document.getElementById("inquiry-input");
  return input ? input.value : state.query;
}

function getFollowupInputValue() {
  const input = document.getElementById("followup-input");
  return input ? input.value : state.followupQuery;
}

async function loadAppConfig() {
  for (const base of apiBaseCandidates()) {
    try {
      const response = await fetch(`${base}/api/config`);
      if (!response.ok) {
        continue;
      }
      API_BASE = base;
      const payload = await response.json();
      if (payload.default_model) {
        state.modelLabel = payload.default_model;
      }
      if (Array.isArray(payload.supported_models)) {
        state.supportedModels = payload.supported_models;
      }
      if (payload.default_profile) {
        state.defaultProfile = payload.default_profile;
      }
      render();
      return true;
    } catch {
      // Try the next candidate. The UI remains usable without backend config.
    }
  }
  return false;
}

function apiBaseCandidates() {
  return uniqueStrings([
    API_BASE,
    defaultApiBase(),
    "http://127.0.0.1:8000"
  ]).map((base) => base.replace(/\/+$/, ""));
}

function selectPresetQuery(index) {
  const example = curatedExamples[index];
  if (!example) {
    return;
  }
  state.activeSessionId = null;
  state.query = example.title;
  state.submitError = "";
  state.userMenuOpen = false;
  state.followupOpen = false;
  state.followupQuery = "";
  persistActiveSession();
  render({ autoScroll: true });

  const input = document.getElementById("inquiry-input");
  if (input) {
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
  }
  syncSubmitButton();
}

function syncSubmitButton() {
  const button = document.querySelector(".begin-button");
  if (!button) {
    return;
  }
  const isSubmitting = state.sessions.some((session) => session.status === "running");
  button.disabled = !state.query.trim() || isSubmitting;
}

async function startInquiry(rawQuery) {
  const trimmed = rawQuery.trim();
  const isSubmitting = state.sessions.some((session) => session.status === "running");
  if (!trimmed || isSubmitting) {
    return;
  }

  await loadAppConfig();

  const id = createClientSessionId();
  const now = new Date().toISOString();
  const nextSession = {
    id,
    title: titleFromQuery(trimmed),
    query: trimmed,
    createdAt: now,
    updatedAt: now,
    status: "running",
    streamStatus: "Connecting to backend",
    lastHeartbeatAt: now,
    liveReasoning: "",
    liveAnswer: "",
    liveTrace: [],
    liveTimeline: [],
    streamedTextTurns: {},
    streamedReasoningTurns: {},
    rawEvents: [
      createRawTraceRecord("session_start", {
        session_id: id,
        query: trimmed,
        model: state.modelLabel,
        profile: state.defaultProfile,
        pipeline_id: "react_base"
      }, now)
    ]
  };

  state.submitError = "";
  state.query = "";
  state.userMenuOpen = false;
  state.followupOpen = false;
  state.followupQuery = "";
  state.expandedProcesses = {
    ...state.expandedProcesses,
    [id]: true
  };
  state.sessions = [nextSession, ...state.sessions];
  state.activeSessionId = id;
  persistSessions();
  persistActiveSession();
  render({ autoScroll: true });

  const controller = new AbortController();
  activeInquiryControllers.set(id, controller);

  try {
    const response = await fetch(`${API_BASE}/api/inquiries/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        query: trimmed,
        session_id: id,
        model: state.modelLabel,
        profile: state.defaultProfile,
        wall_time_s: INQUIRY_WALL_TIME_S
      })
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed with ${response.status}`);
    }
    if (!response.body) {
      throw new Error("Streaming response body is unavailable");
    }

    await readSSEStream(response, (event) => {
      applyStreamEvent(id, trimmed, event);
    });

    const current = state.sessions.find((session) => session.id === id);
    if (current?.status === "running") {
      throw new Error("Stream ended before the backend returned a final response");
    }
  } catch (error) {
    if (error && typeof error === "object" && error.name === "AbortError") {
      markSessionStopped(id);
      return;
    }
    const message = error instanceof Error ? error.message : "Inquiry failed";
    recordSessionRawEvent(id, "client_error", { message });
    state.submitError = message;
    state.expandedProcesses = {
      ...state.expandedProcesses,
      [id]: false
    };
    updateSession(id, {
      status: "failed",
      updatedAt: new Date().toISOString(),
      response: {
        id,
        session_id: id,
        query: trimmed,
        status: "failed",
        final_answer: "",
        trace: [],
        duration_seconds: 0,
        pipeline_id: "react_base",
        profile: state.defaultProfile,
        model: state.modelLabel,
        error: message
      }
    });
  } finally {
    activeInquiryControllers.delete(id);
  }
}

function createClientSessionId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).slice(2, 12);
  return `session-${timestamp}-${random}`;
}

function handleInquiryStartError(error) {
  const message = error instanceof Error ? error.message : "Inquiry failed to start";
  state.submitError = message;
  render({ autoScroll: false });
}

async function readSSEStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\n\n|\r\n\r\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const event = parseSSEBlock(block);
      if (event) {
        onEvent(event);
      }
    }
  }

  buffer += decoder.decode();
  const finalEvent = parseSSEBlock(buffer);
  if (finalEvent) {
    onEvent(finalEvent);
  }
}

function parseSSEBlock(block) {
  const text = String(block || "").trim();
  if (!text) {
    return null;
  }
  let type = "message";
  const data = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      type = value || "message";
    }
    if (field === "data") {
      data.push(value);
    }
  }
  const rawData = data.join("\n");
  let payload = {};
  if (rawData) {
    try {
      payload = JSON.parse(rawData);
    } catch {
      payload = { message: rawData };
    }
  }
  return { type, payload };
}

function exportActiveSessionTrace() {
  const session = state.sessions.find((item) => item.id === state.activeSessionId);
  if (!session) {
    return;
  }
  const records = buildTraceExportRecords(session);
  const jsonl = `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
  const blob = new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = traceDownloadName(session);
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function buildTraceExportRecords(session) {
  const rawEvents = Array.isArray(session.rawEvents) && session.rawEvents.length
    ? session.rawEvents
    : synthesizeRawTraceRecords(session);
  return [
    ...rawEvents.map((event, index) => ({
      sequence: index + 1,
      ...event
    })),
    createRawTraceRecord("session_snapshot", buildSessionExportSnapshot(session))
  ];
}

function synthesizeRawTraceRecords(session) {
  const records = [
    createRawTraceRecord("session_start", {
      session_id: session.id,
      query: session.query,
      model: session.response?.model || state.modelLabel,
      profile: session.response?.profile || state.defaultProfile,
      pipeline_id: session.response?.pipeline_id || "react_base"
    }, session.createdAt)
  ];
  for (const item of buildTimelineForRender(session)) {
    if (item?.type === "reasoning") {
      records.push(createRawTraceRecord("reasoning", {
        turn: item.turn || "",
        content: item.content || ""
      }, session.updatedAt || session.createdAt));
    }
    if (item?.type === "tool") {
      records.push(createRawTraceRecord("tool", item.step || {}, session.updatedAt || session.createdAt));
    }
  }
  const finalReport = session.response?.final_answer || session.liveAnswer || "";
  if (finalReport) {
    records.push(createRawTraceRecord("final_report", {
      content: finalReport
    }, session.updatedAt || session.createdAt));
  }
  return records;
}

function buildSessionExportSnapshot(session) {
  return {
    session_id: session.id,
    backend_task_id: session.backendTaskId || session.response?.id || "",
    query: session.query,
    title: session.title,
    status: session.status,
    created_at: session.createdAt,
    updated_at: session.updatedAt,
    model: session.response?.model || state.modelLabel,
    profile: session.response?.profile || state.defaultProfile,
    pipeline_id: session.response?.pipeline_id || "react_base",
    stream_status: session.streamStatus || "",
    final_report: session.response?.final_answer || session.liveAnswer || "",
    tool_trace: session.liveTrace?.length ? session.liveTrace : session.response?.trace || [],
    timeline: buildTimelineForRender(session)
  };
}

function createRawTraceRecord(type, payload = {}, timestamp = new Date().toISOString()) {
  return {
    t: type,
    ts: timestamp,
    payload: cloneForJson(payload)
  };
}

function appendRawTraceRecord(records, type, payload = {}, timestamp = new Date().toISOString()) {
  return [
    ...(Array.isArray(records) ? records : []),
    createRawTraceRecord(type, payload, timestamp)
  ];
}

function recordSessionRawEvent(id, type, payload = {}) {
  state.sessions = state.sessions.map((session) =>
    session.id === id
      ? { ...session, rawEvents: appendRawTraceRecord(session.rawEvents, type, payload) }
      : session
  );
  persistSessions();
}

function cloneForJson(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return String(value);
  }
}

function traceDownloadName(session) {
  const timestamp = String(session.createdAt || new Date().toISOString())
    .replace(/[:.]/g, "-")
    .replace(/[^\w-]+/g, "_");
  const slug = String(session.title || session.query || "inquiry")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48) || "inquiry";
  return `apodex-trace-${timestamp}-${slug}.jsonl`;
}

async function stopInquiry(id) {
  const session = state.sessions.find((item) => item.id === id);
  if (!session || session.status !== "running") {
    return;
  }

  mutateSession(id, (draft) => {
    draft.stopRequested = true;
    draft.streamStatus = "Stopping current agent run";
    draft.rawEvents = appendRawTraceRecord(draft.rawEvents, "client_stop_requested", {
      session_id: id
    });
  }, { force: true });

  try {
    await fetch(`${API_BASE}/api/inquiries/${encodeURIComponent(id)}/stop`, {
      method: "POST"
    });
  } catch {
    // Aborting the fetch below still tears down the active stream locally.
  }

  const controller = activeInquiryControllers.get(id);
  if (controller) {
    controller.abort();
  }
  markSessionStopped(id);
}

function markSessionStopped(id) {
  state.expandedProcesses = {
    ...state.expandedProcesses,
    [id]: false
  };
  mutateSession(id, (session) => {
    session.status = "stopped";
    session.stopRequested = true;
    session.streamStatus = "Stopped by user";
    session.updatedAt = new Date().toISOString();
    session.rawEvents = appendRawTraceRecord(session.rawEvents, "client_stopped", {
      session_id: id
    }, session.updatedAt);
    session.liveTimeline = markTimelineComplete(session.liveTimeline);
    if (!session.response) {
      session.response = {
        id: session.backendTaskId || id,
        session_id: id,
        query: session.query,
        status: "failed",
        final_answer: session.liveAnswer || "",
        trace: session.liveTrace || [],
        duration_seconds: 0,
        pipeline_id: "react_base",
        profile: state.defaultProfile,
        model: state.modelLabel,
        error: "Stopped by user"
      };
    }
  }, { force: true });
}

function applyStreamEvent(id, query, event) {
  const payload = event.payload || {};
  mutateSession(id, (session) => {
    const now = new Date().toISOString();
    session.updatedAt = now;
    session.lastHeartbeatAt = session.updatedAt;
    session.rawEvents = appendRawTraceRecord(session.rawEvents, event.type, payload, now);

    if (session.stopRequested && event.type !== "done" && event.type !== "stopped") {
      return;
    }

    if (event.type === "queued") {
      session.streamStatus = "Queued behind another running inquiry";
      return;
    }

    if (event.type === "run_start") {
      session.backendTaskId = payload.id;
      session.streamStatus = "Agent loop started";
      return;
    }

    if (event.type === "loop_start") {
      session.streamStatus = `LLM loop active · ${payload.max_turns || ""} turn budget`.trim();
      return;
    }

    if (event.type === "llm_delta") {
      const turn = String(payload.turn ?? "0");
      if (payload.reasoning_delta) {
        session.liveReasoning = appendMarkdownChunk(session.liveReasoning, payload.reasoning_delta);
        session.liveTimeline = appendTimelineReasoning(
          session.liveTimeline,
          turn,
          payload.reasoning_delta,
          true
        );
        session.streamedReasoningTurns = {
          ...(session.streamedReasoningTurns || {}),
          [turn]: true
        };
      }
      if (payload.delta) {
        session.liveAnswer = appendMarkdownChunk(session.liveAnswer, payload.delta);
        session.streamedTextTurns = {
          ...(session.streamedTextTurns || {}),
          [turn]: true
        };
      }
      session.streamStatus = `Streaming LLM turn ${payload.turn ?? ""}`.trim();
      return;
    }

    if (event.type === "llm_response") {
      const turn = String(payload.turn ?? "0");
      if (payload.reasoning && !(session.streamedReasoningTurns || {})[turn]) {
        session.liveReasoning = appendMarkdownSection(session.liveReasoning, payload.reasoning);
        session.liveTimeline = appendTimelineReasoning(
          session.liveTimeline,
          turn,
          payload.reasoning,
          false
        );
      }
      session.liveTimeline = markTimelineReasoningComplete(session.liveTimeline, turn);
      if (
        payload.text &&
        Number(payload.tool_calls_count || 0) === 0 &&
        !(session.streamedTextTurns || {})[turn]
      ) {
        session.liveAnswer = appendMarkdownSection(session.liveAnswer, payload.text);
      }
      session.streamStatus = Number(payload.tool_calls_count || 0)
        ? `LLM turn ${payload.turn} requested tools`
        : `LLM turn ${payload.turn} produced an answer`;
      return;
    }

    if (event.type === "tool_call" || event.type === "tool_result") {
      session.liveTrace = mergeTraceStep(session.liveTrace || [], payload.step);
      session.liveTimeline = mergeTimelineToolStep(session.liveTimeline, payload.step);
      session.streamStatus = event.type === "tool_call"
        ? `Calling ${payload.step?.tool_name || "tool"}`
        : `Tool ${payload.step?.tool_name || "call"} completed`;
      return;
    }

    if (event.type === "phase_update") {
      session.streamStatus = "Workflow phase completed";
      return;
    }

    if (event.type === "error") {
      session.streamError = payload.message || "Backend stream error";
      session.streamStatus = "Backend returned an error";
      return;
    }

    if (event.type === "stopped") {
      session.status = "stopped";
      session.stopRequested = true;
      session.streamStatus = payload.message || "Stopped by user";
      session.liveTimeline = markTimelineComplete(session.liveTimeline);
      state.expandedProcesses = {
        ...state.expandedProcesses,
        [id]: false
      };
      return;
    }

    if (event.type === "final") {
      const response = {
        ...payload,
        query: payload.query || query,
        trace: Array.isArray(payload.trace) ? payload.trace : []
      };
      session.title = titleFromQuery(response.query || query);
      session.status = response.status === "completed" ? "completed" : "failed";
      session.streamStatus = session.status === "completed" ? "Completed" : "Failed";
      state.expandedProcesses = {
        ...state.expandedProcesses,
        [id]: false
      };
      session.response = response;
      if (!session.liveAnswer && response.final_answer) {
        session.liveAnswer = response.final_answer;
      }
      if (!session.liveTrace?.length && response.trace?.length) {
        session.liveTrace = response.trace;
      }
      if (!session.liveTimeline?.length && response.trace?.length) {
        session.liveTimeline = response.trace.map((step) => ({
          id: `tool-${step.index}`,
          type: "tool",
          step
        }));
      }
      return;
    }

    if (event.type === "done") {
      session.streamStatus = "Stream closed";
    }
  }, { force: event.type === "final" || event.type === "error" || event.type === "done" });
}

function mutateSession(id, mutator, options = {}) {
  let changed = false;
  state.sessions = state.sessions.map((session) => {
    if (session.id !== id) {
      return session;
    }
    changed = true;
    const next = { ...session };
    mutator(next);
    return next;
  });
  if (!changed) {
    return;
  }
  scheduleSessionRender(Boolean(options.force));
}

function scheduleSessionRender(force) {
  if (force) {
    if (streamRenderFrame) {
      cancelAnimationFrame(streamRenderFrame);
      streamRenderFrame = 0;
    }
    lastStreamPersistAt = Date.now();
    persistSessions();
    render({ autoScroll: true });
    return;
  }

  if (streamRenderFrame) {
    return;
  }
  streamRenderFrame = requestAnimationFrame(() => {
    streamRenderFrame = 0;
    const now = Date.now();
    if (now - lastStreamPersistAt >= STREAM_PERSIST_INTERVAL_MS) {
      lastStreamPersistAt = now;
      persistSessions();
    }
    render({ autoScroll: true });
  });
}

function appendMarkdownChunk(current, chunk) {
  if (!chunk) {
    return current || "";
  }
  return `${current || ""}${chunk}`;
}

function appendMarkdownSection(current, section) {
  const clean = String(section || "").trim();
  if (!clean) {
    return current || "";
  }
  return current ? `${current.trimEnd()}\n\n${clean}` : clean;
}

function buildTimelineForRender(session) {
  if (Array.isArray(session.liveTimeline) && session.liveTimeline.length) {
    return session.status === "running"
      ? session.liveTimeline
      : markTimelineComplete(session.liveTimeline);
  }

  const legacyItems = [];
  const trace = Array.isArray(session.liveTrace) && session.liveTrace.length
    ? session.liveTrace
    : session.response?.trace || [];
  if (!trace.length && session.liveReasoning) {
    legacyItems.push({
      id: "legacy-reasoning",
      type: "reasoning",
      turn: "",
      content: session.liveReasoning,
      live: session.status === "running"
    });
  }
  for (const step of trace) {
    legacyItems.push({
      id: `tool-${step.index}`,
      type: "tool",
      step
    });
  }
  return legacyItems;
}

function appendTimelineReasoning(timeline, turn, chunk, live) {
  if (!chunk) {
    return Array.isArray(timeline) ? timeline : [];
  }
  const next = Array.isArray(timeline) ? [...timeline] : [];
  const last = next[next.length - 1];
  if (last?.type === "reasoning" && String(last.turn) === String(turn)) {
    next[next.length - 1] = {
      ...last,
      content: appendMarkdownChunk(last.content, chunk),
      live
    };
    return next;
  }

  next.push({
    id: `reasoning-${turn}-${next.length + 1}`,
    type: "reasoning",
    turn,
    content: chunk,
    live
  });
  return next;
}

function markTimelineReasoningComplete(timeline, turn) {
  if (!Array.isArray(timeline) || !timeline.length) {
    return [];
  }
  return timeline.map((item) =>
    item.type === "reasoning" && String(item.turn) === String(turn)
      ? { ...item, live: false }
      : item
  );
}

function markTimelineComplete(timeline) {
  if (!Array.isArray(timeline) || !timeline.length) {
    return [];
  }
  return timeline.map((item) =>
    item.type === "reasoning" ? { ...item, live: false } : item
  );
}

function mergeTimelineToolStep(timeline, step) {
  if (!step || typeof step !== "object") {
    return Array.isArray(timeline) ? timeline : [];
  }
  const next = Array.isArray(timeline) ? [...timeline] : [];
  const stepIndex = Number(step.index || 0);
  const existingIndex = next.findIndex(
    (item) => item.type === "tool" && Number(item.step?.index) === stepIndex
  );
  const item = {
    id: `tool-${stepIndex || next.length + 1}`,
    type: "tool",
    step
  };
  if (existingIndex === -1) {
    next.push(item);
  } else {
    next[existingIndex] = {
      ...next[existingIndex],
      step: { ...next[existingIndex].step, ...step }
    };
  }
  return next;
}

function mergeTraceStep(trace, step) {
  if (!step || typeof step !== "object") {
    return trace || [];
  }
  const nextTrace = Array.isArray(trace) ? [...trace] : [];
  const stepIndex = Number(step.index || nextTrace.length + 1);
  const normalized = {
    ...step,
    index: stepIndex,
    status: step.status || "completed"
  };
  const existingIndex = nextTrace.findIndex((item) => Number(item.index) === stepIndex);
  if (existingIndex === -1) {
    nextTrace.push(normalized);
  } else {
    nextTrace[existingIndex] = { ...nextTrace[existingIndex], ...normalized };
  }
  return nextTrace.sort((a, b) => Number(a.index || 0) - Number(b.index || 0));
}

function updateSession(id, patch) {
  state.sessions = state.sessions.map((session) =>
    session.id === id ? { ...session, ...patch } : session
  );
  persistSessions();
  render({ autoScroll: true });
}

function readSessions() {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function persistSessions() {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(state.sessions));
}

function persistActiveSession() {
  if (state.activeSessionId) {
    localStorage.setItem(ACTIVE_KEY, state.activeSessionId);
  } else {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

function titleFromQuery(query) {
  return query.length > 54 ? `${query.slice(0, 51).trim()}...` : query;
}

function formatTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatShortTime(value) {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatRelativeAge(value) {
  const then = new Date(value).getTime();
  if (!Number.isFinite(then)) {
    return "0s";
  }
  const seconds = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  return `${Math.floor(minutes / 60)}h`;
}

function syncHeartbeatRenderTimer() {
  const hasRunningSession = state.sessions.some((session) => session.status === "running");
  if (hasRunningSession && !heartbeatRenderTimer) {
    heartbeatRenderTimer = window.setInterval(() => {
      if (state.sessions.some((session) => session.status === "running")) {
        render({ autoScroll: false });
      } else {
        syncHeartbeatRenderTimer();
      }
    }, HEARTBEAT_RENDER_INTERVAL_MS);
  }
  if (!hasRunningSession && heartbeatRenderTimer) {
    window.clearInterval(heartbeatRenderTimer);
    heartbeatRenderTimer = 0;
  }
}

function captionFromQuery(query) {
  const clean = String(query || "").replace(/\s+/g, " ").trim();
  if (!clean) {
    return "New inquiry";
  }
  return clean.length > 58 ? `${clean.slice(0, 55).trim()}...` : clean;
}

function getResearchStats(session, timeline) {
  const items = Array.isArray(timeline) ? timeline : [];
  const turnOrder = [];
  const seenTurns = new Set();
  const reasoningIdsWithoutTurn = [];
  let toolCount = 0;

  for (const item of items) {
    if (item?.type === "tool") {
      toolCount += 1;
    }
    const turn = getTimelineTurn(item);
    if (turn && !seenTurns.has(turn)) {
      seenTurns.add(turn);
      turnOrder.push(turn);
    }
  }

  for (const item of items) {
    if (item?.type !== "reasoning") {
      continue;
    }
    const turn = getTimelineTurn(item);
    if (!turn) {
      reasoningIdsWithoutTurn.push(item.id || `reasoning-${reasoningIdsWithoutTurn.length + 1}`);
    }
  }

  const turnStepMap = Object.fromEntries(turnOrder.map((turn, index) => [turn, index + 1]));
  const reasoningStepMap = {};
  reasoningIdsWithoutTurn.forEach((id, index) => {
    reasoningStepMap[id] = turnOrder.length + index + 1;
  });

  let stepCount = turnOrder.length + reasoningIdsWithoutTurn.length;
  if (!stepCount) {
    const trace = getTraceForStats(session);
    toolCount = toolCount || trace.length;
    const traceTurns = uniqueStrings(trace.map((step) => step?.turn).filter((turn) => turn != null));
    stepCount = traceTurns.length || (session.status === "running" ? 1 : 0);
  }

  return {
    stepCount,
    toolCount,
    turnStepMap,
    reasoningStepMap
  };
}

function getTraceForStats(session) {
  if (Array.isArray(session.liveTrace) && session.liveTrace.length) {
    return session.liveTrace;
  }
  if (Array.isArray(session.response?.trace) && session.response.trace.length) {
    return session.response.trace;
  }
  return [];
}

function getTimelineTurn(item) {
  if (!item || typeof item !== "object") {
    return "";
  }
  const rawTurn = item.type === "tool" ? item.step?.turn : item.turn;
  if (rawTurn == null || rawTurn === "") {
    return "";
  }
  return String(rawTurn);
}

function getStepNumberForTimelineItem(item, researchStats) {
  const turn = getTimelineTurn(item);
  if (turn && researchStats?.turnStepMap?.[turn]) {
    return researchStats.turnStepMap[turn];
  }
  const id = item?.id || "";
  if (id && researchStats?.reasoningStepMap?.[id]) {
    return researchStats.reasoningStepMap[id];
  }
  return researchStats?.stepCount || 1;
}

function getToolArg(step, key) {
  const args = step?.tool_args;
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return "";
  }
  return args[key] ?? "";
}

function normalizeListArg(value) {
  if (value == null || value === "") {
    return [];
  }
  if (Array.isArray(value)) {
    return value.flatMap(normalizeListArg);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) {
      return [];
    }
    if ((trimmed.startsWith("[") && trimmed.endsWith("]")) || (trimmed.startsWith("\"") && trimmed.endsWith("\""))) {
      try {
        return normalizeListArg(JSON.parse(trimmed));
      } catch {
        return [trimmed];
      }
    }
    return [trimmed];
  }
  return [String(value)];
}

function uniqueStrings(values) {
  const seen = new Set();
  const output = [];
  for (const value of values) {
    const clean = cleanUrl(String(value || "").trim());
    if (!clean || seen.has(clean)) {
      continue;
    }
    seen.add(clean);
    output.push(clean);
  }
  return output;
}

function parseSearchResults(observation) {
  const text = String(observation || "");
  if (!text.trim() || text.includes("[ERROR]:")) {
    return [];
  }
  return text
    .split(/(?=\[\d+\]\s*Title:)/g)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => ({
      title: extractLabeledField(block, "Title", ["Date", "Snippet", "URL"]),
      snippet: extractLabeledField(block, "Snippet", ["URL"]),
      url: cleanUrl(extractLabeledField(block, "URL", []))
    }))
    .filter((result) => result.title || result.url);
}

function extractLabeledField(block, label, nextLabels) {
  const start = block.indexOf(`${label}:`);
  if (start === -1) {
    return "";
  }
  const rest = block.slice(start + label.length + 1).trim();
  let end = rest.length;
  for (const nextLabel of nextLabels) {
    const marker = rest.search(new RegExp(`\\s${escapeRegExp(nextLabel)}:`));
    if (marker !== -1) {
      end = Math.min(end, marker);
    }
  }
  const nextResult = rest.search(/\s\[\d+\]\s*Title:/);
  if (nextResult !== -1) {
    end = Math.min(end, nextResult);
  }
  return rest.slice(0, end).trim();
}

function extractUrls(value) {
  const matches = String(value || "").match(/https?:\/\/[^\s<>"')\]]+/g) || [];
  return matches.map(cleanUrl).filter(Boolean);
}

function cleanUrl(value) {
  return String(value || "").trim().replace(/[),.;:]+$/g, "");
}

function domainFromUrl(value) {
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function faviconUrl(value) {
  const domain = domainFromUrl(value);
  return domain
    ? `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=32`
    : "";
}

function renderFavicon(url) {
  const src = faviconUrl(url);
  return src
    ? `<img class="favicon" src="${escapeHtml(src)}" alt="" loading="lazy">`
    : `<span class="favicon favicon-fallback">${icon("globe", 12)}</span>`;
}

function splitReferences(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const referenceIndex = lines.findIndex((line) => /^\s*(?:#{1,6}\s*)?references\s*:?\s*$/i.test(line));
  if (referenceIndex === -1) {
    return {
      body: lines.join("\n"),
      references: []
    };
  }
  return {
    body: lines.slice(0, referenceIndex).join("\n").trim(),
    references: parseReferences(lines.slice(referenceIndex + 1).join("\n"))
  };
}

function parseReferences(value) {
  return String(value || "")
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.replace(/^\s*(?:[-*]\s*)?(?:\[\d+\]|\d+[.)])\s*/, ""))
    .map(parseReferenceLine)
    .filter((reference) => reference.title || reference.url);
}

function parseReferenceLine(line) {
  const link = line.match(/\[([^\]]+)]\((https?:\/\/[^)\s]+)\)/);
  if (link) {
    return {
      title: link[1].trim(),
      url: cleanUrl(link[2])
    };
  }
  const angleLink = line.match(/<\s*(https?:\/\/[^>\s]+)\s*>/);
  const url = angleLink
    ? angleLink[1]
    : (line.match(/https?:\/\/[^\s<>"')\]]+/) || [])[0] || "";
  if (url) {
    const title = line
      .replace(angleLink ? angleLink[0] : url, "")
      .replace(/\s*[-–—:.]\s*$/, "")
      .trim();
    return {
      title: title || domainFromUrl(url),
      url: cleanUrl(url)
    };
  }
  return {
    title: line,
    url: ""
  };
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderMarkdown(value) {
  const text = String(value || "").replace(/\r\n/g, "\n").trim();
  if (!text) {
    return "";
  }

  const lines = text.split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];
  let orderedItems = [];
  let codeLines = [];
  let inCode = false;
  let codeLang = "";

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    html.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br>")}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) {
      return;
    }
    html.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  const flushOrderedList = () => {
    if (!orderedItems.length) {
      return;
    }
    html.push(`<ol>${orderedItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ol>`);
    orderedItems = [];
  };

  const flushCode = () => {
    const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
    html.push(`<pre><code${langClass}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
    codeLang = "";
  };

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const fence = line.match(/^```([\w-]*)\s*$/);
    if (fence) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        flushOrderedList();
        inCode = true;
        codeLang = fence[1] || "";
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (isMarkdownTableStart(lines, lineIndex)) {
      flushParagraph();
      flushList();
      flushOrderedList();
      const table = collectMarkdownTable(lines, lineIndex);
      html.push(renderMarkdownTable(table.rows));
      lineIndex = table.endIndex;
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      flushList();
      flushOrderedList();
      continue;
    }

    if (/^\s*---+\s*$/.test(line)) {
      flushParagraph();
      flushList();
      flushOrderedList();
      html.push("<hr>");
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      flushOrderedList();
      const level = Math.min(6, heading[1].length + 1);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const list = line.match(/^\s*[-*]\s+(.+)$/);
    if (list) {
      flushParagraph();
      flushOrderedList();
      listItems.push(list[1]);
      continue;
    }

    const orderedList = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (orderedList) {
      flushParagraph();
      flushList();
      orderedItems.push(orderedList[1]);
      continue;
    }

    flushList();
    flushOrderedList();
    paragraph.push(line);
  }

  if (inCode) {
    flushCode();
  }
  flushParagraph();
  flushList();
  flushOrderedList();
  return html.join("");
}

function isMarkdownTableStart(lines, index) {
  const current = lines[index] || "";
  const next = lines[index + 1] || "";
  return isTableRow(current) && isTableDivider(next);
}

function collectMarkdownTable(lines, startIndex) {
  const rows = [parseTableRow(lines[startIndex])];
  let index = startIndex + 2;
  while (index < lines.length && isTableRow(lines[index])) {
    rows.push(parseTableRow(lines[index]));
    index += 1;
  }
  return {
    rows,
    endIndex: index - 1
  };
}

function isTableRow(line) {
  const trimmed = String(line || "").trim();
  const pipes = trimmed.match(/(?<!\\)\|/g) || [];
  return pipes.length >= 1;
}

function isTableDivider(line) {
  if (!isTableRow(line)) {
    return false;
  }
  return parseTableRow(line).every((cell) => /^:?-{3,}:?$/.test(cell.trim()));
}

function parseTableRow(line) {
  return String(line || "")
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split(/(?<!\\)\|/)
    .map((cell) => cell.replaceAll("\\|", "|").trim());
}

function renderMarkdownTable(rows) {
  if (!rows.length) {
    return "";
  }
  const [header, ...bodyRows] = rows;
  const columnCount = Math.max(header.length, ...bodyRows.map((row) => row.length));
  const normalizeRow = (row) => Array.from({ length: columnCount }, (_, index) => row[index] || "");
  return `
    <div class="markdown-table-wrap">
      <table>
        <thead>
          <tr>${normalizeRow(header).map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${bodyRows
            .map((row) => `<tr>${normalizeRow(row).map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\[([^\]]+)]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function icon(name, size = 18, extraClass = "") {
  const attrs = `width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" class="${extraClass}"`;
  const paths = {
    ban: '<circle cx="12" cy="12" r="9"/><path d="m5.7 5.7 12.6 12.6"/>',
    "book-open": '<path d="M12 7v14"/><path d="M3 18a3 3 0 0 1 3-3h6V5H6a3 3 0 0 0-3 3z"/><path d="M21 18a3 3 0 0 0-3-3h-6V5h6a3 3 0 0 1 3 3z"/>',
    brain: '<path d="M9.5 2a3 3 0 0 0-3 3v1A3.5 3.5 0 0 0 3 9.5c0 1 .4 1.9 1.1 2.5A3.5 3.5 0 0 0 3 14.5 3.5 3.5 0 0 0 6.5 18H7a3 3 0 0 0 5 2.2"/><path d="M14.5 2a3 3 0 0 1 3 3v1A3.5 3.5 0 0 1 21 9.5c0 1-.4 1.9-1.1 2.5a3.5 3.5 0 0 1 1.1 2.5A3.5 3.5 0 0 1 17.5 18H17a3 3 0 0 1-5 2.2"/><path d="M12 5v15"/>',
    check: '<path d="m20 6-11 11-5-5"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "chevron-up": '<path d="m18 15-6-6-6 6"/>',
    "corner-down-left": '<path d="m9 10-5 5 5 5"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/>',
    "file-output": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h8"/><path d="M14 2v6h6"/><path d="M20 13v5"/><path d="m17 15 3 3 3-3"/>',
    "git-branch": '<circle cx="6" cy="6" r="3"/><circle cx="18" cy="18" r="3"/><path d="M6 9v3a6 6 0 0 0 6 6h3"/>',
    globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 0 20"/><path d="M12 2a15.3 15.3 0 0 0 0 20"/>',
    list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "list-tree": '<path d="M21 12H9"/><path d="M21 6H9"/><path d="M21 18H9"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/><path d="M6 6v12"/>',
    loader: '<path d="M21 12a9 9 0 1 1-6.2-8.56"/>',
    lock: '<rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "message-plus": '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/><path d="M12 7v6"/><path d="M9 10h6"/>',
    "message-square": '<path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/>',
    mic: '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/>',
    paperclip: '<path d="m21.4 11.6-8.5 8.5a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 1 1-2.8-2.8l8.5-8.5"/>',
    plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
    search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',
    settings: '<path d="M12.2 2h-.4l-1 3a7 7 0 0 0-1.7.7l-3-1.3-.3.3-2 3 .2.3 2.4 2a7 7 0 0 0 0 2L4 14l-.2.3 2 3 .3.3 3-1.3c.5.3 1.1.5 1.7.7l1 3h.4l1-3c.6-.2 1.2-.4 1.7-.7l3 1.3.3-.3 2-3-.2-.3-2.4-2a7 7 0 0 0 0-2L20 8l.2-.3-2-3-.3-.3-3 1.3a7 7 0 0 0-1.7-.7z"/><circle cx="12" cy="12" r="3"/>',
    share: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4"/><path d="m15.4 6.5-6.8 4"/>',
    sparkles: '<path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8Z"/>',
    square: '<rect width="14" height="14" x="5" y="5" rx="2"/>',
    tool: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4l-6.1 6.1a2 2 0 1 0 2.8 2.8l6.1-6.1a4 4 0 0 0 5.4-5.4l-2.8 2.8-2.8-2.8z"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    zap: '<path d="M13 2 3 14h8l-1 8 10-12h-8z"/>'
  };
  return `<svg ${attrs}>${paths[name] || paths.sparkles}</svg>`;
}
