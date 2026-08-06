(() => {
  "use strict";
  const PAGES = window.BCG_PAGES;
  const NAV = window.BCG_NAV;
  const ORDER = window.BCG_ORDER;
  const ICON_PATHS = {"brain":"<path d=\"M9.5 4.5A3 3 0 0 0 4 6a3 3 0 0 0 .7 5.9A3 3 0 0 0 7 17.5h2.5V4.5Z\"/><path d=\"M14.5 4.5A3 3 0 0 1 20 6a3 3 0 0 1-.7 5.9 3 3 0 0 1-2.3 5.6h-2.5V4.5Z\"/><path d=\"M9.5 8H7.8M14.5 8h1.7M9.5 12H7M14.5 12h2.5M9.5 16H8M14.5 16h1.5\"/>","gauge":"<path d=\"M4 16a8 8 0 1 1 16 0\"/><path d=\"m12 16 4-5\"/><path d=\"M7 16h10\"/>","gauge-high":"<path d=\"M4 16a8 8 0 1 1 16 0\"/><path d=\"m12 16 5-6\"/><path d=\"M7 16h10\"/>","highlighter":"<path d=\"m9 11-5 5v4h4l5-5\"/><path d=\"m13 15 7-7-4-4-7 7 4 4Z\"/><path d=\"M4 20h16\"/>","clock":"<circle cx=\"12\" cy=\"12\" r=\"8\"/><path d=\"M12 7v5l3 2\"/>","clock-rotate-left":"<path d=\"M4 8V4m0 0h4M4 4l3 3\"/><path d=\"M5.5 9A8 8 0 1 1 4 14\"/><path d=\"M12 8v4l3 2\"/>","diagram-project":"<rect x=\"3\" y=\"4\" width=\"6\" height=\"5\" rx=\"1\"/><rect x=\"15\" y=\"4\" width=\"6\" height=\"5\" rx=\"1\"/><rect x=\"9\" y=\"15\" width=\"6\" height=\"5\" rx=\"1\"/><path d=\"M6 9v2h12V9M12 11v4\"/>","share-nodes":"<circle cx=\"5\" cy=\"12\" r=\"2.5\"/><circle cx=\"18\" cy=\"6\" r=\"2.5\"/><circle cx=\"18\" cy=\"18\" r=\"2.5\"/><path d=\"m7.2 10.9 8.5-3.8M7.2 13.1l8.5 3.8\"/>","circle-nodes":"<circle cx=\"12\" cy=\"12\" r=\"3\"/><circle cx=\"5\" cy=\"6\" r=\"2\"/><circle cx=\"19\" cy=\"6\" r=\"2\"/><circle cx=\"5\" cy=\"18\" r=\"2\"/><circle cx=\"19\" cy=\"18\" r=\"2\"/><path d=\"m7 7.5 3 2.5m4 0 3-2.5m-10 9 3-2.5m4 0 3 2.5\"/>","code-merge":"<circle cx=\"7\" cy=\"5\" r=\"2\"/><circle cx=\"17\" cy=\"5\" r=\"2\"/><circle cx=\"12\" cy=\"19\" r=\"2\"/><path d=\"M7 7v3c0 4 5 3 5 7M17 7v3c0 4-5 3-5 7\"/>","magnifying-glass-chart":"<circle cx=\"10\" cy=\"10\" r=\"6\"/><path d=\"m14.5 14.5 5 5\"/><path d=\"M7 12V9m3 3V7m3 5v-2\"/>","terminal":"<rect x=\"3\" y=\"4\" width=\"18\" height=\"16\" rx=\"2\"/><path d=\"m7 9 3 3-3 3M12 15h5\"/>","python":"<path d=\"M8 4h5a3 3 0 0 1 3 3v3H9a3 3 0 0 0-3 3v1H4V9a3 3 0 0 1 3-3h5\"/><path d=\"M16 20h-5a3 3 0 0 1-3-3v-3h7a3 3 0 0 0 3-3v-1h2v5a3 3 0 0 1-3 3h-5\"/><circle cx=\"9\" cy=\"7\" r=\".7\" fill=\"currentColor\" stroke=\"none\"/><circle cx=\"15\" cy=\"17\" r=\".7\" fill=\"currentColor\" stroke=\"none\"/>","server":"<rect x=\"4\" y=\"4\" width=\"16\" height=\"6\" rx=\"1.5\"/><rect x=\"4\" y=\"14\" width=\"16\" height=\"6\" rx=\"1.5\"/><path d=\"M8 7h.01M8 17h.01M12 7h5M12 17h5\"/>","eye":"<path d=\"M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z\"/><circle cx=\"12\" cy=\"12\" r=\"2.5\"/>","feather":"<path d=\"M20 4C12 4 6 9 5 17l-2 3 3-2c8-1 13-7 14-14Z\"/><path d=\"m6 18 9-9M9 15h4v-4\"/>","quote-left":"<path d=\"M7 17H4a2 2 0 0 1-2-2v-3c0-4 2-7 6-8v3c-2 .7-3 2-3 4h2v6Zm10 0h-3a2 2 0 0 1-2-2v-3c0-4 2-7 6-8v3c-2 .7-3 2-3 4h2v6Z\"/>","book-open":"<path d=\"M3 5.5A3.5 3.5 0 0 1 6.5 4H11v15H6.5A3.5 3.5 0 0 0 3 20.5v-15Z\"/><path d=\"M21 5.5A3.5 3.5 0 0 0 17.5 4H13v15h4.5a3.5 3.5 0 0 1 3.5 1.5v-15Z\"/>","route":"<circle cx=\"5\" cy=\"5\" r=\"2\"/><circle cx=\"19\" cy=\"19\" r=\"2\"/><path d=\"M7 5h5a3 3 0 0 1 3 3v1a3 3 0 0 1-3 3H9a3 3 0 0 0-3 3v2a2 2 0 0 0 2 2h9\"/>","list":"<path d=\"M9 6h11M9 12h11M9 18h11\"/><circle cx=\"4.5\" cy=\"6\" r=\"1\"/><circle cx=\"4.5\" cy=\"12\" r=\"1\"/><circle cx=\"4.5\" cy=\"18\" r=\"1\"/>","shield-check":"<path d=\"M12 3 20 6v5c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-3Z\"/><path d=\"m8.5 12 2.2 2.2 4.8-5\"/>","download":"<path d=\"M12 3v12m-4-4 4 4 4-4\"/><path d=\"M4 19h16\"/>","rocket":"<path d=\"M14 4c3-1 5-1 6-1 0 1 0 3-1 6l-6 6-4-4 5-7Z\"/><path d=\"M9 11 5 12l-2 3 5 1M13 15l-1 4-3 2-1-5\"/><circle cx=\"16\" cy=\"7\" r=\"1.5\"/>","boxes":"<rect x=\"3\" y=\"4\" width=\"8\" height=\"7\" rx=\"1\"/><rect x=\"13\" y=\"4\" width=\"8\" height=\"7\" rx=\"1\"/><rect x=\"8\" y=\"13\" width=\"8\" height=\"7\" rx=\"1\"/>","plug":"<path d=\"M8 3v5m8-5v5M6 8h12v2a6 6 0 0 1-6 6v5M9 21h6\"/>","chart-network":"<path d=\"M4 19V5M4 19h16\"/><circle cx=\"8\" cy=\"14\" r=\"1.5\"/><circle cx=\"13\" cy=\"9\" r=\"1.5\"/><circle cx=\"18\" cy=\"12\" r=\"1.5\"/><path d=\"m9.2 13 2.6-2.8m2.6-.3 2.3 1.3\"/>","activity":"<path d=\"M3 12h4l2-5 4 10 2-5h6\"/>","play":"<circle cx=\"12\" cy=\"12\" r=\"9\"/><path d=\"m10 8 6 4-6 4V8Z\"/>","bot":"<rect x=\"5\" y=\"7\" width=\"14\" height=\"11\" rx=\"3\"/><path d=\"M12 3v4M8 12h.01M16 12h.01M9 15h6\"/>","file-input":"<path d=\"M6 3h8l4 4v14H6V3Z\"/><path d=\"M14 3v5h5M9 14h6m-3-3 3 3-3 3\"/>","file-output":"<path d=\"M6 3h8l4 4v14H6V3Z\"/><path d=\"M14 3v5h5M15 14H9m3-3-3 3 3 3\"/>","sliders-horizontal":"<path d=\"M4 7h10m4 0h2M4 17h2m4 0h10M14 5v4M6 15v4\"/>","key":"<circle cx=\"8\" cy=\"12\" r=\"4\"/><path d=\"m12 12 8-8m-3 3 3 3m-6 0 3 3\"/>","cloud":"<path d=\"M6 18h11a4 4 0 0 0 .6-8A6 6 0 0 0 6.3 8.5 4.8 4.8 0 0 0 6 18Z\"/>","wrench":"<path d=\"M14 6a5 5 0 0 0-6.5 6.5L3 17l4 4 4.5-4.5A5 5 0 0 0 18 10l-3 3-4-4 3-3Z\"/>","folder-tree":"<path d=\"M3 5h7l2 2h9v12H3V5Z\"/><path d=\"M8 10v6m0-3h5m0 0v3m0-3h4\"/>"};
  const iconSvg = name => `
    <svg class="title-icon-svg" viewBox="0 0 24 24" aria-hidden="true"
      fill="none" stroke="currentColor" stroke-width="1.8"
      stroke-linecap="round" stroke-linejoin="round">
      ${ICON_PATHS[name] || ICON_PATHS["circle-nodes"]}
    </svg>`;
  const els = {
    body: document.body,
    sidebar: document.getElementById("sidebarNav"),
    article: document.getElementById("article"),
    toc: document.getElementById("toc"),
    searchOverlay: document.getElementById("searchOverlay"),
    searchInput: document.getElementById("searchInput"),
    searchResults: document.getElementById("searchResults"),
    searchTrigger: document.getElementById("searchTrigger"),
  };

  const routeState = () => {
    const raw = location.hash.replace(/^#\/?/, "");
    const [pathPart, queryPart = ""] = raw.split("?");
    const path = pathPart.replace(/\/$/, "");
    const params = new URLSearchParams(queryPart);
    return {
      slug: PAGES[path] ? path : "index",
      section: params.get("section") || "",
    };
  };
  const cleanRoute = () => routeState().slug;
  const activateToc = headingId => {
    const links = [...document.querySelectorAll(".toc a[data-heading]")];
    links.forEach(link => {
      const active = link.dataset.heading === headingId;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  };

  const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[ch]);

  function buildSidebar() {
    els.sidebar.innerHTML = NAV.map(tab => {
      const groups = tab.groups.map(group => `
        <div class="nav-group-label">${esc(group.name)}</div>
        ${group.pages.map(slug => `
          <a class="nav-link" data-route="${esc(slug)}" href="#/${esc(slug)}">
            ${esc(PAGES[slug].title)}
          </a>`).join("")}
      `).join("");
      return `<div class="nav-tab">${esc(tab.tab)}</div>${groups}`;
    }).join("");
  }

  function pageNav(slug) {
    const i = ORDER.indexOf(slug);
    const prev = i > 0 ? PAGES[ORDER[i - 1]] : null;
    const next = i >= 0 && i < ORDER.length - 1 ? PAGES[ORDER[i + 1]] : null;
    return `
      <footer class="page-footer">
        ${prev ? `<a class="page-nav" href="#/${prev.slug}"><span>Previous</span><strong>← ${esc(prev.title)}</strong></a>` : "<span></span>"}
        ${next ? `<a class="page-nav next" href="#/${next.slug}"><span>Next</span><strong>${esc(next.title)} →</strong></a>` : ""}
      </footer>`;
  }

  function buildToc(page) {
    if (!page.headings.length) {
      els.toc.innerHTML = "";
      els.toc.hidden = true;
      return;
    }
    els.toc.hidden = false;
    els.toc.innerHTML = `<div class="toc-title">On this page</div>` +
      page.headings.map(h => `<a class="level-${h.level}" href="#/${page.slug}?section=${encodeURIComponent(h.id)}" data-heading="${h.id}">${esc(h.text)}</a>`).join("");

    const links = [...els.toc.querySelectorAll("a[data-heading]")];
    const currentSection = routeState().section;
    activateToc(currentSection || page.headings[0].id);

    links.forEach(link => {
      link.addEventListener("click", event => {
        event.preventDefault();
        const headingId = link.dataset.heading;
        const target = document.getElementById(headingId);
        if (!target) return;

        activateToc(headingId);
        const nextUrl = `#/${page.slug}?section=${encodeURIComponent(headingId)}`;
        history.pushState({section: headingId}, "", nextUrl);

        const root = document.documentElement;
        const previousBehavior = root.style.scrollBehavior;
        root.style.scrollBehavior = "auto";
        target.scrollIntoView({block: "start", behavior: "auto"});
        requestAnimationFrame(() => {
          root.style.scrollBehavior = previousBehavior;
        });
      });
    });
  }

  function render() {
    const slug = cleanRoute();
    const page = PAGES[slug];
    document.title = `${page.title} · BCG Documentation`;
    els.article.innerHTML = `
      <header class="page-head">
        <div class="page-title-row">
          <span class="page-title-icon" aria-hidden="true">${iconSvg(page.icon)}</span>
          <h1>${esc(page.title)}</h1>
        </div>
        ${page.description ? `<p class="page-description">${esc(page.description)}</p>` : ""}
      </header>
      <div class="article-body">${page.html}</div>
      ${pageNav(slug)}
    `;
    document.querySelectorAll(".nav-link").forEach(a => {
      a.classList.toggle("active", a.dataset.route === slug);
    });
    const active = document.querySelector(`.nav-link[data-route="${CSS.escape(slug)}"]`);
    if (active) active.scrollIntoView({block:"nearest"});
    buildToc(page);
    bindArticleInteractions();
    els.body.classList.remove("menu-open");
    const {section} = routeState();
    if (section) {
      requestAnimationFrame(() => {
        const target = document.getElementById(section);
        const root = document.documentElement;
        const previousBehavior = root.style.scrollBehavior;
        root.style.scrollBehavior = "auto";
        if (target) target.scrollIntoView({block:"start", behavior:"auto"});
        else window.scrollTo({top:0, behavior:"auto"});
        requestAnimationFrame(() => {
          root.style.scrollBehavior = previousBehavior;
        });
      });
    } else {
      window.scrollTo({top:0, behavior:"auto"});
    }
    setTimeout(initHeadingObserver, 30);
  }

  function bindArticleInteractions() {
    document.querySelectorAll(".offline-tabs").forEach(group => {
      const buttons = [...group.querySelectorAll(".tab-button")];
      const panels = [...group.querySelectorAll(".tab-panel")];
      buttons.forEach(button => button.addEventListener("click", () => {
        const index = button.dataset.tabIndex;
        buttons.forEach(b => b.classList.toggle("active", b === button));
        panels.forEach(p => p.classList.toggle("active", p.dataset.tabPanel === index));
      }));
    });

    document.querySelectorAll(".copy-code").forEach(button => {
      button.addEventListener("click", async () => {
        const code = button.closest(".code-shell").querySelector("code").innerText;
        try {
          await navigator.clipboard.writeText(code);
        } catch {
          const area = document.createElement("textarea");
          area.value = code;
          document.body.appendChild(area);
          area.select();
          document.execCommand("copy");
          area.remove();
        }
        button.textContent = "Copied";
        button.classList.add("copied");
        setTimeout(() => {
          button.textContent = "Copy";
          button.classList.remove("copied");
        }, 1100);
      });
    });

    document.querySelectorAll("[data-pressable]").forEach(card => {
      card.addEventListener("pointerdown", () => card.classList.add("is-selected"));
      ["pointerup", "pointerleave"].forEach(name =>
        card.addEventListener(name, () => setTimeout(() => card.classList.remove("is-selected"), 120))
      );
    });

    const motif = document.querySelector(".belief-motif");
    if (motif) {
      const items = [...motif.querySelectorAll(".motif-inspectable")];
      const readoutTitle = motif.querySelector("#motifReadoutTitle");
      const readout = motif.querySelector("#motifReadout");
      const choose = item => {
        items.forEach(candidate => {
          const selected = candidate === item;
          candidate.classList.toggle("selected", selected);
          candidate.setAttribute("aria-pressed", selected ? "true" : "false");
        });
        motif.dataset.activeKind = item.dataset.kind || "evidence";
        readoutTitle.textContent = item.dataset.title || "";
        readout.textContent = item.dataset.description || "";
      };
      items.forEach(item => item.addEventListener("click", () => choose(item)));
      const initial = items.find(item => item.dataset.default === "true") || items[0];
      if (initial) choose(initial);
    }

    document.querySelectorAll(".article-body a[href^='#/']").forEach(link => {
      link.addEventListener("click", () => els.body.classList.remove("menu-open"));
    });
  }

  let observer;
  function initHeadingObserver() {
    if (observer) observer.disconnect();
    const links = [...document.querySelectorAll(".toc a[data-heading]")];
    const targets = links.map(link => document.getElementById(link.dataset.heading)).filter(Boolean);
    if (!targets.length) return;

    observer = new IntersectionObserver(entries => {
      const visible = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

      if (visible.length) {
        activateToc(visible[0].target.id);
        return;
      }

      const passed = targets
        .filter(target => target.getBoundingClientRect().top <= 112)
        .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top);

      if (passed.length) activateToc(passed[0].id);
    }, {
      rootMargin: "-92px 0px -68% 0px",
      threshold: [0, 0.2, 1],
    });

    targets.forEach(target => observer.observe(target));
  }

  function openSearch() {
    els.searchOverlay.classList.add("open");
    els.searchOverlay.setAttribute("aria-hidden", "false");
    els.searchTrigger.classList.add("clicked");
    els.searchInput.value = "";
    search("");
    setTimeout(() => els.searchInput.focus(), 10);
  }

  function closeSearch() {
    els.searchOverlay.classList.remove("open");
    els.searchOverlay.setAttribute("aria-hidden", "true");
    els.searchTrigger.classList.remove("clicked");
  }

  function search(query) {
    const q = query.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    const ranked = ORDER.map(slug => {
      const p = PAGES[slug];
      const title = p.title.toLowerCase();
      const desc = p.description.toLowerCase();
      const body = p.text.toLowerCase();
      let score = 0;
      for (const term of terms) {
        if (title.includes(term)) score += 12;
        if (desc.includes(term)) score += 5;
        const count = body.split(term).length - 1;
        score += Math.min(count, 6);
      }
      return {p, score};
    }).filter(item => !q || item.score > 0)
      .sort((a,b) => b.score - a.score)
      .slice(0, 12);

    if (!ranked.length) {
      els.searchResults.innerHTML = `<div class="search-empty">No matching documentation page.</div>`;
      return;
    }
    els.searchResults.innerHTML = ranked.map(({p}) => `
      <a class="search-result" href="#/${p.slug}">
        <strong>${esc(p.title)}</strong>
        <p>${esc(p.description || p.text.slice(0, 150))}</p>
      </a>`).join("");
    els.searchResults.querySelectorAll("a").forEach(a => a.addEventListener("click", closeSearch));
  }

  buildSidebar();
  window.addEventListener("hashchange", render);
  els.searchTrigger.addEventListener("click", openSearch);
  document.getElementById("searchClose").addEventListener("click", closeSearch);
  els.searchOverlay.addEventListener("click", e => { if (e.target === els.searchOverlay) closeSearch(); });
  els.searchInput.addEventListener("input", e => search(e.target.value));
  document.getElementById("mobileMenu").addEventListener("click", () => els.body.classList.toggle("menu-open"));
  document.getElementById("printPage").addEventListener("click", () => window.print());
  document.addEventListener("keydown", e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); openSearch();
    } else if (e.key === "/" && !["INPUT","TEXTAREA"].includes(document.activeElement.tagName)) {
      e.preventDefault(); openSearch();
    } else if (e.key === "Escape") {
      closeSearch(); els.body.classList.remove("menu-open");
    }
  });
  if (!location.hash) history.replaceState(null, "", "#/index");
  render();
})();
