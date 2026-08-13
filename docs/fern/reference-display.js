(() => {
  const HOSTS = [
    "http://127.0.0.1:8848",
    "https://host.com",
    "http://host.com",
  ];

  const pathKind = () => {
    const path = window.location.pathname || "";
    if (path === "/sdk" || path.startsWith("/sdk/")) return "sdk";
    if (path === "/api" || path.startsWith("/api/")) return "api";
    if (path.includes("/sdk-reference/")) return "sdk";
    if (path.includes("/http-api/")) return "api";
    return "";
  };

  const setBodyState = () => {
    if (!document.body) return;
    const kind = pathKind();
    document.body.classList.toggle("bcg-sdk-reference", kind === "sdk");
    document.body.classList.toggle("bcg-http-reference", kind === "api");
  };

  const stripHosts = (value) => {
    let result = String(value ?? "");
    for (const host of HOSTS) {
      result = result.split(host).join("");
    }
    return result;
  };

  const rewriteVisibleHosts = (root = document.body) => {
    if (!root || !pathKind()) return;

    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const parent = node.parentElement;
          if (!parent || parent.closest("script, style")) {
            return NodeFilter.FILTER_REJECT;
          }
          const value = node.nodeValue || "";
          return HOSTS.some((host) => value.includes(host))
            ? NodeFilter.FILTER_ACCEPT
            : NodeFilter.FILTER_REJECT;
        },
      }
    );

    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    for (const node of textNodes) {
      node.nodeValue = stripHosts(node.nodeValue);
    }

    root.querySelectorAll?.("input, textarea").forEach((element) => {
      if (HOSTS.some((host) => String(element.value || "").includes(host))) {
        element.value = stripHosts(element.value);
      }
      if (HOSTS.some((host) => String(element.placeholder || "").includes(host))) {
        element.placeholder = stripHosts(element.placeholder);
      }
    });

    // Fern can render the host and relative path as sibling elements.
    // If a leaf element is only the host, remove it entirely.
    root.querySelectorAll?.("*").forEach((element) => {
      if (element.children.length !== 0) return;
      const text = (element.textContent || "").trim();
      if (HOSTS.includes(text)) {
        element.textContent = "";
        element.style.display = "none";
      }
    });
  };

  const flattenSdkCode = () => {
    if (pathKind() !== "sdk") return;
    document.querySelectorAll(".fern-layout-reference pre").forEach((pre) => {
      const wrapper = pre.closest(
        '[class*="code-block"], [class*="CodeBlock"], [class*="code-snippet"], [class*="CodeSnippet"], figure'
      );
      if (wrapper && wrapper.closest(".fern-layout-reference")) {
        wrapper.classList.add("bcg-flat-code-surface");
      }
    });
  };

  const run = () => {
    setBodyState();
    rewriteVisibleHosts(document.body);
    flattenSdkCode();
  };

  const scheduleRun = () => {
    requestAnimationFrame(() => requestAnimationFrame(run));
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleRun, { once: true });
  } else {
    scheduleRun();
  }

  const observer = new MutationObserver(() => scheduleRun());
  const startObserver = () => {
    if (document.body) {
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    }
  };

  if (document.body) startObserver();
  else document.addEventListener("DOMContentLoaded", startObserver, { once: true });

  for (const name of ["pushState", "replaceState"]) {
    const original = history[name];
    history[name] = function (...args) {
      const result = original.apply(this, args);
      scheduleRun();
      return result;
    };
  }

  window.addEventListener("popstate", scheduleRun);
})();
