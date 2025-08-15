// Interactive i2 Glossary of Terms
// - Parses glossary.md into entries
// - Provides search and A–Z filtering
// - Clean, modern UI with theme toggle

const els = {
  search: document.getElementById("searchInput"),
  filters: document.getElementById("filters"),
  cards: document.getElementById("cards"),
  stats: document.getElementById("stats"),
  status: document.getElementById("status"),
  themeToggle: document.getElementById("themeToggle"),
};

const state = {
  entries: [],
  filtered: [],
  search: "",
  letter: "All",
};

init();

async function init() {
  setupTheme();
  buildFilterBar();
  bindSearch();
  try {
    const mdText = await fetchMarkdown("glossary.md");
    state.entries = parseGlossary(mdText);
    state.filtered = state.entries.slice();
    render();
  } catch (err) {
    showStatus(`Failed to load glossary.md: ${err?.message || err}`);
  }
}

function setupTheme() {
  const saved = localStorage.getItem("glossary-theme");
  let initial = saved;
  if (!initial) {
    const prefersLight =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches;
    initial = prefersLight ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", initial);
  updateThemeToggleLabel(initial);

  els.themeToggle?.addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("glossary-theme", next);
    updateThemeToggleLabel(next);
  });
}

function updateThemeToggleLabel(theme) {
  if (!els.themeToggle) return;
  const next = theme === "light" ? "dark" : "light";
  els.themeToggle.title = `Switch to ${next} mode`;
  els.themeToggle.setAttribute("aria-label", `Switch to ${next} mode`);
}

function buildFilterBar() {
  const frag = document.createDocumentFragment();
  const makeBtn = (label) => {
    const btn = document.createElement("button");
    btn.className = "filter-btn" + (label === state.letter ? " active" : "");
    btn.textContent = label;
    btn.setAttribute("data-letter", label);
    btn.addEventListener("click", () => {
      state.letter = label;
      document
        .querySelectorAll(".filter-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyFilters();
    });
    return btn;
  };
  frag.appendChild(makeBtn("All"));
  for (let code = "A".charCodeAt(0); code <= "Z".charCodeAt(0); code++) {
    frag.appendChild(makeBtn(String.fromCharCode(code)));
  }
  els.filters.appendChild(frag);
}

function bindSearch() {
  const onInput = debounce((e) => {
    state.search = e.target.value.trim();
    applyFilters();
  }, 120);
  els.search?.addEventListener("input", onInput);
}

async function fetchMarkdown(filename) {
  // Encode spaces and special chars
  const url = encodeURI("./" + filename);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.text();
}

function parseGlossary(md) {
  const lines = md.split(/\r?\n/);
  const entries = [];
  let currentLetter = null;
  let currentTerm = null;
  let buffer = [];

  const flush = () => {
    if (!currentTerm) return;
    const definition = buffer.join("\n").trim();
    entries.push({
      term: currentTerm,
      definition,
      letter: (currentTerm[0] || "#").toUpperCase(),
    });
    currentTerm = null;
    buffer = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    const isH1 = line.startsWith("# ");
    if (isH1) {
      // On new heading, flush previous entry
      flush();
      const title = line.replace(/^#\s+/, "").trim();
      // Section headings are single letters like "A", "B"...
      if (/^[A-Z]$/.test(title)) {
        currentLetter = title;
        continue;
      }
      // Otherwise, it's a term heading
      const clean = stripMarkdownInline(title);
      currentTerm = clean;
      continue;
    }

    if (currentTerm) buffer.push(raw); // keep original spacing
  }
  flush();

  // Remove empty or malformed
  const cleaned = entries.filter((e) => e.term && e.definition);
  // Sort by term
  cleaned.sort((a, b) =>
    a.term.localeCompare(b.term, undefined, { sensitivity: "base" })
  );
  return cleaned;
}

function stripMarkdownInline(str) {
  // Remove bold/italics markers, keep content
  let s = str
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/`(.*?)`/g, "$1");
  // Collapse multiple spaces
  s = s.replace(/\s+/g, " ").trim();
  return s;
}

function escapeHTML(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function mdToMinimalHTML(text) {
  // Escape first
  let html = escapeHTML(text);
  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Simple paragraphs: split on blank lines
  const parts = html.split(/\n\s*\n/);
  return parts.map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function applyFilters() {
  const q = state.search.toLowerCase();
  const byLetter = state.letter;
  state.filtered = state.entries.filter((e) => {
    const matchesLetter =
      byLetter === "All" ? true : e.term[0]?.toUpperCase() === byLetter;
    if (!matchesLetter) return false;
    if (!q) return true;
    return (
      e.term.toLowerCase().includes(q) || e.definition.toLowerCase().includes(q)
    );
  });
  render();
}

function render() {
  // Stats
  const total = state.entries.length;
  const shown = state.filtered.length;
  const parts = [];
  parts.push(`${shown.toLocaleString()} of ${total.toLocaleString()} entries`);
  if (state.letter !== "All") parts.push(`Letter: ${state.letter}`);
  if (state.search) parts.push(`Query: "${escapeHTML(state.search)}"`);
  els.stats.textContent = parts.join(" · ");

  // Cards
  els.cards.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const e of state.filtered) {
    const card = document.createElement("article");
    card.className = "card";

    const h3 = document.createElement("h3");
    h3.textContent = e.term;

    const meta = document.createElement("div");
    meta.className = "meta";
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = e.term[0]?.toUpperCase() || "#";
    meta.appendChild(badge);

    const body = document.createElement("div");
    body.className = "body";
    body.innerHTML = mdToMinimalHTML(e.definition);

    card.appendChild(h3);
    card.appendChild(meta);
    card.appendChild(body);
    frag.appendChild(card);
  }
  els.cards.appendChild(frag);

  // Empty state
  if (!shown) {
    showStatus(
      "No entries match your filters. Try a different letter or search query."
    );
  } else {
    hideStatus();
  }
}

function showStatus(msg) {
  if (!els.status) return;
  els.status.hidden = false;
  els.status.textContent = msg;
}

function hideStatus() {
  if (!els.status) return;
  els.status.hidden = true;
  els.status.textContent = "";
}

function debounce(fn, wait = 150) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(null, args), wait);
  };
}
