// Interactive i2 Glossary of Terms
// This script fetches a glossary from a Markdown file,
// parses it, and creates an interactive, searchable interface.

// --- DOM Element References ---
// A collection of all the DOM elements the script interacts with.
const els = {
  search: document.getElementById("searchInput"),
  categorySelect: document.getElementById("categorySelect"),
  filters: document.getElementById("filters"),
  cards: document.getElementById("cards"),
  stats: document.getElementById("stats"),
  status: document.getElementById("status"),
  themeToggle: document.getElementById("themeToggle"),
  modal: document.getElementById("modal"),
  modalDialog: document.querySelector("#modal .modal-dialog"),
  modalOverlay: document.querySelector("#modal .modal-overlay"),
  modalClose: document.getElementById("modalClose"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
};

// --- Application State ---
// A central object to hold the application's state.
const state = {
  entries: [], // All glossary entries parsed from Markdown
  filtered: [], // Entries currently visible after filtering
  search: "", // Current search query
  letter: "All", // Current alphabetical filter
  category: "All", // Current category filter
  modalOpen: false, // Is the modal currently open?
  lastFocus: null, // Element to return focus to after modal closes
  // Preview configuration
  previewLimit: 175, // Max characters for definition previews
  previewOverride: false, // Was the preview limit set via URL/storage?
  // Deep-linking state
  currentSlug: null, // The slug of the term currently in the modal
};

// --- Initialization ---
init();

/**
 * Initializes the application.
 * Sets up the theme, builds UI components, fetches and parses data.
 */
async function init() {
  setupTheme();
  buildFilterBar();
  bindSearch();
  setupModal();

  // Determine preview length (URL param > localStorage > viewport default)
  const cfg = getConfiguredPreviewLimit();
  state.previewLimit = cfg.limit;
  state.previewOverride = cfg.override;

  // If not overridden, adapt preview length on window resize
  if (!state.previewOverride) {
    window.addEventListener(
      "resize",
      debounce(() => {
        const next = computeViewportLimit();
        if (next !== state.previewLimit) {
          state.previewLimit = next;
          render(); // Re-render with new preview length
        }
      }, 150)
    );
  }

  // Fetch, parse, and display the glossary data
  try {
    const mdText = await fetchMarkdown("glossary.md");
    state.entries = parseGlossary(mdText);
    state.filtered = state.entries.slice(); // Initially, all entries are shown
    buildCategoryDropdown();
    render();

    // Handle deep-links on page load and on hash changes
    maybeOpenFromHash();
    window.addEventListener("hashchange", maybeOpenFromHash);
  } catch (err) {
    showStatus(`Failed to load glossary.md: ${err?.message || err}`);
  }
}

/**
 * Sets up the theme (light/dark) based on user preference or saved settings.
 */
function setupTheme() {
  const saved = localStorage.getItem("glossary-theme");
  let initial = saved;
  if (!initial) {
    // If no saved theme, use the system preference
    const prefersLight =
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches;
    initial = prefersLight ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", initial);
  updateThemeToggleLabel(initial);

  // Add click listener to the theme toggle button
  els.themeToggle?.addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("glossary-theme", next); // Save the new theme
    updateThemeToggleLabel(next);
  });
}

/**
 * Updates the title and ARIA label of the theme toggle button.
 * @param {string} theme - The current theme ("light" or "dark").
 */
function updateThemeToggleLabel(theme) {
  if (!els.themeToggle) return;
  const next = theme === "light" ? "dark" : "light";
  els.themeToggle.title = `Switch to ${next} mode`;
  els.themeToggle.setAttribute("aria-label", `Switch to ${next} mode`);
}

/**
 * Creates and appends the A-Z filter buttons to the filter bar.
 */
function buildFilterBar() {
  const frag = document.createDocumentFragment();
  const makeBtn = (label) => {
    const btn = document.createElement("button");
    btn.className = "filter-btn" + (label === state.letter ? " active" : "");
    btn.textContent = label;
    btn.setAttribute("data-letter", label);
    btn.addEventListener("click", () => {
      state.letter = label;
      // Update active class on buttons
      document
        .querySelectorAll(".filter-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyFilters();
    });
    return btn;
  };
  frag.appendChild(makeBtn("All")); // "All" button
  // A-Z buttons
  for (let code = "A".charCodeAt(0); code <= "Z".charCodeAt(0); code++) {
    frag.appendChild(makeBtn(String.fromCharCode(code)));
  }
  els.filters.appendChild(frag);
}

/**
 * Binds the search input to a debounced filter function.
 */
function bindSearch() {
  const onInput = debounce((e) => {
    state.search = e.target.value.trim();
    applyFilters();
  }, 120);
  els.search?.addEventListener("input", onInput);
}

/**
 * Populates the category dropdown with unique categories from the entries.
 */
function buildCategoryDropdown() {
  if (!els.categorySelect) return;
  // Collect unique categories
  const set = new Set();
  for (const e of state.entries) {
    for (const c of e.categories || []) {
      if (c) set.add(c);
    }
  }
  const cats = Array.from(set).sort((a, b) =>
    a.localeCompare(b, undefined, { sensitivity: "base" })
  );

  // Preserve selection if possible
  const prev = state.category || "All";
  els.categorySelect.innerHTML = "";

  // "All categories" option
  const optAll = document.createElement("option");
  optAll.value = "All";
  optAll.textContent = "All categories";
  els.categorySelect.appendChild(optAll);

  // Add each category as an option
  for (const c of cats) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    els.categorySelect.appendChild(opt);
  }
  els.categorySelect.disabled = cats.length === 0;

  // Restore previous selection if it still exists
  if (cats.includes(prev)) {
    els.categorySelect.value = prev;
    state.category = prev;
  } else {
    els.categorySelect.value = "All";
    state.category = "All";
  }

  // Add change listener to the dropdown
  els.categorySelect.addEventListener("change", (e) => {
    state.category = e.target.value || "All";
    applyFilters();
  });
}

/**
 * Fetches the content of a Markdown file.
 * @param {string} filename - The name of the file to fetch.
 * @returns {Promise<string>} The text content of the file.
 */
async function fetchMarkdown(filename) {
  const url = encodeURI("./" + filename); // Encode spaces and special chars
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.text();
}

/**
 * Parses the Markdown text into an array of glossary entries.
 * Assumes H1 headings are terms or letter sections.
 * @param {string} md - The Markdown text.
 * @returns {Array<Object>} An array of entry objects.
 */
function parseGlossary(md) {
  const lines = md.split(/\r?\n/);
  const entries = [];
  let currentTerm = null;
  let buffer = [];

  const flush = () => {
    if (!currentTerm) return;
    // Extract categories from special lines like "Category: AI, Data Science"
    const cleanedLines = [];
    const categories = [];
    for (const rawLine of buffer) {
      const m = rawLine.match(/^\s*(?:Categories?|Tags?)\s*:\s*(.+)$/i);
      if (m) {
        const parts = m[1]
          .split(/[;,|]/)
          .map((s) => s.trim())
          .filter(Boolean);
        for (const c of parts) {
          if (!categories.includes(c)) categories.push(c);
        }
      } else {
        cleanedLines.push(rawLine);
      }
    }
    const definition = cleanedLines.join("\n").trim();
    entries.push({
      term: currentTerm,
      definition,
      letter: (currentTerm[0] || "#").toUpperCase(),
      categories,
    });
    currentTerm = null;
    buffer = [];
  };

  for (const raw of lines) {
    const line = raw.trim();
    const isH1 = line.startsWith("# ");
    if (isH1) {
      flush(); // On new heading, process the previous entry
      const title = line.replace(/^#\s+/, "").trim();
      if (/^[A-Z]$/.test(title)) {
        // Skip single-letter section headings (e.g., "# A")
        continue;
      }
      currentTerm = stripMarkdownInline(title);
      continue;
    }

    if (currentTerm) buffer.push(raw); // Collect lines for the current term
  }
  flush(); // Process the last entry

  // Clean up and sort the entries
  const cleaned = entries.filter((e) => e.term && e.definition);
  cleaned.sort((a, b) =>
    a.term.localeCompare(b.term, undefined, { sensitivity: "base" })
  );
  return cleaned;
}

/**
 * Removes inline Markdown formatting (bold, italics, code).
 * @param {string} str - The string to clean.
 * @returns {string} The cleaned string.
 */
function stripMarkdownInline(str) {
  let s = str
    .replace(/\*\*(.*?)\*\*/g, "$1") // bold
    .replace(/\*(.*?)\*/g, "$1")   // italics
    .replace(/`(.*?)`/g, "$1");      // code
  s = s.replace(/\s+/g, " ").trim(); // Collapse multiple spaces
  return s;
}

/**
 * Escapes HTML special characters to prevent XSS.
 * @param {string} str - The string to escape.
 * @returns {string} The escaped string.
 */
function escapeHTML(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Converts a string with basic Markdown to minimal HTML.
 * Supports paragraphs and inline code.
 * @param {string} text - The text to convert.
 * @returns {string} The resulting HTML string.
 */
function mdToMinimalHTML(text) {
  let html = escapeHTML(text);
  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Simple paragraphs: split on blank lines
  const parts = html.split(/\n\s*\n/);
  return parts.map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

// --- Preview Helpers ---

/**
 * Converts Markdown text to plain text for previews.
 * @param {string} text - The Markdown text.
 * @returns {string} The plain text representation.
 */
function mdToPlainText(text) {
  return text
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\*(.*?)\*/g, "$1")
    .replace(/\r?\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Generates a truncated preview from a definition.
 * @param {string} text - The full definition text.
 * @param {number} [limit=175] - The character limit for the preview.
 * @returns {{preview: string, truncated: boolean}}
 */
function getPreview(text, limit = 175) {
  const plain = mdToPlainText(text);
  if (plain.length <= limit) return { preview: plain, truncated: false };
  let slice = plain.slice(0, limit);
  // Avoid cutting words in half
  const lastSpace = slice.lastIndexOf(" ");
  if (lastSpace > 0) slice = slice.slice(0, lastSpace);
  return { preview: slice.trim() + "…", truncated: true };
}

/**
 * Computes a viewport-aware preview limit.
 * @returns {number} The suggested character limit.
 */
function computeViewportLimit() {
  const isSmall = window.matchMedia && window.matchMedia("(max-width: 480px)").matches;
  return isSmall ? 150 : 175;
}

/**
 * Reads the preview limit from URL parameters or localStorage.
 * Falls back to a viewport-based default.
 * @returns {{limit: number, override: boolean}}
 */
function getConfiguredPreviewLimit() {
  try {
    // Check URL first
    const url = new URL(window.location.href);
    const qp = url.searchParams.get("preview");
    if (qp) {
      const v = parseInt(qp, 10);
      if (Number.isFinite(v) && v >= 50 && v <= 500) {
        return { limit: v, override: true };
      }
    }
  } catch (_) {}
  // Check localStorage next
  const stored = localStorage.getItem("glossary-preview-limit");
  if (stored) {
    const v = parseInt(stored, 10);
    if (Number.isFinite(v) && v >= 50 && v <= 500) {
      return { limit: v, override: true };
    }
  }
  // Fallback to default
  return { limit: computeViewportLimit(), override: false };
}

// --- Modal Logic ---

/**
 * Sets up event listeners for the modal (close, overlay click, ESC key).
 */
function setupModal() {
  if (!els.modal) return;
  els.modalOverlay?.addEventListener("click", (e) => {
    if (e.target?.dataset?.close) closeModal();
  });
  els.modalClose?.addEventListener("click", () => closeModal());

  // Global key handling for ESC and focus trapping
  document.addEventListener("keydown", (e) => {
    if (!state.modalOpen) return;
    if (e.key === "Escape") {
      e.preventDefault();
      closeModal();
      return;
    }
    // Simple focus trap
    if (e.key === "Tab") {
      const focusables = els.modalDialog.querySelectorAll(
        'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'
      );
      const list = Array.from(focusables).filter((el) => !el.hasAttribute("disabled"));
      if (list.length === 0) {
        e.preventDefault();
        els.modalDialog.focus();
        return;
      }
      const first = list[0];
      const last = list[list.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  });
}

/**
 * Opens the modal to display the full definition of an entry.
 * @param {Object} entry - The glossary entry to display.
 * @param {HTMLElement} [triggerEl] - The element that triggered the modal.
 */
function openModal(entry, triggerEl) {
  if (!els.modal) return;
  state.lastFocus = triggerEl || document.activeElement;
  els.modalTitle.textContent = entry.term;
  els.modalBody.innerHTML = mdToMinimalHTML(entry.definition);
  els.modal.hidden = false;
  document.body.classList.add("no-scroll");
  state.modalOpen = true;

  // Update URL hash for deep-linking
  const slug = slugifyTerm(entry.term);
  state.currentSlug = slug;
  const targetHash = `#term-${slug}`;
  if ((location.hash || "") .toLowerCase() !== targetHash) {
    try { history.pushState(null, "", targetHash); } catch (_) { location.hash = targetHash; }
  }

  // Move focus to the modal dialog
  requestAnimationFrame(() => {
    els.modalDialog?.focus();
  });
}

/**
 * Closes the modal and restores the previous state.
 */
function closeModal() {
  if (!els.modal) return;
  els.modal.hidden = true;
  document.body.classList.remove("no-scroll");
  state.modalOpen = false;

  // Restore focus to the element that opened the modal
  if (state.lastFocus && typeof state.lastFocus.focus === "function") {
    state.lastFocus.focus();
  }
  state.lastFocus = null;

  // Clear the URL hash if it matches the closed entry
  if (state.currentSlug) {
    const currentTarget = `#term-${state.currentSlug}`;
    if ((location.hash || "").toLowerCase() === currentTarget) {
      try {
        history.replaceState(null, "", window.location.pathname + window.location.search);
      } catch (_) {
        location.hash = ""; // Fallback
      }
    }
  }
  state.currentSlug = null;
}

/**
 * Applies all active filters (search, letter, category) to the entries.
 */
function applyFilters() {
  const q = state.search.toLowerCase();
  const byLetter = state.letter;
  const byCategory = state.category;

  state.filtered = state.entries.filter((e) => {
    // Letter filter
    const matchesLetter =
      byLetter === "All" ? true : e.term[0]?.toUpperCase() === byLetter;
    if (!matchesLetter) return false;

    // Category filter
    const hasCategory = (e.categories || []);
    const matchesCategory =
      byCategory === "All"
        ? true
        : hasCategory.some(
            (c) => c.toLowerCase() === String(byCategory).toLowerCase()
          );
    if (!matchesCategory) return false;

    // Search query filter (matches term or definition)
    if (!q) return true;
    return (
      e.term.toLowerCase().includes(q) || e.definition.toLowerCase().includes(q)
    );
  });

  render(); // Re-render the UI with filtered results
}

/**
 * Renders the UI based on the current state.
 * Updates stats, and creates/updates entry cards.
 */
function render() {
  // Update stats display
  const total = state.entries.length;
  const shown = state.filtered.length;
  const parts = [];
  parts.push(`${shown.toLocaleString()} of ${total.toLocaleString()} entries`);
  if (state.letter !== "All") parts.push(`Letter: ${state.letter}`);
  if (state.category !== "All") parts.push(`Category: ${escapeHTML(state.category)}`);
  if (state.search) parts.push(`Query: "${escapeHTML(state.search)}"`);
  els.stats.textContent = parts.join(" · ");

  // Render entry cards
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
    const { preview, truncated } = getPreview(e.definition, state.previewLimit);
    const p = document.createElement("p");
    p.textContent = preview; // Use textContent to avoid HTML injection
    body.innerHTML = "";
    body.appendChild(p);

    card.appendChild(h3);
    card.appendChild(meta);
    card.appendChild(body);

    // If definition is truncated, add an "Expand" button
    if (truncated) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "more-btn";
      more.textContent = "Expand";
      more.setAttribute("aria-label", `Expand definition for ${e.term}`);
      more.setAttribute("title", `Expand definition for ${e.term}`);
      more.setAttribute("aria-haspopup", "dialog");
      more.setAttribute("aria-controls", "modal");
      more.dataset.term = e.term;

      // Add chevron icon
      const svgNS = "http://www.w3.org/2000/svg";
      const icon = document.createElementNS(svgNS, "svg");
      icon.setAttribute("aria-hidden", "true");
      icon.setAttribute("width", "14");
      icon.setAttribute("height", "14");
      icon.setAttribute("viewBox", "0 0 24 24");
      const path = document.createElementNS(svgNS, "path");
      path.setAttribute("fill", "currentColor");
      path.setAttribute("d", "M6 9l6 6 6-6"); // Down chevron
      icon.appendChild(path);

      more.appendChild(icon);
      more.addEventListener("click", () => openModal(e, more));
      card.appendChild(more);
    }
    frag.appendChild(card);
  }
  els.cards.appendChild(frag);

  // Show "No results" message if needed
  if (!shown) {
    showStatus(
      "No entries match your filters. Try a different letter or search query."
    );
  } else {
    hideStatus();
  }
}

/**
 * Shows a message in the status area.
 * @param {string} msg - The message to display.
 */
function showStatus(msg) {
  if (!els.status) return;
  els.status.hidden = false;
  els.status.textContent = msg;
}

/**
 * Hides the status message area.
 */
function hideStatus() {
  if (!els.status) return;
  els.status.hidden = true;
  els.status.textContent = "";
}

/**
 * Creates a debounced function that delays invoking `fn`
 * until `wait` milliseconds have passed since the last time it was invoked.
 * @param {Function} fn - The function to debounce.
 * @param {number} [wait=150] - The number of milliseconds to delay.
 * @returns {Function} The new debounced function.
 */
function debounce(fn, wait = 150) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn.apply(null, args), wait);
  };
}

// --- Deep-linking Helpers ---

/**
 * Converts a term string into a URL-friendly slug.
 * @param {string} str - The term to slugify.
 * @returns {string} The slugified term.
 */
function slugifyTerm(str) {
  return String(str)
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-") // Replace non-alphanumeric with -
    .replace(/^-+|-+$/g, "");   // Remove leading/trailing -
}

/**
 * Finds a glossary entry by its slug.
 * @param {string} slug - The slug to search for.
 * @returns {Object|null} The matching entry or null.
 */
function findEntryBySlug(slug) {
  return state.entries.find((e) => slugifyTerm(e.term) === slug) || null;
}

/**
 * Checks the URL hash on load or change, and opens the corresponding
 * modal if it matches a term slug.
 */
function maybeOpenFromHash() {
  const h = (location.hash || "").toLowerCase();
  const m = h.match(/^#term-(.+)$/);
  if (m && m[1]) {
    const slug = m[1];
    if (state.modalOpen && state.currentSlug === slug) return; // Modal for this term is already open
    const entry = findEntryBySlug(slug);
    if (entry) {
      openModal(entry);
    }
    return;
  }
  // If no valid hash, close the modal if it's open
  if (state.modalOpen) closeModal();
}
