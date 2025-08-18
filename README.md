# Interactive i2 Glossary of Terms

A clean, modern, searchable glossary UI that renders directly from `glossary.md`. Includes full‑text search, A–Z filters, category filtering, a theme toggle, and deep‑linkable term modals.

## Features
- **Search**: Instant search across terms and definitions
- **A–Z filters**: Filter by starting letter or show all
- **Category dropdown**: Populated from each term’s optional `Categories:` line
- **Theme toggle**: Light/dark via the header button (`data-theme` attribute)
- **Responsive header**: Search, category select, and theme button stay on one line on small screens
- **Details modal + deep links**: Expand long definitions in a modal; each term has a URL hash like `#term-your-slug`
- **Stats bar**: Shows counts and active filters
- **Configurable previews**: Control card preview length via `?preview=150` or localStorage (`glossary-preview-limit`)
- **Zero build**: Pure static files; update by editing `glossary.md`

## Quick start
1. Ensure you’re in the `glossary/` directory.
2. Start a local server (choose one):
   - Python 3: `python3 -m http.server 8000`
   - VS Code (Live Server extension):
     - Install "Live Server" by Ritwick Dey from the Extensions marketplace
     - Open the `glossary/` folder in VS Code
     - Right‑click `index.html` → "Open with Live Server" (or click the "Go Live" button in the status bar)
     - The app will open automatically (default: http://127.0.0.1:5500)
3. If using Python, open http://127.0.0.1:8000 in your browser.

Files:
- `index.html` – App shell (header, search, filters, content area)
- `styles.css` – Modern responsive styles; attribute‑based light/dark theming
- `app.js` – Markdown fetch/parse, rendering, search, A–Z and category filters, modal, deep‑links, stats
- `glossary.md` – Source of truth for definitions

## Content source and format (`glossary.md`)
The parser expects very simple Markdown conventions:
- **Section letters** as H1 headings with a single capital letter:
  - `# A`, `# B`, ... `# Z`
- **Each term** as an H1 heading (`# Term Name`) followed by one or more paragraphs of definition text separated by blank lines.
- **Optional categories line**: Add a line starting with `Categories:` under a term. Separate multiple categories with commas, semicolons, or pipes.

Example:
```
# A

# AI (Artificial Intelligence)
 Categories: Language (LLMs & NLP), Applications & Assistants
A brief definition paragraph here. May include inline `code` or minimal formatting.

# Alignment
 Categories: Safety & Alignment
Another definition paragraph.
```
Notes:
- Only `#`-level headings are considered for parsing.
- Lines under a term, until the next `#` heading, are treated as the term’s definition.
- If present, the `Categories:` line is parsed and used to populate the category dropdown and filter. Category names are free‑form strings.

## Editing or adding definitions
- Place new terms under the correct letter section.
- Use a single `#` before the term name. Keep the definition below, separated by a blank line.
- Optionally add a `Categories:` line under the term. Use commas/semicolons/pipes to separate multiple categories. Choose from your preferred taxonomy; names appear verbatim in the dropdown.
- Keep the first sentence crisp; add 1–2 paragraphs max for clarity.
- Prefer neutral, vendor‑agnostic language. Add specific examples when helpful.
- Use inline code for technical tokens (e.g., `token`, `embedding`).

## Contributing new definitions (contributors)
We welcome improvements and new entries from the i2 orgs.

- **Add or edit a term**: Update `glossary.md` following the format above.
- **Submission**: Open a Pull Request with a clear description of your change.
- **Quality guidelines**:
  - Be accurate, concise, and neutral in tone.
  - When adding acronyms, expand them once in the term (e.g., `# ASR (Automatic Speech Recognition)`).
  - Include context, caveats, and links only when essential; avoid promotional language.
  - Keep definitions readable for non‑experts while retaining precision.
- **Review**: Insights team maintainers will review for clarity, accuracy, and consistency with the style.

If you can’t open a PR, you can also file an Issue with:
- The proposed term
- A draft definition (1–3 short paragraphs)
- Any reputable citations or references (optional)

## Development
- Pure static app; no build step required.
- Minimal Markdown rendering (paragraphs, inline code). Heavy Markdown features aren’t required for glossary use.
- Header layout is mobile‑friendly; the search input, category select, and theme toggle stay on one line on small screens.

### Advanced configuration
- **Preview length**: Append `?preview=NUMBER` to the URL (50–500) or set `localStorage.setItem('glossary-preview-limit', '175')`.
- **Theme**: Toggled via the header button; persisted in `localStorage` under `glossary-theme`.