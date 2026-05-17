# Claude collaboration notes for `flika---Georges-Edition`

This project was built with substantial assistance from **Claude (Anthropic)**, primarily via **Claude Code**. Git history contains explicit `Co-Authored-By: Claude` commit trailers.

This fork of FLIKA includes a **built-in Claude integration ecosystem** (interactive analysis assistant, scripting helper, natural-language plugin generator) and **explicit safety controls for AI-generated scientific code** (automated source-code validation of generated plugins, detection of unsafe operations, session-level approval gates, and an explicit safety-policy layer).

## Global preferences

George's global Claude preferences live in `~/.claude/CLAUDE.md` and apply to every project. Highlights:

- Keep all source files under 500 lines; split into focused modules if approaching that limit.
- For Python projects, maintain an `INTERFACE.md` at the project root as a navigation map of modules, classes, and key functions.
- Prefer many small, focused files over fewer large ones.
- Group related functionality; extract reusable logic into utility modules.

## Project navigation

See [`INTERFACE.md`](./INTERFACE.md) for a navigation map of this repository's modules and how they connect. Read `INTERFACE.md` before opening individual source files.

## Working with Claude on this repo

When directing Claude Code on this codebase, follow the global rules above, prefer making targeted edits via the `Edit` tool over rewriting whole files, and update `INTERFACE.md` whenever the project structure changes. The in-app Plugin Generator emits Python plugins that should always pass the project's AST-validation and unsafe-pattern checks before being approved at the session-level gate.

Project-specific conventions, build instructions, and lab-specific notes should be added below as the project evolves.
