# flika --- George's Edition

A fork of [FLIKA](https://github.com/flika-org/flika) (the Python/Qt scientific imaging environment) extended for super-resolution microscopy, single-particle tracking, calcium imaging, FRAP, spectral unmixing, microscopy simulation, and 4D data.

## What's different in this fork

This edition adds a built-in Claude integration ecosystem:

- **Claude Live Session** — agentic tool-use for interactive analysis.
- **Script Assistant** — multi-turn AI chat for scripting help.
- **Plugin Generator** — generates FLIKA plugins from natural-language descriptions.

The fork also ships explicit **safety scaffolding for AI-generated scientific code**: AST-based validation of generated plugins, regex pattern detection for dangerous code, session-level approval gates, and an explicit Claude safety-policy layer.

See [`CLAUDE.md`](./CLAUDE.md) for Claude collaboration notes and [`INTERFACE.md`](./INTERFACE.md) when present for a navigation map of modules.

---
*Built with AI assistance from [Claude (Anthropic)](https://claude.com/).*
