# -*- coding: utf-8 -*-
"""Built-in documentation browser for flika.

Renders markdown files from flika/docs/ in a browser-style QWidget with
navigation history, table of contents sidebar, and search.
"""
from __future__ import annotations

import os
import re

from qtpy import QtCore, QtGui, QtWidgets

try:
    import markdown as _md
    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False


_DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #24292e;
    background: #ffffff;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 30px;
}
h1 { font-size: 28px; border-bottom: 1px solid #eaecef; padding-bottom: 8px; margin-top: 24px; }
h2 { font-size: 22px; border-bottom: 1px solid #eaecef; padding-bottom: 6px; margin-top: 20px; }
h3 { font-size: 18px; margin-top: 16px; }
h4 { font-size: 15px; margin-top: 12px; }
code {
    background: #f6f8fa;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 13px;
}
pre {
    background: #f6f8fa;
    padding: 12px 16px;
    border-radius: 6px;
    overflow-x: auto;
    line-height: 1.45;
}
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #dfe2e5; padding: 6px 13px; text-align: left; }
th { background: #f6f8fa; font-weight: 600; }
tr:nth-child(even) { background: #f6f8fa; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote {
    border-left: 4px solid #dfe2e5;
    padding: 0 16px;
    color: #6a737d;
    margin: 12px 0;
}
ul, ol { padding-left: 24px; }
hr { border: none; border-top: 1px solid #eaecef; margin: 20px 0; }
img { max-width: 100%; }
"""


def _md_to_html(md_text: str) -> str:
    """Convert markdown text to HTML."""
    if _HAS_MARKDOWN:
        html_body = _md.markdown(
            md_text,
            extensions=["tables", "fenced_code", "toc", "nl2br"],
        )
    else:
        # Minimal fallback: wrap in <pre>
        import html
        html_body = "<pre>" + html.escape(md_text) + "</pre>"
    return f"<html><head><style>{_CSS}</style></head><body>{html_body}</body></html>"


def _extract_toc(md_text: str) -> list[tuple[int, str, str]]:
    """Extract table of contents from markdown. Returns [(level, title, anchor), ...]."""
    toc = []
    for line in md_text.splitlines():
        m = re.match(r'^(#{1,4})\s+(.+)$', line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Generate anchor matching python-markdown's toc extension
            anchor = re.sub(r'[^\w\s-]', '', title.lower())
            anchor = re.sub(r'[\s]+', '-', anchor).strip('-')
            toc.append((level, title, anchor))
    return toc


def _list_doc_files() -> list[tuple[str, str]]:
    """List available doc files. Returns [(filename, display_name), ...]."""
    if not os.path.isdir(_DOCS_DIR):
        return []
    # Preferred ordering
    order = [
        "index.md", "getting_started.md", "user_interface.md",
        "file_operations.md", "image_menu.md", "process_menu.md",
        "roi_guide.md", "spt_guide.md", "ai_tools.md",
        "interoperability.md", "plugins.md", "api_reference.md",
        "keyboard_shortcuts.md", "troubleshooting.md",
    ]
    files = []
    seen = set()
    for name in order:
        path = os.path.join(_DOCS_DIR, name)
        if os.path.isfile(path):
            display = name.replace("_", " ").replace(".md", "").title()
            files.append((name, display))
            seen.add(name)
    # Add any extras not in the preferred order
    for name in sorted(os.listdir(_DOCS_DIR)):
        if name.endswith(".md") and name not in seen:
            display = name.replace("_", " ").replace(".md", "").title()
            files.append((name, display))
    return files


class DocBrowser(QtWidgets.QMainWindow):
    """Browser-style documentation viewer."""

    _instance = None

    @classmethod
    def instance(cls, parent=None):
        if cls._instance is None or not cls._instance.isVisible():
            cls._instance = cls(parent)
        return cls._instance

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Flika Documentation")
        self.resize(1050, 700)
        self._history: list[str] = []
        self._history_pos: int = -1
        self._current_file: str = ""

        # --- Toolbar ---
        toolbar = self.addToolBar("Navigation")
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(18, 18))

        self._back_btn = QtWidgets.QToolButton()
        self._back_btn.setText("\u25C0")
        self._back_btn.setToolTip("Back")
        self._back_btn.clicked.connect(self._go_back)
        toolbar.addWidget(self._back_btn)

        self._fwd_btn = QtWidgets.QToolButton()
        self._fwd_btn.setText("\u25B6")
        self._fwd_btn.setToolTip("Forward")
        self._fwd_btn.clicked.connect(self._go_forward)
        toolbar.addWidget(self._fwd_btn)

        self._home_btn = QtWidgets.QToolButton()
        self._home_btn.setText("\u2302")
        self._home_btn.setToolTip("Home")
        self._home_btn.clicked.connect(lambda: self.navigate("index.md"))
        toolbar.addWidget(self._home_btn)

        toolbar.addSeparator()

        self._search_box = QtWidgets.QLineEdit()
        self._search_box.setPlaceholderText("Search documentation...")
        self._search_box.setMaximumWidth(250)
        self._search_box.returnPressed.connect(self._do_search)
        toolbar.addWidget(self._search_box)

        search_btn = QtWidgets.QToolButton()
        search_btn.setText("Search")
        search_btn.clicked.connect(self._do_search)
        toolbar.addWidget(search_btn)

        # --- Splitter: sidebar + content ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Sidebar
        sidebar = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        sidebar_label = QtWidgets.QLabel("Contents")
        sidebar_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        sidebar_layout.addWidget(sidebar_label)

        self._file_list = QtWidgets.QListWidget()
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        sidebar_layout.addWidget(self._file_list)

        self._toc_label = QtWidgets.QLabel("On This Page")
        self._toc_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px; margin-top: 8px;")
        sidebar_layout.addWidget(self._toc_label)

        self._toc_list = QtWidgets.QTreeWidget()
        self._toc_list.setHeaderHidden(True)
        self._toc_list.setIndentation(16)
        self._toc_list.itemClicked.connect(self._on_toc_clicked)
        sidebar_layout.addWidget(self._toc_list)

        splitter.addWidget(sidebar)

        # Content area
        self._browser = QtWidgets.QTextBrowser()
        self._browser.setOpenLinks(False)
        self._browser.anchorClicked.connect(self._on_link_clicked)
        splitter.addWidget(self._browser)

        splitter.setSizes([220, 830])

        # Populate file list
        self._doc_files = _list_doc_files()
        for fname, display in self._doc_files:
            self._file_list.addItem(display)

        # Navigate to index
        self.navigate("index.md")
        self._update_nav_buttons()

    def navigate(self, filename: str, anchor: str = ""):
        """Navigate to a documentation file, optionally scrolling to an anchor."""
        filepath = os.path.join(_DOCS_DIR, filename)
        if not os.path.isfile(filepath):
            self._browser.setHtml(
                f"<html><body><h1>Not Found</h1>"
                f"<p>Documentation file <code>{filename}</code> not found.</p>"
                f"<p>Docs directory: <code>{_DOCS_DIR}</code></p></body></html>"
            )
            return

        with open(filepath, "r", encoding="utf-8") as f:
            md_text = f.read()

        html = _md_to_html(md_text)
        self._browser.setHtml(html)
        self._current_file = filename

        # Update history
        if self._history_pos < 0 or self._history[self._history_pos] != filename:
            # Truncate forward history
            self._history = self._history[:self._history_pos + 1]
            self._history.append(filename)
            self._history_pos = len(self._history) - 1

        self._update_nav_buttons()
        self._update_toc(md_text)
        self._select_file_in_list(filename)

        if anchor:
            self._browser.scrollToAnchor(anchor)

        self.setWindowTitle(f"Flika Documentation - {filename}")

    def _update_toc(self, md_text: str):
        """Update the table of contents sidebar."""
        self._toc_list.clear()
        toc = _extract_toc(md_text)
        if not toc:
            return

        min_level = min(level for level, _, _ in toc)
        parent_stack: dict[int, QtWidgets.QTreeWidgetItem] = {}

        for level, title, anchor in toc:
            item = QtWidgets.QTreeWidgetItem([title])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, anchor)

            if level == min_level:
                self._toc_list.addTopLevelItem(item)
            else:
                # Find nearest parent
                parent = None
                for lv in range(level - 1, min_level - 1, -1):
                    if lv in parent_stack:
                        parent = parent_stack[lv]
                        break
                if parent:
                    parent.addChild(item)
                else:
                    self._toc_list.addTopLevelItem(item)

            parent_stack[level] = item

        self._toc_list.expandAll()

    def _select_file_in_list(self, filename: str):
        """Select the corresponding entry in the file list."""
        for i, (fname, _) in enumerate(self._doc_files):
            if fname == filename:
                self._file_list.blockSignals(True)
                self._file_list.setCurrentRow(i)
                self._file_list.blockSignals(False)
                return

    def _on_file_selected(self, row: int):
        if 0 <= row < len(self._doc_files):
            self.navigate(self._doc_files[row][0])

    def _on_toc_clicked(self, item: QtWidgets.QTreeWidgetItem, column: int):
        anchor = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if anchor:
            self._browser.scrollToAnchor(anchor)

    def _on_link_clicked(self, url: QtCore.QUrl):
        """Handle clicks on links within documentation."""
        link = url.toString()

        # External URL
        if link.startswith("http://") or link.startswith("https://"):
            QtGui.QDesktopServices.openUrl(url)
            return

        # Internal .md link, possibly with anchor
        anchor = ""
        if "#" in link:
            link, anchor = link.rsplit("#", 1)

        if link.endswith(".md"):
            self.navigate(link, anchor)
        elif link == "" and anchor:
            # Same-page anchor
            self._browser.scrollToAnchor(anchor)

    def _go_back(self):
        if self._history_pos > 0:
            self._history_pos -= 1
            filename = self._history[self._history_pos]
            # Don't add to history again
            filepath = os.path.join(_DOCS_DIR, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    md_text = f.read()
                self._browser.setHtml(_md_to_html(md_text))
                self._current_file = filename
                self._update_toc(md_text)
                self._select_file_in_list(filename)
            self._update_nav_buttons()

    def _go_forward(self):
        if self._history_pos < len(self._history) - 1:
            self._history_pos += 1
            filename = self._history[self._history_pos]
            filepath = os.path.join(_DOCS_DIR, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    md_text = f.read()
                self._browser.setHtml(_md_to_html(md_text))
                self._current_file = filename
                self._update_toc(md_text)
                self._select_file_in_list(filename)
            self._update_nav_buttons()

    def _update_nav_buttons(self):
        self._back_btn.setEnabled(self._history_pos > 0)
        self._fwd_btn.setEnabled(self._history_pos < len(self._history) - 1)

    def _do_search(self):
        """Search all doc files for the query and show results."""
        query = self._search_box.text().strip()
        if not query:
            return

        results = []
        query_lower = query.lower()
        for fname, display in self._doc_files:
            filepath = os.path.join(_DOCS_DIR, fname)
            if not os.path.isfile(filepath):
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            matches = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    # Get context: the line itself
                    clean = line.strip().lstrip("#").strip()
                    if clean:
                        matches.append(clean[:120])
            if matches:
                results.append((fname, display, matches))

        # Render results as HTML
        html_parts = [f"<h1>Search Results for \"{query}\"</h1>"]
        if not results:
            html_parts.append("<p>No results found.</p>")
        else:
            total = sum(len(m) for _, _, m in results)
            html_parts.append(f"<p>Found {total} match(es) in {len(results)} file(s).</p>")
            for fname, display, matches in results:
                html_parts.append(f'<h2><a href="{fname}">{display}</a></h2>')
                html_parts.append("<ul>")
                for match in matches[:8]:
                    highlighted = re.sub(
                        re.escape(query),
                        f"<b>{query}</b>",
                        match,
                        flags=re.IGNORECASE,
                    )
                    html_parts.append(f"<li>{highlighted}</li>")
                if len(matches) > 8:
                    html_parts.append(f"<li><i>...and {len(matches) - 8} more</i></li>")
                html_parts.append("</ul>")

        full_html = f"<html><head><style>{_CSS}</style></head><body>{''.join(html_parts)}</body></html>"
        self._browser.setHtml(full_html)
        self._toc_list.clear()


def show_documentation(parent=None):
    """Show the documentation browser window."""
    browser = DocBrowser.instance(parent)
    browser.show()
    browser.raise_()
    browser.activateWindow()
