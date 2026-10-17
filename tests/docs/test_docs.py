"""The documents that indexes keep in order: the journal (one file per entry, docs/JOURNAL.md the index) and the
lessons (topic files, docs/LESSONS.md the index). An entry written without its index line, or an index line whose
file was renamed, fails here rather than silently dropping out of the record."""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"


def linked(index: Path, folder: str) -> list[str]:
    return re.findall(rf"\]\({folder}/([^)]+\.md)\)", index.read_text())


def test_journal_index_and_entry_files_agree():
    links = linked(DOCS / "JOURNAL.md", "journal")
    files = sorted(p.name for p in (DOCS / "journal").glob("*.md"))
    assert len(links) == len(set(links)), "an entry is listed twice"
    assert sorted(links) == files
    assert links == sorted(links), "the index lists entries in date order, newest last"


def test_journal_entry_names_and_headings():
    for path in (DOCS / "journal").glob("*.md"):
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{2})-[a-z0-9-]+\.md", path.name)
        assert m, f"{path.name}: expected <date>-<nn>-<slug>.md"
        first = path.read_text().split("\n", 1)[0]
        assert first.startswith(f"# {m[1]}"), f"{path.name}: the heading starts with the entry's date"


def test_lessons_index_and_topic_files_agree():
    links = linked(DOCS / "LESSONS.md", "lessons")
    files = sorted(p.name for p in (DOCS / "lessons").glob("*.md"))
    assert sorted(links) == files
    for name in files:
        text = (DOCS / "lessons" / name).read_text()
        count = sum(1 for line in text.split("\n") if line.startswith("- "))
        assert f"(lessons/{name}) ({count})" in (DOCS / "LESSONS.md").read_text(), f"{name}: the index's count is {count}"


def test_readme_layout_is_generated_from_the_docstrings():
    """tools/layout.py writes README.md's layout table from the module docstrings; a stale table fails here."""
    import sys

    sys.path.insert(0, str(DOCS.parent / "tools"))
    import layout

    assert layout.main(["--check"]) == 0, "run: uv run python tools/layout.py"


def test_relative_links_resolve():
    """Every relative link in the living documents points at a file that exists (the journal and the archived
    roadmap keep the paths of their day)."""
    root = DOCS.parent
    files = [*root.glob("*.md"), *DOCS.glob("*.md"), *DOCS.glob("lessons/*.md"), *DOCS.glob("specs/*.md"),
             *DOCS.glob("reviews/*.md"), *root.glob("validation/**/*.md"), *root.glob("src/**/*.md")]
    broken = []
    for f in files:
        if f.name in ("ROADMAP_ARCHIVE.md",):
            continue
        for target in re.findall(r"\]\(([^)#\s]+)", f.read_text()):
            if not target.startswith(("http:", "https:", "mailto:")) and not (f.parent / target).exists():
                broken.append(f"{f.relative_to(root)}: {target}")
    assert not broken, broken
