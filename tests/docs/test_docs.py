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
