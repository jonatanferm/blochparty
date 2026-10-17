"""Sequence.meta's keys are documented: every key the code that builds sequences (``sequence.py`` and
``sequences/``: builder, library, Pulseq reader, Sequence methods) writes into a sequence's meta is in
``sequences.meta.KEYS``, and every listed key is written somewhere. Other ``meta`` dictionaries (an anatomy's, a
generated sample's) are not scanned."""

import ast
import pathlib
from collections import defaultdict

from blochparty.sequences.meta import KEYS

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "blochparty"
SEQUENCE_CODE = (SRC / "sequence.py", SRC / "sequences")


def dict_keys(node):
    if isinstance(node, ast.Dict):
        out = set()
        for k, v in zip(node.keys, node.values):
            if k is None:
                out |= dict_keys(v)
            elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                out.add(k.value)
        return out
    if isinstance(node, ast.IfExp):
        return dict_keys(node.body) | dict_keys(node.orelse)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return dict_keys(node.left) | dict_keys(node.right)
    return set()


def is_meta(node):
    return (isinstance(node, ast.Attribute) and node.attr == "meta") or (
        isinstance(node, ast.Name) and node.id == "meta"
    )


def scan(roots=SEQUENCE_CODE):
    found = defaultdict(set)
    roots = [roots] if isinstance(roots, (str, pathlib.Path)) else roots
    paths = [p for r in map(pathlib.Path, roots) for p in (sorted(r.rglob("*.py")) if r.is_dir() else [r])]
    for path in paths:
        tree = ast.parse(path.read_text())
        for n in ast.walk(tree):
            keys = set()
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                    if isinstance(t, ast.Subscript) and is_meta(t.value) and isinstance(t.slice, ast.Constant):
                        keys.add(t.slice.value)
                    if isinstance(t, ast.Name) and t.id == "meta":
                        keys |= dict_keys(n.value)
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Attribute) and f.attr == "build":
                    for kw in n.keywords:
                        keys |= {kw.arg} if kw.arg else dict_keys(kw.value)
                if isinstance(f, ast.Attribute) and f.attr == "update" and is_meta(f.value):
                    for a in n.args:
                        keys |= dict_keys(a)
                    keys |= {kw.arg for kw in n.keywords if kw.arg}
                for kw in n.keywords:
                    if kw.arg == "meta":
                        keys |= dict_keys(kw.value)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) and is_meta(n.left):
                keys |= dict_keys(n.right)
            for k in keys:
                found[k].add(str(path))
    return found


def test_every_written_key_is_documented():
    written = scan()
    missing = {k: sorted(v) for k, v in written.items() if k not in KEYS}
    assert not missing, f"Sequence.meta keys without an entry in sequences/meta.py: {missing}"


def test_every_documented_key_is_written():
    assert set(KEYS) <= set(scan()), sorted(set(KEYS) - set(scan()))


def test_the_scan_sees_each_way_of_writing_a_key(tmp_path):
    """The scan is the test's only eyes: it must catch every form the package uses."""
    (tmp_path / "m.py").write_text(
        "seq.meta['a'] = 1\nmeta = {'b': 2, **({} if x else {'c': 3})}\nb.build(d=4)\n"
        "Sequence(meta={'e': 5})\nseq.meta.update({'f': 6})\nf(meta | {'g': 7})\n"
    )
    assert set(scan(tmp_path)) == set("abcdefg")
