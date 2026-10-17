"""``blochparty.builders`` is ``blochparty.sequences.library`` (moved in the 2026-10 refactor): this module replaces
itself by that one, so its names, private helpers and monkeypatching are the same objects."""

import sys

from .sequences import library as _module

sys.modules[__name__] = _module
