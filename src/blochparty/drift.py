"""``blochparty.drift`` is ``blochparty.sequences.drift`` (moved in the 2026-10 refactor): this module replaces
itself by that one, so its names, private helpers and monkeypatching are the same objects."""

import sys

from .sequences import drift as _module

sys.modules[__name__] = _module
