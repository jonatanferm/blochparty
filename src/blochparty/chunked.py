"""``blochparty.chunked`` is ``blochparty.engines.chunked`` (moved in the 2026-10 refactor): this module replaces
itself by that one, so its names, private helpers and monkeypatching are the same objects."""

import sys

from .engines import chunked as _module

sys.modules[__name__] = _module
