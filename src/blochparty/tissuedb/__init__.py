"""``blochparty.tissuedb`` is ``blochparty.anatomy.tissuedb`` (moved in the 2026-10 refactor): this module
replaces itself by that one, so its names and its ``ROOT`` (the CSV folder) are the same objects."""

import sys

from ..anatomy import tissuedb as _tissuedb

sys.modules[__name__] = _tissuedb
