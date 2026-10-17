"""``blochparty.metal`` is ``blochparty.engines.gpu.metal`` (moved in the 2026-10 refactor): this module
replaces itself by that one, so its names, private helpers and monkeypatching are the same objects."""

import sys

from ..engines.gpu import metal as _metal

sys.modules[__name__] = _metal
