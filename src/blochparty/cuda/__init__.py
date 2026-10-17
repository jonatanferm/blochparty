"""``blochparty.cuda`` is ``blochparty.engines.gpu.cuda`` (moved in the 2026-10 refactor): this module
replaces itself by that one, so its names, private helpers and monkeypatching are the same objects."""

import sys

from ..engines.gpu import cuda as _cuda

sys.modules[__name__] = _cuda
