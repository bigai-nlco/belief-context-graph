"""(compatibility re-export; implementation moved to ``bcg.apps.benchmark.runner`` in step 6)."""

from bcg.apps.benchmark.runner import *  # noqa: F403,F401
from bcg.apps.benchmark.runner import (  # noqa: F401
    _execute,
    _interleaved_work,
)
