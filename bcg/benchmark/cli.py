"""(compatibility re-export; implementation moved to ``bcg.apps.benchmark.cli`` in step 6)."""

from bcg.apps.benchmark.cli import *  # noqa: F403,F401
from bcg.apps.benchmark.cli import main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    main()
