"""(compatibility re-export; implementation moved to ``bcg.apps.run`` in step 6)."""

from bcg.apps.run import *  # noqa: F403,F401
from bcg.apps.run import main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    main()
