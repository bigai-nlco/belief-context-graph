"""(compatibility re-export; implementation moved to ``bcg.apps.online_driver`` in step 6)."""

from bcg.apps.online_driver import *  # noqa: F403,F401
from bcg.apps.online_driver import main  # noqa: F401

if __name__ == "__main__":  # pragma: no cover
    main()
