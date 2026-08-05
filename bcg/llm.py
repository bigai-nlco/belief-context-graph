"""Model client and helpers. (compatibility re-export; implementation moved to ``bcg.core.llm`` in step 5)."""

from bcg.core.llm import *  # noqa: F403,F401
from bcg.core.llm import (  # noqa: F401
    _response_output_items,
    _usage_dict,
)
