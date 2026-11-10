import math

import pytest

from bcg.construct.unified import confidence


def test_unified_evidence_uses_discounted_log_odds() -> None:
    evidence = {"role": "tool", "stance": "confirmed"}
    q = 0.60 * 0.94

    assert confidence.evidence_contribution(evidence) == pytest.approx(
        0.25 * math.log(q / (1.0 - q))
    )

    posterior = confidence.posterior_confidence(
        0.564, confidence.evidence_contribution(evidence), 0.0
    )
    assert posterior == pytest.approx(0.579, abs=0.002)
