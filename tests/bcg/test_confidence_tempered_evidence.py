import math

import pytest

from bcg.construct.unified import confidence


def test_unified_evidence_uses_discounted_log_odds() -> None:
    evidence = {"role": "tool", "stance": "asserted"}
    q = (0.80 + 0.75) / 2.0

    assert confidence.evidence_contribution(evidence) == pytest.approx(
        0.25 * math.log(q / (1.0 - q))
    )

    posterior = confidence.posterior_confidence(
        0.775, confidence.evidence_contribution(evidence), 0.0
    )
    assert posterior == pytest.approx(0.824, abs=0.001)
