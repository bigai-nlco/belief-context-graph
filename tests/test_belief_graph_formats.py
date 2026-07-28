from bcg.agent.belief_graph_client import BeliefGraphClient


def test_deepseek_v4_uses_top_level_belief_role_before_legacy_source_role():
    snapshot = {
        "beliefs": [
            {
                "id": 1,
                "role": "user",
                "belief": "from current user schema",
                "source": {"turn_id": 0},
            },
            {
                "id": 2,
                "role": "assistant",
                "belief": "top-level role wins",
                "source": {"role": "user", "turn_id": 1},
            },
            {"id": 3, "belief": "from legacy user schema", "source": {"role": "user"}},
            {
                "id": 4,
                "belief": "missing role defaults to assistant",
                "source": {"turn_id": 3},
            },
            {"id": 5, "role": "tool", "belief": "unsupported role falls back to user"},
        ]
    }

    encoded = BeliefGraphClient.format_graph_for_prompt(snapshot, fmt="deepseek_v4")

    assert encoded.startswith(
        '<｜begin▁of▁sentence｜><｜User｜>{"id":1,"content":"from current user schema"'
    )
    assert '<｜Assistant｜>{"id":2,"content":"top-level role wins"' in encoded
    assert '<｜User｜>{"id":3,"content":"from legacy user schema"' in encoded
    assert (
        '<｜Assistant｜>{"id":4,"content":"missing role defaults to assistant"'
        in encoded
    )
    assert (
        '<｜User｜>{"id":5,"content":"unsupported role falls back to user"' in encoded
    )


def test_json_format_exposes_current_bcg_top_level_role():
    snapshot = {
        "beliefs": [
            {
                "id": 1,
                "role": "user",
                "belief": "question belief",
                "source": {"turn_id": 0},
            },
        ]
    }

    formatted = BeliefGraphClient.format_graph_for_prompt(snapshot, fmt="json")

    assert '"role": "user"' in formatted


def test_deepseek_v4_xml_payload_inside_role_markers():
    snapshot = {
        "beliefs": [
            {"id": 1, "role": "user", "belief": "A & B", "confidence": 0.8},
            {"id": 2, "role": "assistant", "belief": "answer", "confidence": 0.9},
        ],
        "relations": [
            {"from_id": 1, "to_id": 2, "type": "supports", "note": "x < y"},
        ],
    }

    encoded = BeliefGraphClient.format_graph_for_prompt(
        snapshot,
        fmt="deepseek_v4",
        deepseek_v4_payload_format="xml",
    )

    assert encoded.startswith('<｜begin▁of▁sentence｜><｜User｜><belief id="1">')
    assert "<content>A &amp; B</content>" in encoded
    assert '<relation direction="outgoing" to="2" type="supports">' in encoded
    assert "<reason>x &lt; y</reason>" in encoded
    assert "<confidence>0.8</confidence>" in encoded
    assert '<｜Assistant｜><belief id="2">' in encoded
    assert encoded.endswith("</belief><｜end▁of▁sentence｜>")


def test_deepseek_v4_markdown_payload_inside_role_markers():
    snapshot = {
        "beliefs": [
            {"id": 1, "role": "user", "belief": "question", "confidence": 0.8},
        ],
    }

    encoded = BeliefGraphClient.format_graph_for_prompt(
        snapshot,
        fmt="deepseek_v4",
        deepseek_v4_payload_format="markdown",
    )

    assert encoded.startswith("<｜begin▁of▁sentence｜><｜User｜>### Belief 1\n")
    assert "**Content:** question" in encoded
    assert "**Relations:**\n- None" in encoded
    assert encoded.endswith("**Confidence:** 0.8")
