"""Unit tests for the multi-tool-call protocol layer (parse/canonicalize/render)."""

from __future__ import annotations

from bcg.agent.tool_call_protocol import (
    canonicalize_tool_call_text,
    parse_tool_call_blocks,
    render_tool_results_xml,
    validate_raw_ids,
)


def test_parse_multiple_blocks_no_id():
    text = (
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query A"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query B"}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    assert [c.id for c in calls] == ["call_1", "call_2"]
    assert [c.name for c in calls] == ["averitec_search", "averitec_search"]
    assert calls[0].arguments == {"query": "query A"}
    assert calls[1].arguments == {"query": "query B"}
    assert all(c.format_error is None for c in calls)
    assert all(c.raw_id is None for c in calls)


def test_parse_multiple_blocks_with_id_attribute():
    text = (
        '<tool_call id="call_1">\n{"name": "averitec_search", "arguments": {"query": "A"}}\n</tool_call>\n'
        '<tool_call id="call_2">\n{"name": "averitec_search", "arguments": {"query": "B"}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    assert [c.id for c in calls] == ["call_1", "call_2"]
    assert [c.raw_id for c in calls] == ["call_1", "call_2"]


def test_reassigns_ids_regardless_of_model_supplied_out_of_order_ids():
    # Model writes id="call_5" then id="call_2" (out of order / skipping) --
    # final ids must still follow parse order starting at call_1.
    text = (
        '<tool_call id="call_5">\n{"name": "averitec_search", "arguments": {"query": "first"}}\n</tool_call>\n'
        '<tool_call id="call_2">\n{"name": "averitec_search", "arguments": {"query": "second"}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    assert [c.id for c in calls] == ["call_1", "call_2"]
    assert [c.raw_id for c in calls] == ["call_5", "call_2"]
    assert calls[0].arguments == {"query": "first"}
    assert calls[1].arguments == {"query": "second"}


def test_validate_raw_ids_flags_bad_format_and_duplicates():
    text = (
        '<tool_call id="bogus">\n{"name": "a", "arguments": {}}\n</tool_call>\n'
        '<tool_call id="call_1">\n{"name": "b", "arguments": {}}\n</tool_call>\n'
        '<tool_call id="call_1">\n{"name": "c", "arguments": {}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    warnings = validate_raw_ids(calls)
    assert any("bogus" in w for w in warnings)
    assert any("call_1" in w and "repeated" in w for w in warnings)


def test_json_format_error_does_not_consume_id_slot():
    text = (
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "ok"}}\n</tool_call>\n'
        "<tool_call>\nnot valid json{{{\n</tool_call>\n"
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "also ok"}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    assert len(calls) == 3
    assert calls[0].id == "call_1"
    assert calls[1].id == ""
    assert calls[1].format_error is not None
    # The valid block after the broken one still gets the next contiguous id.
    assert calls[2].id == "call_2"


def test_missing_name_or_bad_arguments_shape_is_a_format_error():
    text = (
        '<tool_call>\n{"arguments": {"query": "no name field"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "x", "arguments": "not-an-object"}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    assert len(calls) == 2
    assert all(c.format_error is not None for c in calls)
    assert all(c.id == "" for c in calls)


def test_canonicalize_rewrites_only_opening_tag():
    text = (
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query A"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "averitec_search", "arguments": {"query": "query B"}}\n</tool_call>'
    )
    calls = parse_tool_call_blocks(text)
    canonical = canonicalize_tool_call_text(text, calls)
    expected = (
        '<tool_call id="call_1">\n{"name": "averitec_search", "arguments": {"query": "query A"}}\n</tool_call>\n'
        '<tool_call id="call_2">\n{"name": "averitec_search", "arguments": {"query": "query B"}}\n</tool_call>'
    )
    assert canonical == expected


def test_canonicalize_preserves_surrounding_text_and_no_tool_call_case():
    text = "<think>reasoning here</think>\nSome text before.\n"
    calls = parse_tool_call_blocks(text)
    assert calls == []
    assert canonicalize_tool_call_text(text, calls) == text


def test_canonicalize_preserves_multiline_json_and_leaves_broken_block_untouched():
    text = (
        '<tool_call>\n{\n  "name": "averitec_search",\n  "arguments": {\n    "query": "multi\\nline"\n  }\n}\n</tool_call>\n'
        "<tool_call>\nbroken{{\n</tool_call>"
    )
    calls = parse_tool_call_blocks(text)
    canonical = canonicalize_tool_call_text(text, calls)
    assert canonical.startswith('<tool_call id="call_1">')
    # The broken block keeps its original (id-less) opening tag verbatim.
    assert "<tool_call>\nbroken{{\n</tool_call>" in canonical
    assert 'id="call_2"' not in canonical  # broken block never got an id


def test_render_tool_result_format_matches_spec_and_omits_urls():
    entries = [
        {
            "tool_call_id": "call_1",
            "name": "averitec_search",
            "query": "Did GiveSendGo host a fundraiser for Kyle Rittenhouse?",
            "evidence": [
                {"text": "Evidence text one.", "url": "http://example.com/a"},
                {"text": "Evidence text two.", "url": "http://example.com/b"},
            ],
        },
        {
            "tool_call_id": "call_2",
            "name": "averitec_search",
            "query": "Was there a GiveSendGo campaign in August 2020?",
            "evidence": [{"text": "Another evidence text."}],
        },
    ]
    xml = render_tool_results_xml(entries)

    assert '<tool_result id="call_1" name="averitec_search">' in xml
    assert '<tool_result id="call_2" name="averitec_search">' in xml
    assert "Query: Did GiveSendGo host a fundraiser for Kyle Rittenhouse?" in xml
    assert "Query: Was there a GiveSendGo campaign in August 2020?" in xml
    assert "[call_1_evidence_1]" in xml
    assert "[call_1_evidence_2]" in xml
    assert "[call_2_evidence_1]" in xml
    # No bare/duplicate legacy-style numbering, and no URLs anywhere.
    assert "[Evidence 1]" not in xml
    assert "[Evidence 2]" not in xml
    assert "http://" not in xml
    assert "url" not in xml.lower()
    assert xml.count("</tool_result>") == 2


def test_render_tool_result_handles_no_evidence():
    entries = [
        {"tool_call_id": "call_1", "name": "averitec_search", "query": "q", "evidence": []}
    ]
    xml = render_tool_results_xml(entries)
    assert "Evidence: (none found)" in xml
    assert "http://" not in xml
