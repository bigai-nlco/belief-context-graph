"""Review previews preserve evidence bytes while deferring expensive graphs."""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.case_study_studio.control import Control
from tools.case_study_studio.evidence_preview import preview_bundle
from tools.case_study_studio.tests.test_control import build_fixture
from workflows._shared.case_study import read_bundle


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[Control, Path, str]:
    control = build_fixture(tmp_path)
    path, digest = control.evidence(1, "drb2", "2")
    return control, path, digest


def test_preview_keeps_complete_cited_files_and_never_mutates_original(frozen: tuple[Control, Path, str]) -> None:
    control, path, digest = frozen
    original = path.read_bytes()
    preview, sha = control.evidence_preview(1, "drb2", "2")
    assert sha == digest
    assert preview != path
    assert path.read_bytes() == original
    assert not preview.stat().st_mode & 0o077
    full = read_bundle(path)
    value = read_bundle(preview)
    assert value["projection"] == {
        "kind": "review-preview", "version": 1, "source_sha256": digest,
        "omitted_files": [{"path": f["path"], "size": f["size"]}
                          for f in full["cases"][0]["files"]
                          if f not in value["cases"][0]["files"]],
    }
    kept = {f["path"]: f for f in value["cases"][0]["files"]}
    assert "agent/bcg/state.json" not in kept
    assert {"case.json", "agent/trajectories/react_agent.jsonl", "agent/bcg/requests.jsonl"} <= kept.keys()
    for file in full["cases"][0]["files"]:
        if file["path"] in kept:
            assert kept[file["path"]] == file
    # Reopening and concurrent readers never reread/decompress the full bundle.
    with (
        patch("tools.case_study_studio.evidence_preview.read_bundle", side_effect=AssertionError("cold load")),
        ThreadPoolExecutor(max_workers=4) as pool,
    ):
        assert list(pool.map(lambda _: preview_bundle(path, digest), range(8))) == [preview] * 8


def test_preview_parallel_first_reads_publish_one_valid_file(frozen: tuple[Control, Path, str]) -> None:
    _, path, digest = frozen
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: preview_bundle(path, digest), range(4)))
    assert len(set(results)) == 1
    assert read_bundle(results[0])["projection"]["source_sha256"] == digest
    assert not list(path.parent.glob(".preview-*"))


@pytest.mark.parametrize("mutation", ["full_hash", "source_hash", "source_size", "unsafe_path", "duplicate", "missing_metadata"])
def test_bad_preview_is_never_published(frozen: tuple[Control, Path, str], mutation: str) -> None:
    _, path, digest = frozen
    if mutation == "full_hash":
        digest = "0" * 64
    else:
        bundle = read_bundle(path)
        files = bundle["cases"][0]["files"]
        file = next(f for f in files if f["path"] == "case.json")
        if mutation == "source_hash":
            file["digest"] = "0" * 64
        elif mutation == "source_size":
            file["size"] += 1
        elif mutation == "unsafe_path":
            file["path"] = "../case.json"
        elif mutation == "duplicate":
            files.append(file.copy())
        else:
            files.remove(file)
        path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        preview_bundle(path, digest)
    assert not list(path.parent.glob("*.preview-*.gz"))
    assert not list(path.parent.glob(".preview-*"))


def test_new_frozen_revision_has_a_different_cache(frozen: tuple[Control, Path, str]) -> None:
    _, path, digest = frozen
    first = preview_bundle(path, digest)
    bundle = read_bundle(path)
    bundle["cases"][0]["warnings"].append("New immutable revision")
    path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    next_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    second = preview_bundle(path, next_digest)
    assert first != second
    assert read_bundle(first)["projection"]["source_sha256"] == digest
    assert read_bundle(second)["projection"]["source_sha256"] == next_digest
