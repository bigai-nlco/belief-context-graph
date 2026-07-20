from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_start_script_falls_back_to_path_installed_bcg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "start.sh", scripts / "start.sh")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "args.txt"
    fake_bcg = bin_dir / "bcg"
    fake_bcg.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_FILE"\n',
        encoding="utf-8",
    )
    fake_bcg.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "MODEL": "test-model",
            "CAPTURE_FILE": str(capture),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    env.pop("BCG_BIN", None)

    result = subprocess.run(
        ["bash", str(scripts / "start.sh"), "--max-problems", "2"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "agent",
        "run",
        "--preset",
        "averitec-hero4",
        "--model",
        "test-model",
        "--max-problems",
        "2",
    ]
