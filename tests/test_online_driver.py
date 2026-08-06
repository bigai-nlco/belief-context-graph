from __future__ import annotations

from bcg.apps.online_driver import drive, iter_jsonl


class FakeManager:
    def __init__(self) -> None:
        self.active = {"open"}
        self.seen: list[dict] = []

    def push(self, turn):
        self.seen.append(turn)
        problem_id = turn["problem_id"]
        if turn.get("is_trajectory_end"):
            self.active.discard(problem_id)
        return {
            "problem_id": problem_id,
            "stage": "final" if turn.get("is_trajectory_end") else "turn",
            "n_beliefs": len(self.seen),
            "finalized": bool(turn.get("is_trajectory_end")),
        }

    def active_problem_ids(self):
        return sorted(self.active)

    def all_problem_ids(self):
        return ["closed", "open"]

    def finalize(self, problem_id):
        self.active.discard(problem_id)
        return {"problem_id": problem_id, "finalized": True}


def test_iter_jsonl_skips_invalid_and_non_object_lines(capsys) -> None:
    turns = list(
        iter_jsonl(
            [
                "\n",
                '{"problem_id":"p1"}\n',
                "not-json\n",
                "[1, 2]\n",
            ]
        )
    )

    assert turns == [{"problem_id": "p1"}]
    assert "malformed JSON" in capsys.readouterr().err


def test_drive_finalizes_open_trajectories_at_eof() -> None:
    manager = FakeManager()

    summary = drive(
        manager,
        [
            {"problem_id": "closed", "is_trajectory_end": True},
            {"problem_id": "open"},
        ],
        quiet=True,
    )

    assert summary == {
        "n_turns_pushed": 2,
        "problems": ["closed", "open"],
        "finalized": ["closed", "open"],
    }
