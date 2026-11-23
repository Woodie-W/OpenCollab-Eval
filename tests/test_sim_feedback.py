import hashlib
import json

import pytest

from opencollab_eval.cli import main
from opencollab_eval.commands.sim_feedback import digest, replay


def put(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item))


@pytest.fixture
def campaign(tmp_path):
    task = "079_ImageNet_83.6_test"
    memory = tmp_path / "source-memory.json"
    put(memory, {"version": 1, "problems": {}})
    put(tmp_path / "manifest.json", {"tasks": [{"task": task, "workloads": 16}], "memory_sha256": digest(memory)})
    prediction = dict(
        problem_id=task,
        source_memory_sha256=digest(memory),
        diagnosis="slow reduction",
        edit="retile",
        expected_outcome="higher score",
    )
    for sid, arm in enumerate(("before", "after"), 1):
        source = tmp_path / "cases" / "079" / arm / "submission.py"
        source.parent.mkdir(parents=True)
        source.write_text(f"# revision {sid}")
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        prediction["parent" if arm == "before" else "candidate"] = fingerprint
        folder = tmp_path / "deepseek-official" / "079" / arm
        receipt = dict(task=task, source_sha256=fingerprint, gpu="B200", visibility="Private", submission_id=sid)
        put(folder / "submission-receipt.json", receipt)
        put(folder / "terminal.json", dict(**receipt, verdict="AC", score=sid / 10, stack="v1.1"))
    put(tmp_path / "cases/079/after/prediction.json", prediction)
    return tmp_path, task


@pytest.mark.parametrize(("verdict", "score", "chosen"), [("AC", 0.2, 2), ("WA", 0.9, 1), ("AC", 0.1, 1)])
def test_replay_records_outcomes_and_selects_parent(campaign, capsys, verdict, score, chosen):
    root, task = campaign
    path = root / "deepseek-official/079/after/terminal.json"
    evidence = dict(json.loads(path.read_text()), verdict=verdict, score=score)
    put(root / "deepseek-b200-snapshot.json", {"2": evidence})
    path.unlink()
    assert main(["sim-feedback", str(root)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == replay(root)
    assert result["selected_parents"][task]["id"] == chosen
    attempts = result["memory"]["problems"][task]["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["outcome"]["after_verdict"] == verdict
    assert attempts[0]["outcome"]["promoted"] is (chosen == 2)


def test_pending_and_changed_source(campaign):
    root, task = campaign
    (root / "deepseek-official/079/after/terminal.json").unlink()
    result = replay(root)
    assert result["selected_parents"][task]["id"] == 1
    assert not result["memory"]["problems"]
    (root / "cases/079/after/submission.py").write_text("changed")
    with pytest.raises(ValueError, match="Receipt mismatch"):
        replay(root)
