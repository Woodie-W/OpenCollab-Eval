"""Replay frozen SOL revisions into version-1 self-improvement memory."""

import hashlib
import json
import math
import re
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official(root, task, arm, snapshot):
    folder = root / "deepseek-official" / task[:3] / arm
    if not (folder / "submission-receipt.json").is_file():
        return None
    receipt = read(folder / "submission-receipt.json")
    sid = receipt["submission_id"]
    source = Path("cases") / task[:3] / arm / "submission.py"
    fingerprint = digest(root / source)
    if (receipt["task"], receipt["source_sha256"], receipt["gpu"], receipt["visibility"]) != (
        task,
        fingerprint,
        "B200",
        "Private",
    ):
        raise ValueError(f"Receipt mismatch for {source}")
    evidence = snapshot.get(str(sid))
    if (folder / "terminal.json").is_file():
        terminal = read(folder / "terminal.json")
        if terminal["submission_id"] != sid or (
            evidence and any(terminal[k] != evidence[k] for k in ("task", "source_sha256", "stack", "verdict", "score"))
        ):
            raise ValueError(f"Conflicting evidence for {sid}")
        evidence = terminal
    if evidence is None or evidence["stack"] != "v1.1":
        return None
    if (evidence["task"], evidence["source_sha256"]) != (task, fingerprint):
        raise ValueError(f"Result mismatch for {source}")
    verdict, score = evidence["verdict"], evidence["score"]
    if verdict not in {"AC", "WA", "CE", "RE", "TLE", "MLE"}:
        raise ValueError(f"Unknown terminal verdict for {sid}")
    if score is not None or verdict == "AC":
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError(f"Invalid official score for {sid}")
    return dict(id=sid, verdict=verdict, score=score, sha256=fingerprint, source=str(source))


def replay(root: Path) -> dict:
    """Build optimizer history and selected parents from an immutable campaign input."""
    manifest = read(root / "manifest.json")
    memory_path = root / "source-memory.json"
    memory_hash = digest(memory_path)
    if manifest["memory_sha256"] != memory_hash:
        raise ValueError("Source memory changed")
    memory = read(memory_path)
    if memory.get("version") != 1 or not isinstance(memory.get("problems"), dict):
        raise ValueError("Expected version-1 SIM problems")
    archive = root / "deepseek-b200-snapshot.json"
    snapshot = read(archive) if archive.exists() else {}
    parents = {}
    for row in manifest["tasks"]:
        task = row["task"]
        if not re.fullmatch(r"[0-9]{3}_[a-zA-Z0-9_.-]+", task) or task in parents:
            raise ValueError(f"Invalid or repeated task {task!r}")
        before = official(root, task, "before", snapshot)
        after = official(root, task, "after", snapshot)
        parents[task] = before if before and before["verdict"] == "AC" else None
        if before is None or after is None:
            continue
        case = root / "cases" / task[:3]
        prediction = read(case / "after" / "prediction.json")
        if (
            prediction["problem_id"],
            prediction["parent"],
            prediction["candidate"],
            prediction["source_memory_sha256"],
        ) != (task, before["sha256"], after["sha256"], memory_hash):
            raise ValueError(f"Frozen prediction mismatch for {task}")
        if not all(
            isinstance(prediction.get(k), str) and prediction[k].strip()
            for k in ("diagnosis", "edit", "expected_outcome")
        ):
            raise ValueError(f"Incomplete prediction for {task}")
        promote = after["verdict"] == "AC" and (parents[task] is None or after["score"] > before["score"])
        if promote:
            parents[task] = after
        outcome = {
            "target": "Private B200 v1.1",
            "promoted": promote,
            "a800_workloads_after": None,
            "a800_workloads_total": row["workloads"],
        }
        grade_path = case / "a800" / "grade.json"
        if grade_path.exists():
            grade = read(grade_path)
            coverage = grade["coverage"]
            if (
                grade["source_sha256"],
                coverage["operator"],
                coverage["workloads_expected"],
                coverage["protocol"]["protocol_id"],
            ) != (after["sha256"], task, row["workloads"], "sol94_a800_events_v1"):
                raise ValueError(f"A800 grade mismatch for {task}")
            outcome["a800_workloads_after"] = coverage["workloads_passed"]
        for arm, result in (("before", before), ("after", after)):
            outcome.update(
                {arm + "_id": result["id"], arm + "_verdict": result["verdict"], arm + "_sol": result["score"]}
            )
        problem = memory["problems"].setdefault(task, {"diagnosis": prediction["diagnosis"], "attempts": []})
        entry = {"prediction": prediction, "outcome": outcome}
        if entry not in problem["attempts"]:
            problem["attempts"].append(entry)
    return {"memory": memory, "selected_parents": parents}
