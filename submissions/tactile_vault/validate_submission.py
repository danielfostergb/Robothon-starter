#!/usr/bin/env python3
"""Independent static and dynamic validation for Tactile Vault v2."""

from __future__ import annotations

import json
import py_compile
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(condition: bool, message: str, errors: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    registration = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
    check(registration["project_name"] == "Tactile Vault", "registration project name", errors)
    check(registration["uuid"] != "PASTE-YOUR-UUID-HERE", "registration UUID is populated", errors)

    root = ET.parse(ROOT / "scene.xml").getroot()
    actuators = root.findall("./actuator/*")
    sensors = root.findall("./sensor/*")
    adhesion = root.findall("./actuator/adhesion")
    touch = root.findall("./sensor/touch")
    check(len(actuators) == 14, f"exact actuator inventory ({len(actuators)})", errors)
    check(len(sensors) == 19, f"exact sensor inventory ({len(sensors)})", errors)
    check(len(adhesion) == 5, "five independent adhesion actuators", errors)
    check(len(touch) == 9, "nine physical touch channels", errors)
    check(root.find(".//freejoint[@name='vial_free']") is not None, "medicine vial is a free body", errors)
    lock = root.find("./equality/weld[@name='lid_lock']")
    check(lock is not None and lock.get("active") == "true", "physical lid lock begins active", errors)
    check(len(root.findall("./equality/*")) == 1, "lid lock is the sole equality", errors)
    check(all("vial" not in " ".join(eq.attrib.values()) for eq in root.findall("./equality/*")), "no grasp equality references vial", errors)
    task_joints = {"key_1_slide", "key_2_slide", "key_3_slide", "lid_slide", "vial_free"}
    actuator_joints = {node.get("joint") for node in actuators if node.get("joint")}
    check(task_joints.isdisjoint(actuator_joints), "keys, lid, and vial have no position actuator", errors)

    py_compile.compile(str(ROOT / "run_demo.py"), doraise=True)
    check(True, "Python source compiles", errors)
    for name in ("README.md", "JUDGE_BRIEF.md", "PR_DESCRIPTION.md", "config.json", "requirements.txt", "rubric_scorecard.json", "submission_manifest.json"):
        check((ROOT / name).is_file(), f"required file: {name}", errors)
    pr = (ROOT / "PR_DESCRIPTION.md").read_text(encoding="utf-8")
    uuids = re.findall(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", pr, re.I)
    check(registration["uuid"] in uuids, "registration UUID matches PR description", errors)

    # This is deliberately an actual task execution, not a search for favorable
    # strings in source or a trust of pre-generated report.json.
    from run_demo import run_task

    report, samples = run_task(render=False)
    check(report["task_success"], "independent end-to-end physics run succeeds", errors)
    check(all(report["task_checks"].values()), "all eleven mechanical gates pass", errors)
    check(report["grasp_equalities"] == 0, "runtime report confirms contact-only grasp", errors)
    check(report["task_object_position_actuators"] == 0, "runtime report confirms unactuated task objects", errors)
    check(report["free_joint_qpos_writes_during_task"] == 0, "runtime report confirms no vial teleportation", errors)
    check(any(sample["lock_active"] for sample in samples) and any(not sample["lock_active"] for sample in samples), "trace observes lock transition", errors)
    check(max(sample["contact_count"] for sample in samples) == 5, "trace observes five simultaneous contacts", errors)

    if errors:
        print(f"\n{len(errors)} validation error(s).")
        raise SystemExit(1)
    print("\nStatic invariants and independent physics execution passed.")


if __name__ == "__main__":
    main()
