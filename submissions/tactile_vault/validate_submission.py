#!/usr/bin/env python3
"""Fast local checks for registration, MJCF structure, Python, and artifacts."""

from __future__ import annotations

import json
import py_compile
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check(condition: bool, message: str, errors: list[str]) -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    registration = json.loads((ROOT / "registration.json").read_text())
    check(registration["project_name"] == "Tactile Vault", "registration project name", errors)
    uuid_ready = registration["uuid"] != "PASTE-YOUR-UUID-HERE"
    print(f"[{'PASS' if uuid_ready else 'WARN'}] registration UUID replaced before PR")

    tree = ET.parse(ROOT / "scene.xml")
    root = tree.getroot()
    actuators = root.findall("./actuator/*")
    sensors = root.findall("./sensor/*")
    touch = root.findall("./sensor/touch")
    check(len(actuators) >= 19, f"MJCF actuator depth ({len(actuators)})", errors)
    check(len(sensors) >= 13, f"MJCF sensor depth ({len(sensors)})", errors)
    check(len(touch) >= 7, f"touch sensing ({len(touch)})", errors)
    check(root.find(".//freejoint") is not None, "free-body medicine vial", errors)
    check(len(root.findall(".//joint")) >= 19, "articulated five-finger hand", errors)

    py_compile.compile(str(ROOT / "run_demo.py"), doraise=True)
    check(True, "Python source compiles", errors)
    for name in (
        "README.md", "JUDGE_BRIEF.md", "PR_DESCRIPTION.md", "config.json", "requirements.txt",
        "rubric_scorecard.json", "submission_manifest.json",
    ):
        check((ROOT / name).is_file(), f"required file: {name}", errors)

    if errors:
        print(f"\n{len(errors)} blocking validation error(s).")
        raise SystemExit(1)
    suffix = "Registration UUID is ready." if uuid_ready else "Replace the UUID before opening the PR."
    print(f"\nStatic validation passed. {suffix}")


if __name__ == "__main__":
    main()
