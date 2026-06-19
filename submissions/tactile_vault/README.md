# Tactile Vault

Tactile Vault is a self-contained MuJoCo dexterity demo for Robothon 2026. A procedural five-finger hand retrieves medicine from an emergency vault without relying on vision: it scans the tactile dial, enters a three-direction code, confirms the spring latch, forms an opposed grasp, detects and corrects a simulated slip, and places the vial on a delivery pad.

The idea is deliberately small. The depth comes from observable control behavior rather than a long list of disconnected tasks.

## One-command run

From the repository root:

```bash
python3 -m pip install -r submissions/tactile_vault/requirements.txt
python3 submissions/tactile_vault/run_demo.py
```

Fast checks:

```bash
# Compile and inspect the MuJoCo model
python3 submissions/tactile_vault/run_demo.py --validate-only

# Run physics and regenerate JSON without OpenGL/FFmpeg
python3 submissions/tactile_vault/run_demo.py --no-video

# Render an 8-second smoke-test video
python3 submissions/tactile_vault/run_demo.py --quick

# Validate submission structure
python3 submissions/tactile_vault/validate_submission.py
```

The default command creates a 30-second `artifacts/demo.mp4` plus trajectory, policy, evaluation, narration, and report files. The run uses a fixed seed (`20260619`) and makes no network calls.

## Task and success criteria

1. Bring all 13 MuJoCo sensor channels online.
2. Align the index fingertip with the tactile dial.
3. Track the combination `+70°, -35°, +110°` using dial-angle feedback.
4. Depress the spring latch by at least 30 mm.
5. Close thumb, index, middle, ring, and little finger in an opposed phased grasp.
6. Reject a deterministic 29 mm slip disturbance during transfer.
7. Place the medicine vial within 20 mm of the green delivery target.

`artifacts/report.json` records the final pass/fail result. `artifacts/evaluation.json` compares the controller with and without residual correction over 32 fixed-seed disturbances.

## Robot and MuJoCo use

- Six-axis gantry/wrist motion plus eleven independently actuated finger joints.
- Thumb opposition and non-synchronous finger closure for stable enveloping grasp posture.
- A free-joint medicine vial, a hinge dial, a spring-loaded slide latch, and collision geometry.
- Frame-position, joint-position, joint-velocity, and seven touch sensors.
- 250 Hz implicit MuJoCo simulation with position actuators, damping, friction, and contact dynamics.

## Controller

The controller combines a deterministic seven-stage task prior with a feedback residual. Each tick reads MuJoCo palm, vial, dial, latch, and fingertip measurements. An exponential moving-average pose observer and slip observer correct the nominal gantry command; the dial command closes the loop on measured angle. Per-sample errors, residual actions, touch values, confidence, and slip estimates are exported to `trajectory.json`.

Honest scope: the high-level sequence is scripted for repeatable judging, and the vial uses a kinematic attachment after stable grasp. Gantry, wrist, fingers, dial, and latch are driven through MuJoCo actuators and read through MuJoCo sensors. This is a control demonstration, not a claim of a learned policy.

## Files

| File | Purpose |
|---|---|
| `scene.xml` | Procedural MJCF robot, vault, sensors, actuators, and task objects |
| `run_demo.py` | Controller, stress evaluation, artifact generation, and rendering |
| `config.json` | Human-readable task and controller settings |
| `validate_submission.py` | Static structure and syntax checks |
| `JUDGE_BRIEF.md` | Fast rubric-to-evidence map |
| `PR_DESCRIPTION.md` | Ready-to-paste pull-request description |
| `rubric_scorecard.json` | Unofficial target-score self-audit with direct evidence |
| `submission_manifest.json` | Machine-readable commands and artifact index |
| `artifacts/demo.mp4` | Generated task presentation |
| `artifacts/report.json` | Final task metrics |
| `artifacts/evaluation.json` | 32-rollout baseline/residual comparison |
| `artifacts/policy_card.json` | Controller observations, actions, and honest scope |
| `artifacts/trajectory.json` | Time-indexed sensor and action evidence |
| `artifacts/narration.srt` | Accessible stage captions |

## Rubric map

| Official criterion | Evidence |
|---|---|
| Runnability | One command, fixed seed, quick/headless/validation modes, no external assets |
| Depth of MuJoCo use | MJCF, 19 actuators, 13 sensors, contact, free/hinge/slide joints, physical dial/latch |
| Task design | Clear safety scenario, long-horizon dependency, numerical completion criteria |
| Control | Sensor residual, slip observer, closed-loop dial control, structured trajectory export |
| Dexterous manipulation | Five named fingers, thumb opposition, independent phasing, fine dial and grasp postures |
| Engineering quality | Compact source, config, validator, documented scope, deterministic evaluation |
| Presentation | Generated 30-second video with stage/progress/telemetry overlays and SRT captions |
| Innovation | Tactile-only emergency access and recovery evidence designed for degraded visibility |

## Registration before submission

Replace the placeholders in `registration.json` with your platform-issued UUID and participant name, then put the exact same UUID in the pull-request description. Do not reuse a UUID from another submission.

## Limitations and next steps

- The post-contact vial attachment is kinematic; a future version can switch a MuJoCo weld equality dynamically after a verified multi-contact grasp.
- The residual is analytic and deterministic, not learned.
- Future work could randomize vault geometry and train a tactile policy from the exported trajectories.
