# 🤖 Tactile Vault v2

**FFAI Robothon 2026** — Freestyle Category

> **A 14-actuator, five-digit MuJoCo hand completes degraded-visibility emergency medicine retrieval through physical code entry, sensor-gated unlocking, contact-driven lid opening, five-contact grasping, tactile force rejection, and free-body delivery.**

---

## 📋 Project Overview

This project implements a self-contained emergency-access task in which a dexterous robot hand opens a mechanically locked vault and retrieves a medicine vial. The system combines:

- **Mechanically Gated Vault Access**: Three unactuated spring keys must be physically pressed in the measured order `1 → 3 → 2` before the lid lock can release
- **Contact-Driven Lid Opening**: The hand pushes an unactuated sliding lid through measured handle contact instead of commanding the lid pose
- **Independent Tactile Grasping**: Five fingertip sensors gate five adhesion channels independently before the free vial can be carried
- **Closed-Loop Force Rejection**: Contact-aware finger preload and adhesion reject an applied 3.05 N disturbance and recover the grasp

### Key Achievements

- **8/8 task stages passed** (100% task completion)
- **11/11 terminal checks passed**
- **Physical code**: `1 → 3 → 2`, with every key exceeding 12 mm travel
- **Lid travel**: 420.0 mm through contact-driven manipulation
- **Force recovery**: 30.7626 mm peak slip reduced to 1.9512 mm
- **Tactile evaluation**: 24/24 successful paired rollouts

---

## 🎯 Task Summary (8/8 Passed)

| # | Task | Type | Description |
|---|------|------|-------------|
| 1 | Tactile Self-Check | Sensor Validation | Verify all 19 live sensor channels before motion begins |
| 2 | Physical Tactile Code | Contact Sequencing | Press the three spring keys in raised-shape order `1 → 3 → 2` |
| 3 | Sensor-Gated Unlock | Mechanical Interlock | Release `lid_lock` only after the measured code is correct |
| 4 | Contact-Driven Lid Open | Physical Manipulation | Push the unactuated sliding lid far enough to expose the vial |
| 5 | Five-Contact Grasp | Tactile Grasping | Establish stable contact on thumb, index, middle, ring, and little finger |
| 6 | Tactile Force Rejection | Closed-Loop Recovery | Reject an applied external force and recover to less than 4 mm slip |
| 7 | Free-Body Delivery | Pick and Place | Disable adhesion, open every finger, and let the vial settle on the target |
| 8 | Verify and Export | Validation | Check all mechanical gates and export the complete evidence package |

---

## 🔬 Technical Innovations

### 1. Measured Mechanical Code Gate

```python
if key_touch and key_travel_mm >= minimum_key_travel_mm:
    observed_code.append(key_id)

if observed_code == [1, 3, 2]:
    lid_lock_active = False
```

- Code progress comes from physical key travel and touch sensing
- Each key is an unactuated, spring-loaded body
- Timing alone cannot unlock the vault

### 2. Independent Touch-Gated Adhesion

```python
for finger in fingers:
    adhesion[finger] = tactile_reflex and touch_force[finger] > 0.05
```

- Five fingertip forces are measured separately
- Each adhesion channel activates only at its corresponding physical contact
- The vial is never attached by a grasp equality, mocap body, or free-joint pose write

### 3. Contact-Driven Manipulation

- The lid has no position actuator
- Lid opening requires measured index contact with the handle
- The vial remains a free body throughout grasping, transport, disturbance, and release
- Task-object position actuators remain at zero

### 4. Closed-Loop Force Recovery

- A physical force vector of `[2.4, -1.6, 1.0] N` is applied during transport
- Per-finger tactile feedback preserves all five contacts
- Recovery succeeds only when palm-vial slip falls below 4 mm

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Task Stages Completed | 8/8 |
| Terminal Checks | 11/11 |
| Success Rate | 100% |
| Key Peak Travel | 23.517 mm, 23.692 mm, 25.429 mm |
| Lid Travel | 420.0 mm |
| Applied Disturbance | 3.05 N |
| Peak Grasp Slip | 30.7626 mm |
| Post-Force Slip | 1.9512 mm |
| Final Delivery Error | 18.843 mm |
| Tactile Reflex Success | 100% (24/24) |
| Open-Loop Success | 20.8% (5/24) |
| Control Frequency | 250 Hz |

---

## 🛠️ Technical Specifications

### Robot Configuration

- **Actuators**: 14 total channels
- **Hand**: Five radial digits with five independent closing joints
- **Positioning**: 3-DOF XYZ gantry
- **Orientation**: Actuated wrist yaw
- **Tactile Grasp**: Five independently controlled contact-adhesion channels

### MuJoCo Model

- **Timestep**: 4 ms (250 Hz simulation and control)
- **Contact Model**: Friction, gravity, compliant springs, collision geometry, and applied body forces
- **Sensors**: 19 channels covering frame positions, joint travel, handle/key touch, and five fingertip contacts
- **Task Objects**: Three spring keys, an unactuated sliding lid, and a free-joint medicine vial
- **Equality Constraints**: One physical lid lock; zero grasp equalities
- **Mocap Bodies**: None

### Control Stack

- **Task Planner**: Deterministic eight-stage sensor-gated state machine
- **Tactile Control**: Five independent contact-gated adhesion loops
- **Mechanical Gate**: Correct measured key sequence releases the lid lock
- **Force Recovery**: Per-finger preload and adhesion respond to physical contact during disturbance
- **Evaluation**: 24 paired fixed-seed tactile-reflex versus open-loop rollouts

---

## 📁 File Structure

```text
submissions/tactile_vault/
├── run_demo.py                 # Controller, simulation, evaluation, and artifact generation
├── validate_submission.py       # Static checks and independent end-to-end physics validator
├── scene.xml                    # Five-digit hand and tactile vault MuJoCo scene
├── config.json                  # Seed, control, force, and success thresholds
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── PR_DESCRIPTION.md            # Pull-request summary
├── submission_manifest.json     # Entrypoints and generated-evidence manifest
├── registration.json            # UUID: 1475c5ac-4357-44bb-9f56-2aa775114462
└── artifacts/
    ├── demo.mp4                 # Generated 20.7-second demonstration
    ├── trajectory.json          # Time-indexed observation and action trajectory
    ├── report.json              # Runtime metrics and 11 terminal checks
    ├── evaluation.json          # Paired contact-physics ablation rollouts
    ├── policy_card.json         # Policy observations, actions, and grasp disclosure
    └── narration.srt            # Stage-aligned demonstration captions
```

---

## 🚀 Quick Start

Run from the repository root:

```bash
# Create an isolated environment and install dependencies
python3 -m venv .venv
.venv/bin/python -m pip install -r submissions/tactile_vault/requirements.txt

# Run the full deterministic demo and 24-case paired evaluation
.venv/bin/python submissions/tactile_vault/run_demo.py

# Validate the submission and execute an independent physics run
.venv/bin/python submissions/tactile_vault/validate_submission.py
```

For a faster smoke test with fewer evaluation cases:

```bash
.venv/bin/python submissions/tactile_vault/run_demo.py --quick
```

Use `--no-video` when only physics and controller validation are required, or `--validate-only` to validate existing generated artifacts. The run is deterministic from seed `20260619` and requires no network or external assets after installation.

---

## 📈 Evaluation Results

`artifacts/evaluation.json` records 24 paired perturbation cases, for 48 actual MuJoCo physics rollouts. Each pair receives identical randomized vial offsets and force vectors:

- The five-contact tactile reflex succeeds in 24/24 cases (100%)
- The open-loop fixed grip succeeds in 5/24 cases (20.8%)
- Success requires all five physical contacts and less than 30 mm final palm-vial slip
- Every rollout uses deterministic seed `20260619`

Runtime task measurements and all 11 terminal conditions are recorded separately in `artifacts/report.json`.

---

## 🏆 Why This Submission Stands Out

1. **Mechanically Dependent Long-Horizon Task**: Physical code entry, unlocking, lid opening, grasping, transport, disturbance rejection, and delivery form one causal sequence
2. **Measured Dexterity**: Five independent touch signals directly gate grasp behavior and force response
3. **No Direct Object Control**: The keys and lid are unactuated, the vial remains free, and no grasp equality or task-time pose write is used
4. **Quantitative Validation**: Eleven terminal gates and a 24-case paired contact-physics ablation support the result
5. **Reproducible Evidence**: One deterministic run generates video, trajectory, runtime report, evaluation data, policy disclosure, and captions

---


## 📝 License

This project is submitted for the FFAI Robothon 2026 competition.
