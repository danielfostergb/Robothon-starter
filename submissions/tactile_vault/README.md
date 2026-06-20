# Tactile Vault v2

Tactile Vault is a mechanically coupled MuJoCo dexterity task for degraded-visibility emergency access. A five-digit hand physically enters a raised-shape code, unlocks and pushes open a sliding lid, grasps a free medicine vial with five measured contacts, rejects an external force, and releases the vial into a delivery tray.

The defining invariant is simple: **the task cannot advance on timing alone**. Measured key travel produces the code; the correct code releases the physical lid lock; measured lid travel exposes the vial; all five fingertip contacts enable the tactile grasp reflex.

## Run

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r submissions/tactile_vault/requirements.txt
.venv/bin/python submissions/tactile_vault/run_demo.py
```

Fast checks:

```bash
.venv/bin/python submissions/tactile_vault/run_demo.py --validate-only
.venv/bin/python submissions/tactile_vault/run_demo.py --no-video
.venv/bin/python submissions/tactile_vault/run_demo.py --quick
.venv/bin/python submissions/tactile_vault/validate_submission.py
```

The default command regenerates the demo video and every JSON evidence file. Runtime is deterministic from seed `20260619`; no network or external asset is used after installation.

## Mechanically dependent task

1. Validate 19 live sensor channels.
2. Press the three unactuated spring keys in tactile-shape order `1 -> 3 -> 2`; each must travel at least 12 mm while reporting contact.
3. Release `lid_lock` only after the measured code matches.
4. Push the unactuated lid at least 285 mm through hand-handle contact.
5. Lower five independently actuated radial digits around the now-accessible vial.
6. Require stable touch on thumb, index, middle, ring, and little finger before enabling per-contact adhesion.
7. Reject a physical `3.05 N` force and recover to less than 4 mm palm-vial slip.
8. Disable all adhesion channels, open every finger, and let the free body settle within 25 mm of the delivery target.

## MuJoCo and control depth

- 250 Hz observe-decide-act-step loop with a sensor-gated state machine.
- Free, slide, and hinge joints; compliant key springs; a frictional dynamic lid; collision geometry; applied body forces; runtime equality state for the **lock only**.
- 14 actuators: gantry/wrist motion, five independent close joints, and five touch-gated adhesion channels.
- 19 sensors: frame positions, gantry/key/lid joint positions, key/handle touch, and five independent fingertip touch channels.
- Zero task-object position actuators, zero grasp equalities, zero task-time free-joint writes.
- Touch feedback limits per-finger preload and gates each adhesion channel.

## Evaluation

`artifacts/evaluation.json` contains 24 paired perturbation cases (48 actual MuJoCo rollouts). Every pair uses identical vial offsets and randomized forces. It compares an open-loop fixed grip against the complete five-contact tactile reflex.

The evaluation is deliberately manipulation-specific: success requires five physical contacts and less than 30 mm final palm-vial slip after the randomized force. It is not a proxy based on commanded joint tracking.

## Evidence

| File | What it proves |
|---|---|
| `artifacts/demo.mp4` | End-to-end generated behavior with detail/wide cameras and live telemetry |
| `artifacts/report.json` | Eleven independent mechanical success gates and invariants |
| `artifacts/trajectory.json` | Time-indexed key travel, lock state, lid travel, all five touches, adhesion, slip, and forces |
| `artifacts/evaluation.json` | Paired randomized contact-physics ablation |
| `artifacts/policy_card.json` | Observations, actions, causal mechanics, and grasp model |
| `validate_submission.py` | Static invariants plus an independent end-to-end physics execution |

## Honest scope

The high-level target locations are analytic and known; this is not a learned policy or a perception benchmark. MuJoCo adhesion models a controllable high-friction fingertip material and is enabled independently only at measured contacts. The sole equality constraint models the closed lid lock and never touches the vial or hand.
