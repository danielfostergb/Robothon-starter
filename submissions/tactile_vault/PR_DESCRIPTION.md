Registration UUID: 1475c5ac-4357-44bb-9f56-2aa775114462

## Tactile Vault v2

A five-digit MuJoCo hand retrieves emergency medicine without vision by physically entering a raised-shape code, releasing a mechanical lid lock, pushing the unactuated lid, establishing five tactile contacts, rejecting an external force, and delivering the free vial.

### Why this submission is technically deep

- Every stage is mechanically dependent; timing alone cannot complete the task.
- Keys, lid, and vial have no position actuators.
- Five independent touch channels gate five independent adhesion channels.
- The grasp has no equality, mocap attachment, or free-joint write.
- A 250 Hz sensor-gated state machine exposes its observations and actions in `trajectory.json`.
- 48 paired randomized MuJoCo contact-physics rollouts compare an open-loop grip with the tactile reflex.
- The included validator independently executes the end-to-end physics task.

### Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r submissions/tactile_vault/requirements.txt
.venv/bin/python submissions/tactile_vault/run_demo.py
```

Fast headless verification:

```bash
.venv/bin/python submissions/tactile_vault/validate_submission.py
```

### Evidence

- `artifacts/demo.mp4` — generated end-to-end demo with automatic camera switching and live causal telemetry
- `artifacts/report.json` — eleven passing task gates and simulator invariants
- `artifacts/evaluation.json` — paired open-loop/tactile contact-physics ablation
- `artifacts/trajectory.json` — key, lock, lid, touch, adhesion, force, slip, and action trace
- `artifacts/policy_card.json` — controller and grasp disclosure

### Honest scope

The high-level target locations are analytic rather than learned. Adhesion models controllable fingertip material and activates only at measured contacts. The only equality models the lid lock and never constrains the vial or hand.
