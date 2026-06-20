# Tactile Vault v2 — Judge Brief

## Thirty-second inspection path

1. Watch `artifacts/demo.mp4`: live telemetry shows physical code `1-3-2`, lock release, lid travel, five contacts, force/slip recovery, and free-body release.
2. Run `python3 submissions/tactile_vault/validate_submission.py`: it checks structural invariants and independently executes the full physics task.
3. Open `artifacts/report.json`: all eleven mechanical gates pass; post-force slip is below 4 mm.
4. Open `artifacts/evaluation.json`: 48 paired contact-physics rollouts compare open-loop and tactile-reflex grasping.

## The important causal chain

```text
physical key travel 1 -> 3 -> 2
        -> release lid_lock
        -> hand pushes unactuated lid >285 mm
        -> vial becomes accessible
        -> five stable fingertip contacts
        -> touch-gated adhesion + force recovery
        -> adhesion off + free-body tray delivery
```

The scene has no actuator on any key, lid, or vial. The grasp uses no equality, mocap attachment, or qpos write. The only equality is the initially active lid lock, released after the measured code.

## Rubric evidence

| Criterion | Direct evidence |
|---|---|
| Runnability | Exact pins, local assets, deterministic seed, quick/headless/dynamic-validator modes |
| MuJoCo depth | 14 actuators, 19 sensors, compliant keys, dynamic lid/vial, contact, friction, adhesion, force, lock equality |
| Task design | Eight mechanically dependent, measurable stages in an emergency-access scenario |
| Control | 250 Hz sensor gates, per-finger tactile preload/adhesion, randomized paired physics evaluation |
| Dexterity | Five independent digits and five required stable physical contacts; no grasp weld |
| Engineering | Typed compact code, explicit invariants, configuration, validator, structured artifacts |
| Presentation | 960x540 video, camera switching, live causal telemetry, progress, captions |
| Innovation | Tactile access control and dexterous recovery remain directly inspectable in physics |
