## 🤖 Tactile Vault v2

**UUID**: `1475c5ac-4357-44bb-9f56-2aa775114462`

### Key Innovations

- **Mechanically Gated Vault Access**: A five-digit hand physically enters the raised-shape code `1 → 3 → 2`, releases the lid lock, and pushes open an unactuated sliding lid
- **Five-Contact Tactile Grasp**: Independent touch sensing on every digit gates five adhesion channels before the free medicine vial can be lifted
- **Force-Rejection Reflex**: Contact-aware preload and adhesion reject a physical 3.05 N disturbance while recovering to less than 4 mm palm-vial slip
- **Sensor-Gated Closed-Loop Control**: A deterministic 250 Hz observe-decide-act loop completes the task without task-object position actuators, grasp equalities, or free-joint pose writes

### Tasks (11/11 passed)

1. Live Sensor Validation
2. Physical Code Entry (`1 → 3 → 2`)
3. Minimum Key Travel
4. Physical Lid Lock Release
5. Unactuated Lid Opening
6. Five Stable Fingertip Contacts
7. Contact-Only Grasp Verification
8. External Force Application
9. Post-Force Slip Recovery
10. Adhesion-Free Release
11. Delivery Placement

### Performance

- Key Peak Travel: 23.517 mm, 23.692 mm, and 25.429 mm
- Lid Travel: 420.0 mm
- Applied Disturbance: 3.05 N
- Post-Force Slip: 1.9512 mm
- Final Delivery Error: 18.843 mm
- Tactile Reflex Success Rate: 100% across 24 randomized cases
- Open-Loop Baseline Success Rate: 20.8%
- Control Frequency: 250 Hz

### Files

- run_demo.py - Sensor-gated controller, evaluation, and video generation
- scene.xml - MuJoCo vault, five-digit hand, and manipulation model
- config.json - Motion, contact, force, and success thresholds
- artifacts/demo.mp4 - End-to-end simulation video with live telemetry
- artifacts/report.json - Measured results and 11 terminal task checks
- artifacts/evaluation.json - Paired randomized contact-physics ablation
- artifacts/trajectory.json - Recorded sensor, contact, force, and action trajectory
- artifacts/policy_card.json - Controller observations, actions, and grasp disclosure
- registration.json - UUID and participant registration
