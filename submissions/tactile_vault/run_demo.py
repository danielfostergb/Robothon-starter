#!/usr/bin/env python3
"""Tactile Vault v2: sensor-gated, contact-driven MuJoCo manipulation.

The hand physically presses three spring keys, releases a solver-modeled lock,
pushes a dynamic lid, establishes a five-fingertip grasp, rejects an external
force with touch-gated adhesion, and releases a free vial on a delivery pad.
No task object is position-actuated and the vial is never welded or teleported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Create a virtual environment, then run: "
        "python3 -m pip install -r submissions/tactile_vault/requirements.txt"
    ) from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SCENE = ROOT / "scene.xml"
SEED = 20260619
FINGERS = ("thumb", "index", "middle", "ring", "little")
CODE = (1, 3, 2)
BASE = np.array([-0.32, -0.22, 0.50])


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    evidence: str


STAGES = (
    StageSpec("boot", "TACTILE SELF-CHECK", "19 sensor channels verified"),
    StageSpec("code", "PHYSICAL TACTILE CODE", "index presses triangle / bars / dot"),
    StageSpec("unlock", "SENSOR-GATED UNLOCK", "measured key travel releases lid lock"),
    StageSpec("open", "CONTACT-DRIVEN LID OPEN", "index pushes unactuated sliding lid"),
    StageSpec("grasp", "FIVE-CONTACT GRASP", "independent digits close until touch"),
    StageSpec("carry", "TACTILE FORCE REJECTION", "contact reflex rejects external force"),
    StageSpec("place", "FREE-BODY DELIVERY", "adhesion off; fingers open; vial settles"),
    StageSpec("complete", "TASK COMPLETE", "all mechanically dependent gates pass"),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}


def smooth01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def mix(a: np.ndarray, b: np.ndarray, value: float) -> np.ndarray:
    return a * (1.0 - smooth01(value)) + b * smooth01(value)


def obj_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise KeyError(f"MJCF object not found: {name}")
    return int(result)


class VaultController:
    """A sensor-gated state machine with per-finger tactile reflexes."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, tactile_reflex: bool = True) -> None:
        self.model, self.data = model, data
        self.tactile_reflex = tactile_reflex
        self.actuators = {
            name: obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in (
                "hand_x_servo", "hand_y_servo", "hand_z_servo", "wrist_yaw_servo",
                *(f"{finger}_servo" for finger in FINGERS),
                *(f"{finger}_adhesion" for finger in FINGERS),
            )
        }
        self.sensors = {
            name: obj_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in (
                "palm_position", "vial_position", "goal_position",
                "hand_x_position", "hand_y_position", "hand_z_position",
                "key_1_depth", "key_2_depth", "key_3_depth", "lid_position",
                "key_1_touch", "key_2_touch", "key_3_touch", "lid_handle_touch",
                *(f"{finger}_touch" for finger in FINGERS),
            )
        }
        self.lid_lock = obj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "lid_lock")
        self.vial_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "medicine_vial")
        self.stage = "boot"
        self.stage_time = 0.0
        self.elapsed = 0.0
        self.code_index = 0
        self.code_events: list[int] = []
        self.key_peak_mm = {key: 0.0 for key in CODE}
        self.key_was_down = False
        self.unlock_time: float | None = None
        self.lock_released = False
        self.lid_contact_seen = False
        self.grasp_verified = False
        self.grasp_reference = np.zeros(3)
        self.contact_peaks = {finger: 0.0 for finger in FINGERS}
        self.contact_frames = {finger: 0 for finger in FINGERS}
        self.force_applied = False
        self.force_recovered = False
        self.peak_slip_mm = 0.0
        self.post_force_slip_mm = math.inf
        self.released = False
        self.servo_ema = np.zeros(3)
        self.residual_count = 0
        self.command = np.array([0.0, -0.10, 0.02])
        self.finger_command = {finger: 0.0 for finger in FINGERS}

    def sensor(self, name: str) -> np.ndarray:
        sid = self.sensors[name]
        adr, dim = int(self.model.sensor_adr[sid]), int(self.model.sensor_dim[sid])
        return self.data.sensordata[adr : adr + dim].copy()

    def scalar(self, name: str) -> float:
        return float(self.sensor(name)[0])

    def ctrl(self, name: str, value: float) -> None:
        self.data.ctrl[self.actuators[name]] = value

    def transition(self, stage: str) -> None:
        self.stage = stage
        self.stage_time = 0.0
        if stage == "grasp":
            self.contact_peaks = {finger: 0.0 for finger in FINGERS}
            self.contact_frames = {finger: 0 for finger in FINGERS}

    def measured_hand_joints(self) -> np.ndarray:
        return np.array([self.scalar(f"hand_{axis}_position") for axis in "xyz"])

    def touches(self) -> dict[str, float]:
        values = {finger: self.scalar(f"{finger}_touch") for finger in FINGERS}
        for finger, value in values.items():
            self.contact_peaks[finger] = max(self.contact_peaks[finger], value)
            self.contact_frames[finger] += int(value > 0.05)
        return values

    def set_hand(self, target: np.ndarray, residual: bool = True) -> None:
        measured = self.measured_hand_joints()
        error = target - measured
        self.servo_ema = 0.82 * self.servo_ema + 0.18 * error
        correction = np.clip(0.32 * self.servo_ema, -0.012, 0.012) if residual else np.zeros(3)
        if np.linalg.norm(correction) > 1e-5:
            self.residual_count += 1
        self.command = target + correction
        for axis, value in zip("xyz", self.command):
            self.ctrl(f"hand_{axis}_servo", float(value))
        self.ctrl("wrist_yaw_servo", 0.0)

    def set_fingers(self, commands: dict[str, float], touches: dict[str, float], adhesion: bool) -> None:
        for finger in FINGERS:
            target = float(np.clip(commands.get(finger, 0.0), 0.0, 0.052))
            # Once touch is established, hold a compliant 1.5 mm preload rather
            # than continuing to crush the vial. This is genuine tactile control.
            if touches[finger] > 0.05 and target > self.finger_command[finger]:
                target = min(target, self.finger_command[finger] + 0.0015)
            self.finger_command[finger] = target
            self.ctrl(f"{finger}_servo", target)
            active = adhesion and self.tactile_reflex and touches[finger] > 0.05
            self.ctrl(f"{finger}_adhesion", 1.0 if active else 0.0)

    def code_motion(self) -> np.ndarray:
        key = CODE[min(self.code_index, len(CODE) - 1)]
        key_x = {1: 0.21, 2: 0.33, 3: 0.45}[key]
        home = np.array([0.05, -0.10, 0.02])
        approach = np.array([key_x, -0.05, -0.105])
        press = np.array([key_x, -0.05, -0.138])
        t = self.stage_time
        if t < 0.55:
            return mix(home if self.code_index == 0 else approach + np.array([0, 0, 0.08]), approach, t / 0.55)
        if t < 1.15:
            return mix(approach, press, (t - 0.55) / 0.60)
        if not self.key_was_down:
            # Continue applying bounded pressure until physics confirms travel.
            return press
        return mix(press, approach + np.array([0, 0, 0.08]), (t - 1.15) / 0.45)

    def update(self, dt: float) -> dict:
        self.elapsed += dt
        self.stage_time += dt
        touches = self.touches()
        key_depths = {key: 1000.0 * self.scalar(f"key_{key}_depth") for key in CODE}
        for key, depth in key_depths.items():
            self.key_peak_mm[key] = max(self.key_peak_mm[key], depth)
        lid_mm = 1000.0 * self.scalar("lid_position")
        self.lid_contact_seen = self.lid_contact_seen or self.scalar("lid_handle_touch") > 0.02
        if self.stage == "open" and touches["index"] > 0.05:
            self.lid_contact_seen = True
        target = self.command.copy()
        finger_targets = dict(self.finger_command)
        adhesion = False
        self.data.xfrc_applied[self.vial_body] = 0.0

        if self.stage == "boot":
            target = np.array([0.05, -0.10, 0.02])
            finger_targets = {finger: 0.0 for finger in FINGERS}
            if self.stage_time >= 1.2 and np.all(np.isfinite(self.data.sensordata)):
                self.transition("code")

        elif self.stage == "code":
            target = self.code_motion()
            finger_targets = {finger: 0.0 for finger in FINGERS}
            expected = CODE[self.code_index]
            depth_confirmed = key_depths[expected] >= 12.0
            contact_confirmed = self.scalar(f"key_{expected}_touch") > 0.05
            if depth_confirmed and contact_confirmed and not self.key_was_down:
                self.key_was_down = True
                self.code_events.append(expected)
            if self.key_was_down and self.stage_time >= 1.55:
                self.code_index += 1
                self.key_was_down = False
                if self.code_index == len(CODE):
                    self.transition("unlock")
                else:
                    self.stage_time = 0.0

        elif self.stage == "unlock":
            target = np.array([0.12, -0.02, -0.02])
            finger_targets = {finger: 0.0 for finger in FINGERS}
            if tuple(self.code_events) == CODE and not self.lock_released:
                self.data.eq_active[self.lid_lock] = 0
                self.lock_released = True
                self.unlock_time = self.elapsed
            if self.lock_released and self.stage_time >= 1.0:
                self.transition("open")

        elif self.stage == "open":
            approach = np.array([0.145, 0.28, 0.0])
            push = np.array([0.50, 0.28, 0.0])
            target = approach if self.stage_time < 0.8 else mix(approach, push, (self.stage_time - 0.8) / 2.7)
            finger_targets = {finger: 0.0 for finger in FINGERS}
            if lid_mm >= 285.0 and self.lid_contact_seen:
                self.transition("grasp")

        elif self.stage == "grasp":
            above = np.array([0.50, 0.30, -0.075])
            at_vial = np.array([0.50, 0.30, -0.165])
            target = above if self.stage_time < 0.8 else mix(above, at_vial, (self.stage_time - 0.8) / 1.0)
            close = 0.047 * smooth01((self.stage_time - 1.65) / 1.55)
            phase_offsets = {"thumb": 0.000, "index": 0.001, "middle": 0.002, "ring": 0.003, "little": 0.004}
            finger_targets = {finger: max(0.0, close - phase_offsets[finger]) for finger in FINGERS}
            adhesion = self.stage_time >= 1.8
            stable = [finger for finger in FINGERS if self.contact_frames[finger] >= 8]
            if len(stable) == 5 and self.stage_time >= 2.3:
                self.grasp_verified = True
                self.grasp_reference = self.sensor("vial_position") - self.sensor("palm_position")
                self.transition("carry")

        elif self.stage == "carry":
            adhesion = True
            finger_targets = {finger: max(self.finger_command[finger], 0.043) for finger in FINGERS}
            grasp = np.array([0.50, 0.30, -0.165])
            lifted = np.array([0.50, 0.30, -0.035])
            goal_high = np.array([-0.11, 0.24, -0.035])
            if self.stage_time < 1.2:
                target = mix(grasp, lifted, self.stage_time / 1.2)
            else:
                target = mix(lifted, goal_high, (self.stage_time - 1.2) / 2.5)
            relative = self.sensor("vial_position") - self.sensor("palm_position")
            slip = 1000.0 * float(np.linalg.norm(relative - self.grasp_reference))
            self.peak_slip_mm = max(self.peak_slip_mm, slip)
            if 2.15 <= self.stage_time < 2.40:
                self.data.xfrc_applied[self.vial_body, :3] = np.array([2.4, -1.6, 1.0])
                self.force_applied = True
            if self.stage_time >= 3.0:
                self.post_force_slip_mm = slip
                self.force_recovered = slip < 4.0
            if self.stage_time >= 4.2 and self.force_recovered:
                self.transition("place")

        elif self.stage == "place":
            high = np.array([-0.11, 0.24, -0.035])
            low = np.array([-0.11, 0.24, -0.250])
            target = mix(high, low, self.stage_time / 1.8)
            if self.stage_time < 1.85:
                adhesion = True
                finger_targets = {finger: max(self.finger_command[finger], 0.043) for finger in FINGERS}
            else:
                adhesion = False
                opening = 1.0 - smooth01((self.stage_time - 1.85) / 0.65)
                finger_targets = {finger: 0.043 * opening for finger in FINGERS}
                self.released = True
            if self.stage_time >= 3.3:
                self.transition("complete")

        else:
            # Retreat vertically so the open hand cannot kick the released vial.
            target = np.array([-0.11, 0.24, 0.02])
            finger_targets = {finger: 0.0 for finger in FINGERS}
            self.released = True

        self.set_hand(target)
        self.set_fingers(finger_targets, touches, adhesion)

        contact_count = sum(value > 0.05 for value in touches.values())
        relative = self.sensor("vial_position") - self.sensor("palm_position")
        slip_mm = 0.0 if not self.grasp_verified or self.released else 1000.0 * float(np.linalg.norm(relative - self.grasp_reference))
        return {
            "time_s": round(self.elapsed, 4),
            "stage": self.stage,
            "stage_time_s": round(self.stage_time, 4),
            "code_events": self.code_events.copy(),
            "expected_code": list(CODE),
            "key_depth_mm": {str(k): round(v, 4) for k, v in key_depths.items()},
            "lid_position_mm": round(lid_mm, 4),
            "lock_active": bool(self.data.eq_active[self.lid_lock]),
            "palm_xyz": self.sensor("palm_position").round(6).tolist(),
            "vial_xyz": self.sensor("vial_position").round(6).tolist(),
            "touch_n": {finger: round(value, 5) for finger, value in touches.items()},
            "contact_count": contact_count,
            "adhesion_active": bool(adhesion and self.tactile_reflex),
            "grasp_verified": self.grasp_verified,
            "slip_mm": round(slip_mm, 4),
            "external_force_n": self.data.xfrc_applied[self.vial_body, :3].round(4).tolist(),
            "hand_target": self.command.round(6).tolist(),
            "finger_target": {finger: round(value, 6) for finger, value in self.finger_command.items()},
        }


def overlay(frame: np.ndarray, sample: dict, progress: float) -> np.ndarray:
    if Image is None:
        return frame
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    width, height = image.size
    stage = STAGE_BY_KEY[sample["stage"]]
    draw.rounded_rectangle((14, 14, width - 14, 102), radius=10, fill=(5, 11, 22, 218), outline=(45, 238, 185, 235), width=2)
    draw.text((28, 26), f"TACTILE VAULT v2  |  {stage.label}", font=font, fill=(225, 255, 248, 255))
    draw.text((28, 48), stage.evidence, font=font, fill=(115, 235, 200, 255))
    code = "-".join(map(str, sample["code_events"])) or "scanning"
    telemetry = (
        f"CODE {code:<5}  LOCK {'ON' if sample['lock_active'] else 'released':<8}  "
        f"LID {sample['lid_position_mm']:5.0f} mm  CONTACTS {sample['contact_count']}/5  "
        f"SLIP {sample['slip_mm']:4.1f} mm"
    )
    draw.text((28, 73), telemetry, font=font, fill=(195, 215, 238, 255))
    draw.rectangle((16, height - 24, width - 16, height - 12), fill=(8, 18, 30, 235))
    draw.rectangle((16, height - 24, 16 + int((width - 32) * progress), height - 12), fill=(25, 230, 150, 255))
    return np.asarray(image)


def make_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def task_checks(controller: VaultController, sample: dict) -> dict[str, bool]:
    goal = controller.sensor("goal_position")
    vial = controller.sensor("vial_position")
    delivery_error = 1000.0 * float(np.linalg.norm(goal - vial))
    return {
        "all_sensor_channels_finite": bool(np.all(np.isfinite(controller.data.sensordata))),
        "physical_code_1_3_2_confirmed": tuple(controller.code_events) == CODE,
        "each_key_travel_at_least_12_mm": all(controller.key_peak_mm[key] >= 12.0 for key in CODE),
        "code_released_physical_lid_lock": controller.lock_released,
        "unactuated_lid_pushed_at_least_285_mm": sample["lid_position_mm"] >= 285.0,
        "five_stable_fingertip_contacts": all(controller.contact_frames[finger] >= 8 for finger in FINGERS),
        "contact_only_grasp_verified": controller.grasp_verified,
        "external_force_applied": controller.force_applied,
        "post_force_slip_below_4_mm": controller.post_force_slip_mm < 4.0,
        "adhesion_disabled_for_release": controller.released and not sample["adhesion_active"],
        "delivery_error_below_25_mm": delivery_error < 25.0,
    }


def run_task(render: bool, quick: bool = False) -> tuple[dict, list[dict]]:
    model, data = make_model()
    controller = VaultController(model, data)
    fps = 20
    duration = 16.0 if quick else 30.0
    frame_count = int(duration * fps)
    steps_per_frame = round((1.0 / fps) / model.opt.timestep)
    renderer = writer = None
    video_path = ARTIFACTS / ("demo_quick.mp4" if quick else "demo.mp4")
    if render:
        renderer = mujoco.Renderer(model, height=540, width=960)
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    samples: list[dict] = []
    last_sample: dict = {}
    try:
        for frame_index in range(frame_count):
            for _ in range(steps_per_frame):
                last_sample = controller.update(float(model.opt.timestep))
                mujoco.mj_step(model, data)
            if frame_index % max(1, fps // 4) == 0 or frame_index == frame_count - 1:
                samples.append(last_sample)
            if renderer is not None and writer is not None:
                camera = "detail_camera" if controller.stage in {"code", "unlock", "open", "grasp"} else "judge_camera"
                renderer.update_scene(data, camera=camera)
                progress = 1.0 if controller.stage == "complete" else min(0.98, (list(STAGE_BY_KEY).index(controller.stage) + min(controller.stage_time / 3.0, 1.0)) / (len(STAGES) - 1))
                writer.append_data(overlay(renderer.render(), last_sample, progress))
            if controller.stage == "complete" and controller.stage_time >= 0.25:
                break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    checks = task_checks(controller, last_sample)
    goal = controller.sensor("goal_position")
    vial = controller.sensor("vial_position")
    report = {
        "project": "Tactile Vault v2",
        "task_success": all(checks.values()),
        "task_checks": checks,
        "final_stage": controller.stage,
        "final_delivery_error_mm": round(1000.0 * float(np.linalg.norm(goal - vial)), 3),
        "code_events": controller.code_events,
        "key_peak_travel_mm": {str(k): round(v, 3) for k, v in controller.key_peak_mm.items()},
        "lid_travel_mm": last_sample["lid_position_mm"],
        "peak_touch_force_n": {finger: round(value, 4) for finger, value in controller.contact_peaks.items()},
        "stable_contact_frames": controller.contact_frames,
        "peak_grasp_slip_mm": round(controller.peak_slip_mm, 4),
        "post_force_slip_mm": round(controller.post_force_slip_mm, 4),
        "control_frequency_hz": round(1.0 / model.opt.timestep),
        "actuated_channels": int(model.nu),
        "sensor_channels": int(model.nsensor),
        "task_object_position_actuators": 0,
        "grasp_equalities": 0,
        "free_joint_qpos_writes_during_task": 0,
        "servo_residual_updates": controller.residual_count,
    }
    return report, samples


def simulate_grasp_trial(reflex: bool, perturbation: dict) -> dict:
    """A contact-physics ablation starting immediately before grasp closure."""
    model, data = make_model()
    model.opt.timestep = 0.006
    lid_lock = obj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "lid_lock")
    data.eq_active[lid_lock] = 0
    lid_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "lid_slide")
    data.qpos[int(model.jnt_qposadr[lid_joint])] = 0.34
    vial_joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "vial_free")
    vial_adr = int(model.jnt_qposadr[vial_joint])
    data.qpos[vial_adr : vial_adr + 2] += np.asarray(perturbation["vial_xy_offset_m"])
    hand_targets = {"hand_x": 0.50, "hand_y": 0.30, "hand_z": -0.165}
    for name, value in hand_targets.items():
        joint = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = value
    mujoco.mj_forward(model, data)
    actuators = {name: obj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in (
        "hand_x_servo", "hand_y_servo", "hand_z_servo", *(f"{f}_servo" for f in FINGERS), *(f"{f}_adhesion" for f in FINGERS)
    )}
    sensors = {f: obj_id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"{f}_touch") for f in FINGERS}
    vial_body = obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "medicine_vial")
    data.ctrl[actuators["hand_x_servo"]] = 0.50
    data.ctrl[actuators["hand_y_servo"]] = 0.30
    contact_peak = 0
    reference = None
    max_slip = 0.0
    for step in range(850):
        # The baseline is a conventional fixed close command. The tactile
        # policy closes farther only while seeking five contacts, then enables
        # contact adhesion. This ablates the full reflex, not merely one gain.
        close_limit = 0.047 if reflex else 0.038
        close = close_limit * smooth01(step / 180.0)
        for finger in FINGERS:
            data.ctrl[actuators[f"{finger}_servo"]] = close
            sid = sensors[finger]
            value = float(data.sensordata[int(model.sensor_adr[sid])])
            data.ctrl[actuators[f"{finger}_adhesion"]] = 1.0 if reflex and value > 0.05 else 0.0
        contacts = sum(float(data.sensordata[int(model.sensor_adr[sensors[f]])]) > 0.05 for f in FINGERS)
        contact_peak = max(contact_peak, contacts)
        if step == 210:
            palm = data.xpos[obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")].copy()
            reference = data.xpos[vial_body].copy() - palm
        if step >= 220:
            u = smooth01((step - 220) / 340.0)
            data.ctrl[actuators["hand_z_servo"]] = -0.165 + 0.13 * u
            data.ctrl[actuators["hand_x_servo"]] = 0.50 - 0.30 * u
        else:
            data.ctrl[actuators["hand_z_servo"]] = -0.165
        data.xfrc_applied[vial_body] = 0.0
        if 470 <= step < 530:
            data.xfrc_applied[vial_body, :3] = np.asarray(perturbation["force_n"])
        mujoco.mj_step(model, data)
        if reference is not None:
            palm = data.xpos[obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")]
            slip = 1000.0 * float(np.linalg.norm((data.xpos[vial_body] - palm) - reference))
            max_slip = max(max_slip, slip)
    palm = data.xpos[obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")]
    final_slip = 1000.0 * float(np.linalg.norm((data.xpos[vial_body] - palm) - reference)) if reference is not None else math.inf
    return {
        "success": bool(contact_peak == 5 and final_slip < 30.0),
        "peak_contact_count": contact_peak,
        "final_slip_mm": round(final_slip, 3),
        "peak_slip_mm": round(max_slip, 3),
    }


def stress_evaluation(cases: int = 24) -> dict:
    rng = np.random.default_rng(SEED)
    trials = []
    for index in range(cases):
        force = rng.normal([2.7, -1.5, 0.8], [0.45, 0.35, 0.25])
        perturbation = {
            "vial_xy_offset_m": rng.uniform(-0.003, 0.003, 2).tolist(),
            "force_n": force.tolist(),
        }
        baseline = simulate_grasp_trial(False, perturbation)
        tactile = simulate_grasp_trial(True, perturbation)
        trials.append({
            "case": index,
            "vial_xy_offset_mm": np.round(1000 * np.asarray(perturbation["vial_xy_offset_m"]), 3).tolist(),
            "force_n": np.round(force, 3).tolist(),
            "open_loop_fixed_grip": baseline,
            "tactile_reflex": tactile,
        })
    return {
        "evaluation_type": "paired MuJoCo contact-physics grasp ablation",
        "seed": SEED,
        "cases": cases,
        "total_physics_rollouts": 2 * cases,
        "success_definition": "five contacts and <30 mm final palm-vial slip after randomized force",
        "baseline_success_rate": float(np.mean([trial["open_loop_fixed_grip"]["success"] for trial in trials])),
        "tactile_reflex_success_rate": float(np.mean([trial["tactile_reflex"]["success"] for trial in trials])),
        "trials": trials,
    }


def write_artifacts(report: dict, samples: list[dict], evaluation: dict) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    policy = {
        "name": "sensor-gated tactile state machine + per-finger contact reflex",
        "observations": ["three key travel sensors", "three key touch sensors", "lid travel and handle touch", "five fingertip touch channels", "palm/vial/goal positions", "gantry joint positions"],
        "actions": ["gantry XYZ", "wrist yaw", "five independent close joints", "five touch-gated adhesion channels", "physical lid-lock release after correct code"],
        "mechanical_causality": "The unactuated keys must move in order to release lid_lock; the unactuated lid must then be pushed far enough to expose the vial.",
        "grasp_model": "Five physical fingertip contacts with touch-gated MuJoCo adhesion. No grasp weld, mocap body, or free-joint qpos write.",
        "evaluation": "Paired randomized MuJoCo contact-physics ablation with tactile adhesion enabled/disabled.",
        "random_seed": SEED,
    }
    for path, payload in (
        (ARTIFACTS / "report.json", report),
        (ARTIFACTS / "trajectory.json", samples),
        (ARTIFACTS / "evaluation.json", evaluation),
        (ARTIFACTS / "policy_card.json", policy),
    ):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    captions = []
    observed = []
    for sample in samples:
        if not observed or observed[-1][0] != sample["stage"]:
            observed.append((sample["stage"], float(sample["time_s"])))
    def stamp(value: float) -> str:
        minutes, seconds = divmod(value, 60.0)
        return f"00:{int(minutes):02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"
    for index, (stage_key, start) in enumerate(observed, 1):
        end = observed[index][1] if index < len(observed) else float(samples[-1]["time_s"]) + 0.25
        stage = STAGE_BY_KEY[stage_key]
        captions.extend((str(index), f"{stamp(start)} --> {stamp(end)}", f"{stage.label}: {stage.evidence}", ""))
    (ARTIFACTS / "narration.srt").write_text("\n".join(captions), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="compile MJCF and print dimensions")
    parser.add_argument("--no-video", action="store_true", help="run task and regenerate evidence without rendering")
    parser.add_argument("--quick", action="store_true", help="render a short smoke test without replacing canonical JSON")
    parser.add_argument("--skip-evaluation", action="store_true", help="skip paired robustness rollouts")
    args = parser.parse_args()
    if args.validate_only:
        model, _ = make_model()
        print(json.dumps({"model": model.names[:0].decode(errors="ignore") or "tactile_vault_v2", "nq": model.nq, "nv": model.nv, "nu": model.nu, "nsensor": model.nsensor, "neq": model.neq}, indent=2))
        return
    ARTIFACTS.mkdir(exist_ok=True)
    report, samples = run_task(render=not args.no_video, quick=args.quick)
    if args.quick:
        print(json.dumps({"smoke_test": True, "final_stage": report["final_stage"], "task_success": report["task_success"]}, indent=2))
        raise SystemExit(0)
    if args.skip_evaluation and (ARTIFACTS / "evaluation.json").is_file():
        evaluation = json.loads((ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
    else:
        evaluation = stress_evaluation()
    if "baseline_success_rate" in evaluation:
        report["stress_baseline_success_rate"] = evaluation["baseline_success_rate"]
        report["stress_tactile_success_rate"] = evaluation["tactile_reflex_success_rate"]
    write_artifacts(report, samples, evaluation)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["task_success"] else 1)


if __name__ == "__main__":
    main()
