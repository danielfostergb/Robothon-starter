from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
    import mujoco
except ImportError as exc:
    raise SystemExit(
        "Missing demo dependency. Install with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        f"Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URDF = ROOT / "assets" / "Futurist" / "futurist.urdf"
DEFAULT_OUTPUT = ROOT / "outputs" / "futurist_demo.mp4"
DEFAULT_TRAJECTORY = ROOT / "outputs" / "futurist_trajectory.json"
SHOWCASE_BASE_HEIGHT = 0.96
CARRY_WALK_BASE_HEIGHT = 0.98
SIDE_JOINTS = {
    "left": {
        "hip_roll": "idx01_left_hip_roll",
        "hip_yaw": "idx02_left_hip_yaw",
        "hip_pitch": "idx03_left_hip_pitch",
        "tarsus": "idx04_left_tarsus",
        "toe_pitch": "idx05_left_toe_pitch",
        "shoulder_pitch": "idx13_left_arm_joint1",
        "shoulder_roll": "idx14_left_arm_joint2",
        "shoulder_yaw": "idx15_left_arm_joint3",
        "elbow": "idx16_left_arm_joint4",
        "wrist": "idx19_left_arm_joint7",
    },
    "right": {
        "hip_roll": "idx07_right_hip_roll",
        "hip_yaw": "idx08_right_hip_yaw",
        "hip_pitch": "idx09_right_hip_pitch",
        "tarsus": "idx10_right_tarsus",
        "toe_pitch": "idx11_right_toe_pitch",
        "shoulder_pitch": "idx20_right_arm_joint1",
        "shoulder_roll": "idx21_right_arm_joint2",
        "shoulder_yaw": "idx22_right_arm_joint3",
        "elbow": "idx23_right_arm_joint4",
        "wrist": "idx26_right_arm_joint7",
    },
}


def referenced_meshes(urdf_path: Path) -> list[Path]:
    text = urdf_path.read_text(encoding="utf-8")
    mesh_names = re.findall(r'<mesh\s+filename="([^"]+)"', text)
    return [urdf_path.parent / mesh_name for mesh_name in mesh_names]


def missing_meshes(urdf_path: Path) -> list[str]:
    return [path.name for path in referenced_meshes(urdf_path) if not path.exists()]


def mujoco_ready_urdf(source_urdf: Path, temp_dir: Path) -> Path:
    text = source_urdf.read_text(encoding="utf-8")
    compiler = (
        "  <mujoco>\n"
        f'    <compiler meshdir="{source_urdf.parent}" discardvisual="false"/>\n'
        "  </mujoco>\n"
    )
    if "<mujoco>" not in text:
        text = re.sub(r"(<robot[^>]*>\n)", r"\1" + compiler, text, count=1)

    output_path = temp_dir / source_urdf.name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def build_model(urdf_path: Path, scenario: str) -> mujoco.MjModel:
    with tempfile.TemporaryDirectory() as tmp:
        ready_urdf = mujoco_ready_urdf(urdf_path, Path(tmp))
        spec = mujoco.MjSpec.from_file(str(ready_urdf))
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        spec.option.timestep = 0.002
        spec.option.gravity = [0.0, 0.0, -9.81]

        base = spec.body("base_link")
        if base is None:
            raise ValueError("Missing base_link body in Futurist URDF")
        base.add_freejoint(name="floating_base_joint")

        world = spec.worldbody
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            rgba=[0.05, 0.06, 0.08, 1.0],
        )
        if scenario == "carry_walk":
            world.add_geom(
                name="walkway",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0, 0.0, 0.004],
                size=[0.95, 0.26, 0.004],
                rgba=[0.10, 0.13, 0.18, 1.0],
            )
            world.add_geom(
                name="pickup_pad",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[-0.42, 0.0, 0.012],
                size=[0.12, 0.18, 0.006],
                rgba=[0.05, 0.32, 1.0, 0.8],
            )
            world.add_geom(
                name="drop_pad",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.50, 0.0, 0.012],
                size=[0.12, 0.18, 0.006],
                rgba=[0.10, 0.85, 0.35, 0.8],
            )
            box = world.add_body(name="carry_box", pos=[-0.42, -0.03, 0.52])
            box.add_freejoint(name="carry_box_freejoint")
            box.add_geom(
                name="carry_box_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.12, 0.09, 0.08],
                rgba=[1.0, 0.58, 0.12, 1.0],
            )
        world.add_light(pos=[0, -1.2, 2.4], dir=[0, 0.35, -1], diffuse=[1, 1, 1])
        world.add_light(pos=[-1.2, 0.8, 1.6], dir=[0.5, -0.3, -1], diffuse=[0.5, 0.55, 0.7])

        return spec.compile()


def set_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return

    qpos_addr = int(model.jnt_qposadr[joint_id])
    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = float(np.clip(value, low, high))
    data.qpos[qpos_addr] = value


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if value <= edge0:
        return 0.0
    if value >= edge1:
        return 1.0
    x = (value - edge0) / (edge1 - edge0)
    return x * x * (3.0 - 2.0 * x)


def set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: tuple[float, float, float],
    yaw: float = 0.0,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return

    qpos_addr = int(model.jnt_qposadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def apply_showcase_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    progress = min(1.0, max(0.0, time_s / max(duration_s, 0.1)))
    wave = math.sin(2.0 * math.pi * 0.8 * time_s)

    data.qpos[0] = -0.10 + 0.20 * progress
    data.qpos[1] = 0.0
    data.qpos[2] = SHOWCASE_BASE_HEIGHT + 0.02 * math.sin(2.0 * math.pi * 0.5 * time_s)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Futurist joint names are exported as idxNN_*; keep this demo independent
    # from exact robot dimensions by using gentle, symmetric showcase motions.
    for side, sign in (("left", 1.0), ("right", -1.0)):
        joints = SIDE_JOINTS[side]
        set_joint(model, data, joints["hip_roll"], 0.04 * sign * wave)
        set_joint(model, data, joints["hip_pitch"], 0.10 * math.sin(2.0 * math.pi * time_s))
        set_joint(model, data, joints["shoulder_pitch"], 0.35 * sign * wave)
        set_joint(model, data, joints["elbow"], -0.45 + 0.12 * wave)

    set_joint(model, data, "idx27_head_joint1", 0.18 * math.sin(1.5 * time_s))
    set_joint(model, data, "idx28_head_joint2", 0.08 * math.sin(2.0 * time_s))
    set_freejoint_pose(model, data, "carry_box_freejoint", (-0.42, -0.03, 0.16), 0.0)

    mujoco.mj_forward(model, data)


def apply_carry_walk_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    walk = smoothstep(0.5, duration_s - 1.1, time_s)
    pickup = smoothstep(0.8, 1.7, time_s)
    place = smoothstep(duration_s - 1.4, duration_s - 0.45, time_s)
    gait = 2.0 * math.pi * 1.35 * time_s
    base_x = -0.55 + 1.10 * walk
    base_y = 0.0
    base_z = CARRY_WALK_BASE_HEIGHT + 0.025 * math.sin(gait) * (1.0 - 0.35 * pickup)
    yaw = 0.05 * math.sin(0.7 * gait)

    data.qpos[0] = base_x
    data.qpos[1] = base_y
    data.qpos[2] = base_z
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    for side, sign, phase in (("left", 1.0, 0.0), ("right", -1.0, math.pi)):
        joints = SIDE_JOINTS[side]
        swing = math.sin(gait + phase)
        lift = max(0.0, swing)
        set_joint(model, data, joints["hip_roll"], sign * (0.035 + 0.035 * math.sin(gait + phase)))
        set_joint(model, data, joints["hip_yaw"], 0.045 * math.sin(gait + phase + 0.6))
        set_joint(model, data, joints["hip_pitch"], 0.20 * swing - 0.10)
        set_joint(model, data, joints["tarsus"], -0.34 * lift + 0.08 * math.sin(gait + phase))
        set_joint(model, data, joints["toe_pitch"], 0.18 * lift)

        # Arms fold inward to hold the box while the legs continue walking.
        set_joint(model, data, joints["shoulder_pitch"], 0.45 + 0.20 * pickup - 0.12 * sign * swing * (1.0 - pickup))
        set_joint(model, data, joints["shoulder_roll"], sign * (0.62 + 0.16 * pickup))
        set_joint(model, data, joints["shoulder_yaw"], -sign * (0.28 + 0.20 * pickup))
        set_joint(model, data, joints["elbow"], -0.85 - 0.34 * pickup)
        set_joint(model, data, joints["wrist"], sign * 0.18 * pickup)

    set_joint(model, data, "idx27_head_joint1", 0.10 * math.sin(0.7 * gait))
    set_joint(model, data, "idx28_head_joint2", -0.06 - 0.04 * pickup)

    held_pos = (base_x + 0.17, base_y - 0.02, base_z - 0.24)
    start_pos = (-0.42, -0.03, 0.16)
    end_pos = (0.62, -0.03, 0.16)
    if pickup < 1.0:
        blend = pickup
        box_pos = tuple(start_pos[i] * (1.0 - blend) + held_pos[i] * blend for i in range(3))
    elif place > 0.0:
        blend = place
        box_pos = tuple(held_pos[i] * (1.0 - blend) + end_pos[i] * blend for i in range(3))
    else:
        box_pos = held_pos
    set_freejoint_pose(model, data, "carry_box_freejoint", box_pos, 0.25 * math.sin(0.4 * gait))

    mujoco.mj_forward(model, data)


def apply_front_walk_pose(model: mujoco.MjModel, data: mujoco.MjData, time_s: float, duration_s: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    walk = smoothstep(0.2, duration_s - 0.5, time_s)
    gait = 2.0 * math.pi * 1.45 * time_s
    base_x = -0.38 + 0.76 * walk
    base_z = CARRY_WALK_BASE_HEIGHT + 0.022 * math.sin(gait)
    yaw = 0.025 * math.sin(0.5 * gait)

    data.qpos[0] = base_x
    data.qpos[1] = 0.0
    data.qpos[2] = base_z
    data.qpos[3:7] = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]

    for side, sign, phase in (("left", 1.0, 0.0), ("right", -1.0, math.pi)):
        joints = SIDE_JOINTS[side]
        swing = math.sin(gait + phase)
        lift = max(0.0, swing)
        set_joint(model, data, joints["hip_roll"], 0.02 * math.sin(gait + phase))
        set_joint(model, data, joints["hip_yaw"], 0.03 * math.sin(gait + phase + 0.5))
        set_joint(model, data, joints["hip_pitch"], 0.24 * swing - 0.08)
        set_joint(model, data, joints["tarsus"], -0.30 * lift + 0.06 * math.sin(gait + phase))
        set_joint(model, data, joints["toe_pitch"], 0.16 * lift)

        # Arms hang at the sides (shoulder_roll is mirrored per side) and swing
        # front/back via shoulder_pitch, opposite to the leg on the same side.
        set_joint(model, data, joints["shoulder_pitch"], -0.30 * swing)
        set_joint(model, data, joints["shoulder_roll"], sign * 1.5)
        set_joint(model, data, joints["shoulder_yaw"], 0.0)
        set_joint(model, data, joints["elbow"], -0.30)
        set_joint(model, data, joints["wrist"], 0.0)

    set_joint(model, data, "idx27_head_joint1", 0.03 * math.sin(0.7 * gait))
    set_joint(model, data, "idx28_head_joint2", 0.0)

    mujoco.mj_forward(model, data)


def apply_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    time_s: float,
    duration_s: float,
    scenario: str,
) -> None:
    if scenario == "carry_walk":
        apply_carry_walk_pose(model, data, time_s, duration_s)
    elif scenario == "front_walk":
        apply_front_walk_pose(model, data, time_s, duration_s)
    else:
        apply_showcase_pose(model, data, time_s, duration_s)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> list[float]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body in model: {body_name}")
    return data.xpos[body_id].copy().round(5).tolist()


def update_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    time_s: float,
    scenario: str,
) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = body_position(model, data, "base_link")

    if scenario == "carry_walk":
        camera.distance = 2.35
        camera.azimuth = 102.0 + 18.0 * smoothstep(1.5, 5.0, time_s)
        camera.elevation = -12.0
    elif scenario == "front_walk":
        camera.lookat[2] = 0.90
        camera.distance = 2.7
        camera.azimuth = 180.0
        camera.elevation = -6.0
    else:
        camera.distance = 2.1
        camera.azimuth = 135.0 + 8.0 * math.sin(0.5 * time_s)
        camera.elevation = -14.0


def run_demo(
    *,
    urdf_path: Path,
    video_path: Path,
    trajectory_path: Path,
    duration_s: float,
    fps: int,
    width: int,
    height: int,
    scenario: str,
) -> dict:
    missing = missing_meshes(urdf_path)
    if missing:
        raise FileNotFoundError(
            f"Futurist URDF references {len(missing)} missing mesh file(s), "
            f"for example: {', '.join(missing[:8])}"
        )

    model = build_model(urdf_path, scenario)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    trajectory: list[dict] = []
    total_frames = int(duration_s * fps)

    for frame_idx in range(total_frames):
        time_s = frame_idx / fps
        apply_pose(model, data, time_s, duration_s, scenario)
        update_camera(model, data, camera, time_s, scenario)

        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

        if frame_idx % max(1, fps // 10) == 0:
            sample = {
                "time_s": round(time_s, 3),
                "base_pos": body_position(model, data, "base_link"),
            }
            if scenario == "carry_walk":
                sample["box_pos"] = body_position(model, data, "carry_box")
            trajectory.append(sample)

    if scenario == "carry_walk":
        task = "The Futurist humanoid walks forward, picks up a box, carries it along a marked path, and places it at the drop zone."
    elif scenario == "front_walk":
        task = "The Futurist humanoid performs a clean front-facing walk cycle with no props or extra scene objects."
    else:
        task = "The Futurist humanoid URDF loads with its mesh assets and performs a deterministic showcase animation."
    summary = {
        "project": "FF Futurist MuJoCo Test Demo",
        "task": task,
        "scenario": scenario,
        "model": str(urdf_path),
        "source": str(ROOT / "assets" / "Futurist"),
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "duration_s": duration_s,
        "fps": fps,
        "success": True,
        "final_base_pos": body_position(model, data, "base_link"),
        "trajectory_samples": trajectory,
    }

    try:
        iio.imwrite(video_path, np.asarray(frames), fps=fps, codec="libx264")
    except Exception as exc:
        fallback = video_path.with_suffix(".gif")
        iio.imwrite(fallback, np.asarray(frames), fps=fps)
        summary["video"] = str(fallback)
        summary["video_fallback_reason"] = str(exc)

    trajectory_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a MuJoCo demo video using the packaged FF Futurist humanoid URDF."
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--scenario",
        choices=("showcase", "carry_walk", "front_walk"),
        default="showcase",
        help="Demo motion to render.",
    )
    parser.add_argument("--check-assets", action="store_true", help="Only validate that all URDF mesh files exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = missing_meshes(args.urdf)
    if args.check_assets:
        result = {"urdf": str(args.urdf), "missing_mesh_count": len(missing), "missing_meshes": missing}
        print(json.dumps(result, indent=2))
        return 1 if missing else 0

    summary = run_demo(
        urdf_path=args.urdf,
        video_path=args.output,
        trajectory_path=args.trajectory,
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        scenario=args.scenario,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory_samples"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
