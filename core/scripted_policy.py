import os
import numpy as np


class ScriptedLiftPolicy:
    """
    Stage 0 — force gripper open until confirmed open
    Stage 1 — move above cube with gripper open
    Stage 2 — descend to cube half height with gripper open
    Stage 3 — close gripper firmly
    Stage 4 — lift up
    Stage 5 — hold
    """

    def __init__(self, lift_height=0.25, speed=0.05):
        self.lift_height = lift_height
        self.speed       = speed
        self.stage       = 0
        self.stage_timer = 0

    def reset(self):
        self.stage       = 0
        self.stage_timer = 0

    def act(self, obs_dict: dict) -> np.ndarray:
        cube_pos    = np.array(obs_dict["cube_pos"])
        eef_pos     = np.array(obs_dict["robot0_eef_pos"])
        gripper_val = float(obs_dict["robot0_gripper_qpos"][0])
        action      = np.zeros(7)

        if self.stage == 0:
            action[:3] = np.zeros(3)
            action[6]  = -1.0
            print(f"\r[S0] Forcing gripper open | gripper:{gripper_val:.5f} | need > 0.018", end="", flush=True)
            if gripper_val > 0.018:
                self.stage       = 1
                self.stage_timer = 0
                print(f"\n>>> GRIPPER CONFIRMED OPEN: {gripper_val:.5f}")
                print(f"--- STAGE 1 START --- moving above cube")
                print(f"    EEF:  {eef_pos}")
                print(f"    Cube: {cube_pos}")
                print(f"    Target: {cube_pos + np.array([0,0,0.15])}")

        elif self.stage == 1:
            target     = cube_pos + np.array([0, 0, 0.15])
            delta      = target - eef_pos
            dist       = np.linalg.norm(delta)
            action[:3] = np.clip(delta * 5.0, -self.speed*2, self.speed*2)
            action[6]  = -1.0
            print(f"\r[S1] Above cube | EEF z:{eef_pos[2]:.4f} target z:{target[2]:.4f} dist:{dist:.4f} grip:{gripper_val:.4f}", end="", flush=True)
            if dist < 0.02 and gripper_val > 0.018:
                self.stage       = 2
                self.stage_timer = 0
                print(f"\n--- STAGE 2 START --- descending to cube half height")
                print(f"    EEF:      {eef_pos}")
                print(f"    Cube:     {cube_pos}")
                print(f"    Target z: {cube_pos[2] - 0.0125:.4f}")
                print(f"    Gripper:  {gripper_val:.5f} (confirmed open)")
            elif dist < 0.02:
                action[:3] = np.zeros(3)
                print(f"\r[S1] In position — waiting for gripper open | grip:{gripper_val:.5f} need > 0.018", end="", flush=True)

        elif self.stage == 2:
            target     = cube_pos + np.array([0, 0, -0.0325])
            delta      = target - eef_pos
            dist       = np.linalg.norm(delta)
            action[:3] = np.clip(delta * 3.0, -self.speed*0.5, self.speed*0.5)
            action[6]  = -1.0
            print(f"\r[S2] Descending | EEF z:{eef_pos[2]:.4f} target z:{target[2]:.4f} dist:{dist:.4f} gap:{eef_pos[2]-cube_pos[2]:.4f} grip:{gripper_val:.4f}", end="", flush=True)
            if dist < 0.025 or eef_pos[2] < cube_pos[2] + 0.01:
                self.stage       = 3
                self.stage_timer = 0
                print(f"\n--- STAGE 3 START --- closing gripper")
                print(f"    EEF:     {eef_pos}")
                print(f"    Cube:    {cube_pos}")
                print(f"    Gap z:   {eef_pos[2]-cube_pos[2]:.4f}")
                print(f"    Gripper: {gripper_val:.5f}")
                print(f"    XY dist: {np.linalg.norm(eef_pos[:2]-cube_pos[:2]):.4f}")

        elif self.stage == 3:
            action[:3]        = np.zeros(3)
            action[6]         = 1.0
            self.stage_timer += 1
            print(f"\r[S3] Closing gripper | timer:{self.stage_timer}/35 grip:{gripper_val:.4f}", end="", flush=True)
            if self.stage_timer > 35:
                self.stage = 4
                print(f"\n--- STAGE 4 START --- LIFTING")
                print(f"    EEF:     {eef_pos}")
                print(f"    Cube:    {cube_pos}")
                print(f"    Gap z:   {eef_pos[2]-cube_pos[2]:.4f}")
                print(f"    Gripper: {gripper_val:.5f} (0.039=closed)")

        elif self.stage == 4:
            target     = np.array([eef_pos[0], eef_pos[1],
                                   eef_pos[2] + self.lift_height])
            delta      = target - eef_pos
            action[:3] = np.clip(delta * 5.0, -self.speed*2, self.speed*2)
            action[6]  = 1.0
            print(f"\r[S4] Lifting | EEF z:{eef_pos[2]:.4f} Cube z:{cube_pos[2]:.4f} grip:{gripper_val:.4f}", end="", flush=True)
            if eef_pos[2] > cube_pos[2] + self.lift_height - 0.03:
                self.stage = 5
                print(f"\n--- STAGE 5 START --- HOLDING")
                print(f"    EEF z:     {eef_pos[2]:.4f}")
                print(f"    Cube z:    {cube_pos[2]:.4f}")
                print(f"    Cube moved:{cube_pos[2]:.4f}")
                print(f"    Gripper:   {gripper_val:.5f}")

        elif self.stage == 5:
            action[6] = 1.0
            print(f"\r[S5] Holding | EEF z:{eef_pos[2]:.4f} Cube z:{cube_pos[2]:.4f} grip:{gripper_val:.4f}", end="", flush=True)

        return action.astype(np.float32)


# ── Record function ───────────────────────────────────────────────────────────

def record_scripted(seed=42, width=640, height=480, fps=20,
                    out_path="results/videos/scripted_policy.mp4"):
    import cv2
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = suite.make(
        "Lift", robots="Panda",
        controller_configs=cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    obs = env.reset()

    # pre-open gripper
    for _ in range(100):
        action    = np.zeros(7)
        action[6] = -1.0
        obs, _, _, _ = env.step(action)

    policy = ScriptedLiftPolicy()
    policy.reset()
    policy.stage = 1

    total = 0.0
    lifts = 0
    print(f"[record] Scripted policy → {out_path}")

    for t in range(1000):
        action               = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        total               += reward
        cube_z               = obs["cube_pos"][2]
        lifted               = cube_z > 0.86

        frame = env.sim.render(
            camera_name="agentview",
            width=width, height=height, depth=False,
        )[::-1]

        img = frame.copy()

        # stage color
        stage_colors = {
            1: (255,255,100), 2: (100,255,255),
            3: (100,100,255), 4: (100,255,100), 5: (255,100,100)
        }
        sc = stage_colors.get(policy.stage, (255,255,255))

        stage_names = {
            1: "Moving Above", 2: "Descending",
            3: "Closing Grip", 4: "Lifting", 5: "Holding"
        }
        sn = stage_names.get(policy.stage, "")

        # semi-transparent top bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (width, 55), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # semi-transparent bottom bar
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, height-35), (width, height), (20,20,20), -1)
        cv2.addWeighted(overlay2, 0.6, img, 0.4, 0, img)

        cv2.putText(img, f"Step: {t:4d}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img, f"Stage {policy.stage}: {sn}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)
        cv2.putText(img, f"Cube: {cube_z:.4f}m",
                    (width//2 - 60, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(img,
                    "LIFTED!" if lifted else "not lifted",
                    (width//2 - 60, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0) if lifted else (150,150,150), 1)
        cv2.putText(img, f"Reward: {total:+.1f}",
                    (10, height-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)

        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        lifts += int(lifted)

        print(f"\rStep {t:4d} | Stage {policy.stage} | "
              f"Cube: {cube_z:.4f}m | Lifted: {lifted} | "
              f"Total: {total:+.3f}", end="", flush=True)

        if done:
            break

    writer.release()
    env.close()
    print(f"\n[record] Done. Lift steps: {lifts} | Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = suite.make(
        "Lift", robots="Panda",
        controller_configs=cfg,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )

    obs         = env.reset()
    cube_z_init = obs["cube_pos"][2]

    print(f"=== INITIAL STATE ===")
    print(f"Cube:    {obs['cube_pos']}")
    print(f"EEF:     {obs['robot0_eef_pos']}")
    print(f"Gripper: {obs['robot0_gripper_qpos']}")
    print(f"====================")

    print("Pre-opening gripper for 100 steps...")
    for i in range(100):
        action       = np.zeros(7)
        action[6]    = -1.0
        obs, _, _, _ = env.step(action)
        env.render()
        print(f"\rPre-open step {i:3d} | gripper: {obs['robot0_gripper_qpos'][0]:.5f}", end="", flush=True)

    print(f"\nGripper after pre-open: {obs['robot0_gripper_qpos']}")
    print(f"Starting policy...")

    policy = ScriptedLiftPolicy()
    policy.reset()
    policy.stage = 1
    print(f"\n--- STAGE 1 START --- gripper pre-opened, moving above cube")

    total = 0.0
    lifts = 0

    for t in range(1000):
        action               = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        total               += reward
        cube_z               = obs["cube_pos"][2]
        lifted               = cube_z > cube_z_init + 0.04
        lifts               += int(lifted)
        env.render()

        if done:
            break

    print(f"\n\n=== FINAL ===")
    print(f"Reward:     {total:.3f}")
    print(f"Lift steps: {lifts}")
    print(f"Cube start: {cube_z_init:.4f}")
    print(f"Cube end:   {obs['cube_pos'][2]:.4f}")
    print(f"Cube moved: {obs['cube_pos'][2]-cube_z_init:.4f}m")
    env.close()