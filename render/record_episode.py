# render/record_episode.py
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.robosuite_env   import RobosuitePhased, ROBOSUITE_PHASES, PHASE_SCHEDULE
from core.policy          import load_policy
from core.meta_controller import MetaController, build_meta_state
from training.train       import load_config


# ── Frame grabber ─────────────────────────────────────────────────────────────

def grab_frame(env, width=640, height=480):
    frame = env._env.sim.render(
        camera_name="agentview",
        width=width,
        height=height,
        depth=False,
    )
    return frame[::-1]


# ── Overlay helpers ───────────────────────────────────────────────────────────

PHASE_BGR = {
    0: (255, 200, 180),   # Normal    — blue tint
    1: (180, 240, 255),   # Heavy     — yellow tint
    2: (180, 255, 200),   # Disturbed — green tint
}

ARCH_BGR = {
    0: (235,  99,  37),   # MLP  — blue
    1: ( 37, 141, 234),   # LSTM — orange
}


def draw_overlay(frame, t, phase_id, phase_name,
                 arch_id, arch_name, reward, total_reward,
                 cube_height, switch_count):
    """
    Draws HUD overlay on frame with phase, arch, reward info.
    """
    img = frame.copy()
    h, w = img.shape[:2]

    # Semi-transparent top bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # Semi-transparent bottom bar
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, h-36), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, img, 0.4, 0, img)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.45
    font_med   = 0.55
    white      = (255, 255, 255)
    gray       = (180, 180, 180)

    # Top bar — step and phase
    cv2.putText(img, f"Step: {t:4d}",
                (10, 18), font, font_small, white, 1, cv2.LINE_AA)
    cv2.putText(img, f"Phase: {phase_name}",
                (10, 38), font, font_small,
                PHASE_BGR[phase_id], 1, cv2.LINE_AA)

    # Top bar — arch
    cv2.putText(img, f"Arch: {arch_name}",
                (w//2 - 50, 18), font, font_small,
                ARCH_BGR[arch_id], 1, cv2.LINE_AA)
    cv2.putText(img, f"Switches: {switch_count}",
                (w//2 - 50, 38), font, font_small, gray, 1, cv2.LINE_AA)

    # Top bar — cube height
    cv2.putText(img, f"Cube: {cube_height:.4f}m",
                (w - 160, 18), font, font_small, white, 1, cv2.LINE_AA)

    # Bottom bar — reward
    cv2.putText(img, f"Reward: {reward:+.3f}",
                (10, h - 12), font, font_small, white, 1, cv2.LINE_AA)
    cv2.putText(img, f"Total: {total_reward:+.3f}",
                (w//2 - 60, h - 12), font, font_med,
                (100, 255, 100), 1, cv2.LINE_AA)

    # Phase progress bar
    phase_frac = (t % 100) / 100
    bar_w      = int(w * phase_frac)
    cv2.rectangle(img, (0, 50), (bar_w, 54),
                  PHASE_BGR[phase_id], -1)

    return img


# ── Record fixed agent ────────────────────────────────────────────────────────

def record_fixed(arch="mlp", seed=42, deterministic=True,
                 width=640, height=480, fps=20):
    cfg     = load_config()
    env_cfg = cfg["environment"]
    pol_cfg = cfg["inner_policies"]
    tr_cfg  = cfg["training"]

    os.makedirs(tr_cfg["videos_dir"], exist_ok=True)
    out_path = os.path.join(tr_cfg["videos_dir"], f"fixed_{arch}_seed{seed}.mp4")

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        render_mode="rgb_array",
        seed=seed,
    )

    model = load_policy(
        arch,
        os.path.join(tr_cfg["models_dir"], f"{arch}_pretrained"),
        env, pol_cfg["device"],
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    obs, _       = env.reset(seed=seed)
    total_reward = 0.0
    done         = False

    print(f"[record] Fixed {arch.upper()} | seed={seed} → {out_path}")

    while not done:
        action, _  = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done          = terminated or truncated

        frame = grab_frame(env, width, height)
        frame = draw_overlay(
            frame,
            t=env.t,
            phase_id=info["phase_id"],
            phase_name=info["phase_name"],
            arch_id=0 if arch == "mlp" else 1,
            arch_name=arch.upper(),
            reward=reward,
            total_reward=total_reward,
            cube_height=env.history[-1]["cube_height"],
            switch_count=0,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"\rStep {env.t:4d} | Total: {total_reward:+.3f}",
              end="", flush=True)

    writer.release()
    env.close()
    print(f"\n[record] Saved → {out_path}")
    return out_path


# ── Record self-modifying agent ───────────────────────────────────────────────

def record_self_modifying(seed=0, deterministic=True,
                           width=640, height=480, fps=20):
    cfg      = load_config()
    env_cfg  = cfg["environment"]
    pol_cfg  = cfg["inner_policies"]
    meta_cfg = cfg["meta_controller"]
    tr_cfg   = cfg["training"]

    os.makedirs(tr_cfg["videos_dir"], exist_ok=True)
    out_path = os.path.join(tr_cfg["videos_dir"],
                            f"self_modifying_seed{seed}.mp4")

    env = RobosuitePhased(
        episode_length=env_cfg["episode_length"],
        phase_duration=env_cfg["phase_duration"],
        control_freq=env_cfg["control_freq"],
        render_mode="rgb_array",
        seed=seed,
    )

    policies = {
        "mlp":  load_policy("mlp",
                    os.path.join(tr_cfg["models_dir"], "mlp_pretrained"),
                    env, pol_cfg["device"]),
        "lstm": load_policy("lstm",
                    os.path.join(tr_cfg["models_dir"], "lstm_pretrained"),
                    env, pol_cfg["device"]),
    }

    meta = MetaController(
        n_archs=meta_cfg["n_archs"],
        meta_state_dim=meta_cfg["meta_state_dim"],
        hidden=meta_cfg["hidden"],
        lr=meta_cfg["lr"],
        gamma=meta_cfg["gamma"],
        switching_cost=meta_cfg["switching_cost"],
        decision_interval=meta_cfg["decision_interval"],
        history_window=meta_cfg["history_window"],
        device=meta_cfg["device"],
    )

    meta_path = os.path.join(tr_cfg["models_dir"], f"meta_seed{seed}_final.pt")
    if os.path.exists(meta_path):
        meta.load(meta_path)
    else:
        print(f"[record] No trained meta-controller at {meta_path}, using untrained.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    obs, _       = env.reset(seed=seed)
    meta.reset_episode()
    total_reward = 0.0
    current_arch = 0
    switch_count = 0
    done         = False

    print(f"[record] Self-Modifying | seed={seed} → {out_path}")

    while not done:
        t = env.t

        if t % meta_cfg["decision_interval"] == 0:
            meta_state = build_meta_state(
                meta.reward_history,
                current_arch=current_arch,
                t=t,
                episode_length=env_cfg["episode_length"],
            )
            new_arch = meta.select_arch(meta_state, deterministic=deterministic)

            if new_arch != current_arch:
                switch_count += 1
                if new_arch == 1:
                    policies["lstm"].policy.features_extractor.reset_hidden()

            current_arch      = new_arch
            meta.current_arch = current_arch

        arch_key   = "mlp" if current_arch == 0 else "lstm"
        action, _  = policies[arch_key].predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        meta.update_reward_history(float(reward))
        total_reward += reward
        done          = terminated or truncated

        frame = grab_frame(env, width, height)
        frame = draw_overlay(
            frame,
            t=env.t,
            phase_id=info["phase_id"],
            phase_name=info["phase_name"],
            arch_id=current_arch,
            arch_name="MLP" if current_arch == 0 else "LSTM",
            reward=reward,
            total_reward=total_reward,
            cube_height=env.history[-1]["cube_height"],
            switch_count=switch_count,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(
            f"\rStep {t:4d} | "
            f"Arch: {'MLP ' if current_arch == 0 else 'LSTM'} | "
            f"Switches: {switch_count} | "
            f"Total: {total_reward:+.3f}",
            end="", flush=True,
        )

    writer.release()
    env.close()
    print(f"\n[record] Saved → {out_path}")
    return out_path


# ── Record all agents for comparison ─────────────────────────────────────────

def record_all(seed=0):
    print("\n[record] Recording all agents...")
    record_fixed(arch="mlp",  seed=seed)
    record_fixed(arch="lstm", seed=seed)
    record_self_modifying(seed=seed)
    print("\n[record] All videos saved to results/videos/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record agent episodes to MP4.")
    parser.add_argument("--agent", type=str, default="self_modifying",
                        choices=["mlp", "lstm", "self_modifying", "all"],
                        help="Which agent to record")
    parser.add_argument("--seed",  type=int, default=0,
                        help="Seed for environment")
    parser.add_argument("--fps",   type=int, default=20,
                        help="Video FPS")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.agent == "self_modifying":
        record_self_modifying(seed=args.seed, fps=args.fps,
                              width=args.width, height=args.height)
    elif args.agent == "all":
        record_all(seed=args.seed)
    else:
        record_fixed(arch=args.agent, seed=args.seed, fps=args.fps,
                     width=args.width, height=args.height)