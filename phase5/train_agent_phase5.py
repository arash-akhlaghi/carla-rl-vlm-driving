import os
import re
import time
import json
import glob
import numpy as np

# Prevent GUI backend crashes between OpenCV and Qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from carla_vlm_env import CarlaVLMEnv


# ─────────────────────────────────────────────────────────────
# 📐  Continuous Linear Learning Rate Schedule
# ─────────────────────────────────────────────────────────────
class ContinuousLinearSchedule:
    """
    Step-based continuous learning rate schedule.
    Prevents LR spikes upon resuming by dynamically decaying from the
    optimizer's existing checkpoint LR down to the final target rate.
    """
    def __init__(self, start_step: int, target_step: int, initial_lr: float, final_lr: float):
        self.start_step = int(start_step)
        self.target_step = max(self.start_step + 1, int(target_step))
        self.initial_lr = float(initial_lr)
        self.final_lr = float(final_lr)

    def __call__(self, step: int) -> float:
        if step <= self.start_step:
            return float(self.initial_lr)
        progress = min(1.0, max(0.0, float(step - self.start_step) / (self.target_step - self.start_step)))
        return float(self.initial_lr - (self.initial_lr - self.final_lr) * progress)


# ─────────────────────────────────────────────────────────────
# 💾 Official SB3 Replay Buffer Persistence
# ─────────────────────────────────────────────────────────────
def save_replay_buffer(model, path: str) -> bool:
    try:
        model.save_replay_buffer(path)
        print(f"💾 Replay buffer saved → {path}")
        return True
    except Exception as e:
        print(f"⚠️ Could not save replay buffer: {e}")
        return False


def load_replay_buffer(model, path: str) -> int:
    if not os.path.exists(path):
        print(f"ℹ️ Replay buffer not found at '{path}' — starting with an empty buffer.")
        return 0
    try:
        model.load_replay_buffer(path)
        size = model.replay_buffer.size()
        print(f"✅ Replay buffer restored: {size} transitions loaded.")
        return size
    except Exception as e:
        print(f"⚠️ Could not load replay buffer: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
# 🔍 Strict 1:1 Checkpoint & Buffer Pairing Discovery
# ─────────────────────────────────────────────────────────────
def get_strictly_paired_checkpoint(new_model_path: str):
    ckpt_files = glob.glob(f"{new_model_path}_ckpt_ep*.zip")
    if not ckpt_files:
        return None, None, 0

    def extract_ep(path):
        m = re.search(r"_ckpt_ep(\d+)\.zip$", path)
        return int(m.group(1)) if m else -1

    ckpt_files.sort(key=extract_ep, reverse=True)

    for ckpt in ckpt_files:
        ep_num = extract_ep(ckpt)
        buf_path = f"{new_model_path}_replay_buffer_ep{ep_num}.pkl"
        if os.path.exists(buf_path):
            return ckpt, buf_path, ep_num

    return None, None, 0


# ─────────────────────────────────────────────────────────────
# 📊 Training Callback & Telemetry
# ─────────────────────────────────────────────────────────────
class ConvergenceLogCallbackPhase5(BaseCallback):
    def __init__(self, target_total_timesteps: int, save_path="sac_carla_phase5_right_of_way",
                 checkpoint_every=25, verbose=0, buffer_save_path=None,
                 resume=False, resume_ep=0, lr_schedule=None):
        super(ConvergenceLogCallbackPhase5, self).__init__(verbose)
        self.target_total_timesteps = target_total_timesteps
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_count = 0
        self.best_reward = -np.inf
        self.save_path = save_path
        self.checkpoint_every = checkpoint_every
        self.buffer_save_path = buffer_save_path
        self.log_file = "training_log_phase5_clean.json"
        self.lr_schedule = lr_schedule

        self.episode_ctes = []
        self.episode_speeds = []
        self.episode_had_collision = False
        self.resume = resume

        SENSIBLE_REWARD_CEILING = 1000.0

        if self.resume and os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    raw_data = json.load(f)

                # Keep historical entries up to loaded episode
                self.episode_rewards = [d for d in raw_data if d.get("episode", 0) <= resume_ep]
                self.episode_count = resume_ep

                valid_rewards = [
                    d["reward"] for d in self.episode_rewards
                    if d["reward"] < SENSIBLE_REWARD_CEILING
                ]
                self.best_reward = max(valid_rewards) if valid_rewards else -np.inf

                print(
                    f"📄 Logs aligned to Episode {self.episode_count} | "
                    f"Best valid reward: {self.best_reward:.2f}"
                )

                with open(self.log_file, "w") as f:
                    json.dump(self.episode_rewards, f, indent=4)

            except Exception as e:
                print(f"⚠️ Could not sync log file: {e}")
                self.episode_rewards = []
        else:
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards")[0]
        self.current_reward += reward
        current_step = self.num_timesteps

        # Update PyTorch optimizer param groups directly
        if self.lr_schedule is not None:
            new_lr = self.lr_schedule(current_step)
            optimizers = [
                self.model.policy.actor.optimizer,
                self.model.policy.critic.optimizer,
            ]
            ent_opt = getattr(self.model, "ent_coef_optimizer", None)
            if ent_opt is not None:
                optimizers.append(ent_opt)

            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = new_lr

        try:
            obs = self.locals.get("new_obs")
            if obs is not None and len(obs) > 0:
                self.episode_speeds.append(float(obs[0][0]))
                self.episode_ctes.append(abs(float(obs[0][1])))
        except Exception:
            pass

        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            if reward <= -99.0:
                self.episode_had_collision = True

            self.episode_count += 1
            progress_pct = (current_step / self.target_total_timesteps) * 100

            current_lr = 1e-5
            try:
                current_lr = self.model.policy.actor.optimizer.param_groups[0]['lr']
            except Exception:
                pass

            ent_coef_value = "auto"
            try:
                ent_coef_value = float(self.model.ent_coef_tensor.exp().item())
            except Exception:
                pass

            recent_rewards = [d["reward"] for d in self.episode_rewards[-(min(10, len(self.episode_rewards))):]]
            recent_rewards.append(round(self.current_reward, 2))
            running_avg = np.mean(recent_rewards) if recent_rewards else self.current_reward

            avg_cte = float(np.mean(self.episode_ctes)) if self.episode_ctes else 0.0
            avg_speed = float(np.mean(self.episode_speeds)) if self.episode_speeds else 0.0
            had_collision = bool(self.episode_had_collision)

            print(
                f"🏁 Ep {self.episode_count} Finished | "
                f"Reward: {self.current_reward:.2f} | "
                f"Avg(10): {running_avg:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"α: {ent_coef_value if isinstance(ent_coef_value, str) else f'{ent_coef_value:.4f}'} | "
                f"CTE: {avg_cte:.2f} | Spd: {avg_speed:.1f} km/h | "
                f"Coll: {'YES' if had_collision else 'no'} | "
                f"📈 {progress_pct:.1f}% ({current_step}/{self.target_total_timesteps})"
            )

            self.episode_rewards.append({
                "episode": int(self.episode_count),
                "reward": float(round(self.current_reward, 2)),
                "running_avg_10": float(round(running_avg, 2)),
                "learning_rate": float(current_lr),
                "entropy_coef": float(ent_coef_value) if not isinstance(ent_coef_value, str) else None,
                "timestep": int(current_step),
                "cte": avg_cte,
                "speed": avg_speed,
                "collision": had_collision,
            })

            with open(self.log_file, "w") as f:
                json.dump(self.episode_rewards, f, indent=4)

            # Persist best performing checkpoint
            if (self.current_reward > self.best_reward and self.current_reward < 1000.0):
                self.best_reward = self.current_reward
                best_path = f"{self.save_path}_best"
                try:
                    self.model.save(best_path)
                    print(f"  🏆 New best model saved as '{best_path}.zip'")
                except Exception as e:
                    print(f"  ⚠️ Best model save failed: {e}")

            # Save synchronized checkpoint pair
            if self.episode_count % self.checkpoint_every == 0:
                ckpt_path = f"{self.save_path}_ckpt_ep{self.episode_count}"
                buf_saved = True

                if self.buffer_save_path:
                    buf_path = f"{self.buffer_save_path}_ep{self.episode_count}.pkl"
                    buf_saved = save_replay_buffer(self.model, buf_path)

                if buf_saved:
                    try:
                        self.model.save(ckpt_path)
                        print(f"  💾 Model checkpoint saved: '{ckpt_path}.zip'")
                        self._prune_old_checkpoints(keep=3)
                    except Exception as e:
                        print(f"  ⚠️ Periodic checkpoint save failed: {e}")

            self.current_reward = 0.0
            self.episode_ctes = []
            self.episode_speeds = []
            self.episode_had_collision = False

        return True

    def _prune_old_checkpoints(self, keep=3):
        try:
            ckpt_files = glob.glob(f"{self.save_path}_ckpt_ep*.zip")
            def extract_ep(path):
                m = re.search(r"_ckpt_ep(\d+)\.zip$", path)
                return int(m.group(1)) if m else -1

            ckpt_files.sort(key=extract_ep)
            for old_ckpt in ckpt_files[:-keep]:
                ep = extract_ep(old_ckpt)
                old_buf = f"{self.buffer_save_path}_ep{ep}.pkl"
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                if os.path.exists(old_buf):
                    os.remove(old_buf)
                print(f"  🧹 Pruned checkpoint pair: Episode {ep}")
        except Exception as e:
            print(f"⚠️ Checkpoint pruning warning: {e}")


# ─────────────────────────────────────────────────────────────
# 📈 Plotting
# ─────────────────────────────────────────────────────────────
def plot_convergence(log_file="training_log_phase5_clean.json", output_img="convergence_plot_phase5_clean.png"):
    try:
        if not os.path.exists(log_file):
            return
        with open(log_file, "r") as f:
            data = json.load(f)
        if not data:
            return

        episodes = [d["episode"] for d in data]
        rewards = [d["reward"] for d in data]
        window = min(10, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid') if window > 1 else rewards

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 1, 1]})

        ax1.plot(episodes, rewards, color='lightblue', alpha=0.6, label='Episodic Reward')
        if window > 1:
            ax1.plot(episodes[(window - 1):], moving_avg, color='darkblue', linewidth=2, label=f'{window}-Ep Moving Average')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_title('Phase 5: Right of Way & Intersection Yielding', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Total Reward', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        lrs = [d.get("learning_rate", 0) for d in data]
        ax2.plot(episodes, lrs, color='tab:orange', linewidth=1.5, label='Learning Rate')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=10, color='tab:orange')
        ax2.grid(True, linestyle='--', alpha=0.5)

        ctes = [d.get("cte", 0.0) for d in data]
        speeds = [d.get("speed", 0.0) for d in data]
        collisions = [1 if d.get("collision", False) else 0 for d in data]

        ax3.plot(episodes, ctes, color='tab:red', linewidth=1.2, label='Avg |CTE| (m)')
        ax3.plot(episodes, speeds, color='tab:blue', linewidth=1.2, label='Avg Speed (km/h)')
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('CTE (m) / Speed (km/h)', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.5)

        collision_eps = [ep for ep, c in zip(episodes, collisions) if c == 1]
        collision_ctes = [d["cte"] for d in data if d.get("collision", False)]
        if collision_eps:
            ax3.scatter(collision_eps, collision_ctes, color='red', marker='x', s=40, label='Collision', zorder=5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(output_img, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


# ─────────────────────────────────────────────────────────────
# 🚀 Main Training Loop
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🚦  PHASE 5 — RIGHT OF WAY & INTERSECTION YIELDING")
    print("=" * 70)

    # Runtime configuration: Resume to 80k global timesteps
    RESUME = True
    CUMULATIVE_TARGET_STEPS = 80_000

    env = CarlaVLMEnv()
    base_model_path = "sac_carla_phase5_init"
    new_model_path = "sac_carla_phase5_final"
    buffer_save_base = f"{new_model_path}_replay_buffer"

    load_path = base_model_path
    paired_buffer_path = None
    resume_ep = 0
    resumed = False

    if RESUME:
        final_model_zip = f"{new_model_path}.zip"
        final_buf_pkl = f"{buffer_save_base}_latest.pkl"

        # Priority 1: Load final session state (Episode 146 / 49,972 steps)
        if os.path.exists(final_model_zip) and os.path.exists(final_buf_pkl):
            load_path = new_model_path
            paired_buffer_path = final_buf_pkl

            if os.path.exists("training_log_phase5_clean.json"):
                try:
                    with open("training_log_phase5_clean.json", "r") as f:
                        log_data = json.load(f)
                        resume_ep = int(log_data[-1]["episode"]) if log_data else 0
                except Exception:
                    resume_ep = 146
            else:
                resume_ep = 146

            resumed = True
            print(f"🔥 Found Completed 50k State! Loading Final Model: '{final_model_zip}' ↔ '{final_buf_pkl}' (Ep {resume_ep})")

        # Priority 2: Fallback to latest strictly paired checkpoint
        else:
            ckpt_path, paired_buf, ep_num = get_strictly_paired_checkpoint(new_model_path)
            if ckpt_path and paired_buf:
                load_path = ckpt_path.replace(".zip", "")
                paired_buffer_path = paired_buf
                resume_ep = ep_num
                resumed = True
                print(f"🔄 Checkpoint Pair Found: Model '{ckpt_path}' ↔ Buffer '{paired_buf}' (Ep {ep_num})")
            else:
                print("❌ CRITICAL ERROR: Resume is True, but no compatible checkpoint was found!")
                return

    model = SAC.load(
        load_path,
        env=env,
        device="cuda"
    )
    print(f"   ⏱️  Loaded model internal timestep: {model.num_timesteps}")

    model.tau = 0.005
    model.gradient_steps = 1
    model.action_noise = None

    # Smoke test serialization sanity
    smoke_file = "test_save_smoke.zip"
    try:
        if os.path.exists(smoke_file):
            os.remove(smoke_file)
        model.save("test_save_smoke")
        if os.path.exists(smoke_file):
            os.remove(smoke_file)
        print("✅ Serialization sanity check passed.")
    except Exception as e:
        if os.path.exists(smoke_file):
            os.remove(smoke_file)
        print(f"❌ Critical: model.save failed sanity check: {e}")
        return

    if resumed and paired_buffer_path:
        restored = load_replay_buffer(model, paired_buffer_path)
        if restored > model.batch_size:
            model.learning_starts = 0
            print("⚡ learning_starts set to 0 (Warmup bypassed with loaded buffer).")

    # Read current checkpoint LR to enforce seamless continuation without spikes
    current_loaded_lr = 1e-4
    try:
        current_loaded_lr = float(model.policy.actor.optimizer.param_groups[0]['lr'])
    except Exception:
        pass

    if resumed:
        # Decays smoothly from ~1.005e-5 to 5.00e-6 across the final 30,028 steps
        lr_schedule = ContinuousLinearSchedule(
            start_step=model.num_timesteps,
            target_step=CUMULATIVE_TARGET_STEPS,
            initial_lr=current_loaded_lr,
            final_lr=5e-6
        )
        print(f"🎯 Resumed LR Policy: Smooth decay from {current_loaded_lr:.2e} → 5.00e-06 (No Jump!)")
    else:
        lr_schedule = ContinuousLinearSchedule(
            start_step=0,
            target_step=CUMULATIVE_TARGET_STEPS,
            initial_lr=1e-4,
            final_lr=1e-5
        )

    steps_to_train = max(0, CUMULATIVE_TARGET_STEPS - model.num_timesteps)
    print(f"🎯 Target Steps: {CUMULATIVE_TARGET_STEPS} | Remaining Steps to Train: {steps_to_train}")

    if steps_to_train == 0:
        print("🎉 Target timesteps already achieved!")
        return

    logger = ConvergenceLogCallbackPhase5(
        target_total_timesteps=CUMULATIVE_TARGET_STEPS,
        save_path=new_model_path,
        checkpoint_every=25,
        buffer_save_path=buffer_save_base,
        resume=resumed,
        resume_ep=resume_ep,
        lr_schedule=lr_schedule,
    )

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=steps_to_train,
            callback=logger,
            reset_num_timesteps=not resumed,
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving progress...")
    finally:
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"\n⏱️ Session duration: {elapsed_minutes:.1f} minutes.")

        try:
            model.save(new_model_path)
            print(f"💾 Final Model saved: '{new_model_path}.zip'")
        except Exception as e:
            print(f"⚠️ Could not save final model: {e}")

        final_buf_path = f"{buffer_save_base}_latest.pkl"
        save_replay_buffer(model, final_buf_path)

        env.close()
        plot_convergence(log_file=logger.log_file)
        print("🎉 Phase 5 Session Complete!")


if __name__ == "__main__":
    main()