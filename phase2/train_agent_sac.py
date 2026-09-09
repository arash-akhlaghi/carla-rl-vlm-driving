import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from carla_vlm_env import CarlaVLMEnv

class ConvergenceLogCallbackPhase2(BaseCallback):
    """
    Custom callback to track episodic rewards, save them to JSON,
    and display training progress for Phase 2.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ConvergenceLogCallbackPhase2, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_count = 0
        self.log_file = "training_log_phase2.json"
        
        # Initialize an empty log file
        with open(self.log_file, "w") as f:
            json.dump([], f)

    def _on_step(self) -> bool:
        # Extract reward for this step
        reward = self.locals.get("rewards")[0]
        self.current_reward += reward
        
        # Check if episode has ended
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            self.episode_count += 1
            
            # Calculate Progress
            current_step = self.num_timesteps
            progress_pct = (current_step / self.total_timesteps) * 100
            
            # Print Progress Log
            print(f"🏁 Ep {self.episode_count} Finished | Reward: {self.current_reward:.2f} | 📈 Progress: {progress_pct:.1f}% ({current_step}/{self.total_timesteps} Steps)")
            
            # Save data
            self.episode_rewards.append({
                "episode": int(self.episode_count),
                "reward": float(round(self.current_reward, 2))
            })
            
            # Persist to disk incrementally
            with open(self.log_file, "w") as f:
                json.dump(self.episode_rewards, f, indent=4)
                
            self.current_reward = 0.0
            
        return True

def plot_convergence(log_file="training_log_phase2.json", output_img="convergence_plot_phase2.png"):
    """
    Reads the Phase 2 JSON log and generates a smoothed convergence plot.
    """
    print(f"\n📊 Generating Phase 2 convergence plot from {log_file}...")
    try:
        if not os.path.exists(log_file):
            print(f"⚠️ Log file {log_file} not found.")
            return

        with open(log_file, "r") as f:
            data = json.load(f)
            
        if not data:
            print("⚠️ No data to plot.")
            return

        episodes = [d["episode"] for d in data]
        rewards = [d["reward"] for d in data]

        # Calculate a moving average (window=10) for a smoother trend line
        window = min(10, len(rewards))
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        else:
            moving_avg = rewards

        plt.figure(figsize=(10, 5))
        
        # Plot raw rewards in light coral
        plt.plot(episodes, rewards, color='lightcoral', alpha=0.6, label='Episodic Reward')
        
        # Plot moving average in dark red
        if window > 1:
            plt.plot(episodes[(window-1):], moving_avg, color='darkred', linewidth=2, label=f'{window}-Ep Moving Average')
        
        plt.title('Phase 2: SAC VLM-Guided Braking Convergence', fontsize=14, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_img)
        print(f"🖼️ Convergence plot saved successfully as '{output_img}'!")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


def main():
    print("🚦 Initializing CARLA-VLM Environment for PHASE 2 (TRAFFIC & BRAKING)...")
    env = CarlaVLMEnv()
    
    # 🧠 Paths for models
    base_model_path = "sac_carla_phase1_no_obstacle"
    new_model_path = "sac_carla_phase2_with_obstacle"
    
    # Check if the Phase 1 model exists
    if not os.path.exists(f"{base_model_path}.zip"):
        print(f"❌ ERROR: Could not find Phase 1 model '{base_model_path}.zip'!")
        print("Please ensure it is in the same directory.")
        return

    print(f"🧠 Loading pre-trained Phase 1 Brain: {base_model_path}.zip")
    
    # -------------------------------------------------------------------
    # 🚀 TRANSFER LEARNING: Load the model to continue training
    # -------------------------------------------------------------------
    # We load the existing weights but specify the new environment and device.
    # Note: Using CPU here if you want to save VRAM for VLM, change to "cuda" if possible.
    model = SAC.load(
        base_model_path, 
        env=env,
        device="cpu"  # Keep CPU to leave VRAM for Moondream/Carla
    )
    
    # CRITICAL: Introduce exploration noise so the agent discovers the brake pedal
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.15 * np.ones(n_actions) # Light noise to encourage trying new actions
    )
    model.action_noise = action_noise
    
    # Reset learning rate for the new specific task (fine-tuning)
    model.learning_rate = 3e-4

    print("✅ Transfer Learning Setup Complete! The agent remembers how to steer.")
    print("🔥 Starting Phase 2 Training: Learning to brake for obstacles...")

    total_timesteps = 30000 
    
    # Initialize the customized callback logger
    logger = ConvergenceLogCallbackPhase2(total_timesteps=total_timesteps)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=logger,
            reset_num_timesteps=False # IMPORTANT: Appends training to the existing model history
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving current progress...")
    finally:
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"\n⏱️ Phase 2 Session ended. Duration: {elapsed_minutes:.1f} minutes.")
        
        # Save the new Phase 2 model
        model.save(new_model_path)
        print(f"💾 Phase 2 Model saved successfully as '{new_model_path}.zip'")
        
        env.close()
        
        # Generate the Convergence Plot
        plot_convergence(log_file=logger.log_file)
        
        print("🎉 Phase 2 Training Process Finished! Please check the generated plot.")

if __name__ == "__main__":
    main()