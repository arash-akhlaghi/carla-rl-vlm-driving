import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from carla_vlm_env import CarlaVLMEnv

class ConvergenceLogCallback(BaseCallback):
    """
    Custom callback to track episodic rewards, save them to JSON,
    and display training progress (Percentage & Steps).
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ConvergenceLogCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_count = 0
        self.log_file = "training_log_phase1.json"
        
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
            
            # 🟢 Calculate Progress
            current_step = self.num_timesteps
            progress_pct = (current_step / self.total_timesteps) * 100
            
            # Print Beautiful Progress Log
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

def plot_convergence(log_file="training_log_phase1.json", output_img="convergence_plot_phase1.png"):
    """
    Reads the JSON log and generates a smoothed convergence plot.
    """
    print(f"\n📊 Generating convergence plot from {log_file}...")
    try:
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
        
        # Plot raw rewards in light blue
        plt.plot(episodes, rewards, color='lightblue', alpha=0.6, label='Episodic Reward')
        
        # Plot moving average in dark blue
        if window > 1:
            plt.plot(episodes[(window-1):], moving_avg, color='navy', linewidth=2, label=f'{window}-Ep Moving Average')
        
        plt.title('Phase 1: SAC Path Tracking Convergence (No Obstacles)', fontsize=14, fontweight='bold')
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
    print("🚗 Initializing Phase 1 CARLA-VLM Environment (Path Tracking Only)...")
    env = CarlaVLMEnv()
    
    print("🧠 Creating SAC Agent (Forcing CPU to prevent CUDA OOM)...")
    model = SAC(
        policy="MlpPolicy",         
        env=env,
        learning_rate=0.0003,
        buffer_size=50000,          
        learning_starts=500,  # Starts actual optimization after 500 random exploratory steps      
        batch_size=64,
        ent_coef="auto",      
        device="cpu",         # 🟢 CRITICAL: Keeps VRAM free for VLM/CARLA
        verbose=0
    )

    # 🟢 Set total timesteps to your chosen robust number (20,000)
    total_timesteps = 30000  

    # Initialize the callback logger and pass the total timesteps
    logger = ConvergenceLogCallback(total_timesteps=total_timesteps)
    
    print(f"🔥 Starting Phase 1 Training ({total_timesteps} steps)...")
    start_time = time.time()
    
    try:
        # Start learning
        model.learn(total_timesteps=total_timesteps, callback=logger)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted manually by user. Preparing to save progress...")
    finally:
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"\n⏱️ Training session ended. Duration: {elapsed_minutes:.1f} minutes.")
        
        # 1. Save the neural network weights
        save_path = "sac_carla_phase1_no_obstacle"
        model.save(save_path)
        print(f"💾 Agent Brain saved securely as '{save_path}.zip'!")
        
        # 2. Close environment connections
        env.close()
        
        # 3. Generate Convergence Graph
        plot_convergence(log_file=logger.log_file)
        
        print("\n✅ Phase 1 completed successfully! Ready for inference or Phase 2.")

if __name__ == "__main__":
    main()