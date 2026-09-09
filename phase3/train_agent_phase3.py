# import os
# import time
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import SAC
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.noise import NormalActionNoise
# from carla_vlm_env import CarlaVLMEnv

# class ConvergenceLogCallbackPhase3(BaseCallback):
#     """
#     Custom callback to track episodic rewards, save them to JSON,
#     and display training progress for Phase 3 Fine-Tuning.
#     """
#     def __init__(self, total_timesteps, verbose=0):
#         super(ConvergenceLogCallbackPhase3, self).__init__(verbose)
#         self.total_timesteps = total_timesteps
#         self.episode_rewards = []
#         self.current_reward = 0.0
#         self.episode_count = 0
#         self.log_file = "training_log_phase3.json"
        
#         # Initialize an empty log file
#         with open(self.log_file, "w") as f:
#             json.dump([], f)

#     def _on_step(self) -> bool:
#         # Extract reward for this step
#         reward = self.locals.get("rewards")[0]
#         self.current_reward += reward
        
#         # Check if episode has ended
#         dones = self.locals.get("dones")
#         if dones is not None and dones[0]:
#             self.episode_count += 1
            
#             # Calculate Progress
#             current_step = self.num_timesteps
#             progress_pct = (current_step / self.total_timesteps) * 100
            
#             # Print Progress Log
#             print(f"🏁 Ep {self.episode_count} Finished | Reward: {self.current_reward:.2f} | 📈 Progress: {progress_pct:.1f}% ({current_step}/{self.total_timesteps} Steps)")
            
#             # Save data
#             self.episode_rewards.append({
#                 "episode": int(self.episode_count),
#                 "reward": float(round(self.current_reward, 2))
#             })
            
#             # Persist to disk incrementally
#             with open(self.log_file, "w") as f:
#                 json.dump(self.episode_rewards, f, indent=4)
                
#             self.current_reward = 0.0
            
#         return True

# def plot_convergence(log_file="training_log_phase3.json", output_img="convergence_plot_phase3.png"):
#     """
#     Reads the Phase 3 JSON log and generates a smoothed convergence plot.
#     """
#     print(f"\n📊 Generating Phase 3 convergence plot from {log_file}...")
#     try:
#         if not os.path.exists(log_file):
#             print(f"⚠️ Log file {log_file} not found.")
#             return

#         with open(log_file, "r") as f:
#             data = json.load(f)
            
#         if not data:
#             print("⚠️ No data to plot.")
#             return

#         episodes = [d["episode"] for d in data]
#         rewards = [d["reward"] for d in data]

#         # Calculate a moving average (window=10) for a smoother trend line
#         window = min(10, len(rewards))
#         if window > 1:
#             moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
#         else:
#             moving_avg = rewards

#         plt.figure(figsize=(10, 5))
        
#         # Plot raw rewards in light blue
#         plt.plot(episodes, rewards, color='lightblue', alpha=0.6, label='Episodic Reward')
        
#         # Plot moving average in dark blue
#         if window > 1:
#             plt.plot(episodes[(window-1):], moving_avg, color='darkblue', linewidth=2, label=f'{window}-Ep Moving Average')
        
#         plt.title('Phase 3: Fine-Tuning for Confident Driving in Dynamic Traffic', fontsize=14, fontweight='bold')
#         plt.xlabel('Episode', fontsize=12)
#         plt.ylabel('Total Reward', fontsize=12)
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend()
#         plt.tight_layout()
        
#         plt.savefig(output_img)
#         print(f"🖼️ Convergence plot saved successfully as '{output_img}'!")
        
#     except Exception as e:
#         print(f"⚠️ Could not generate plot: {e}")

# def main():
#     print("🚦 Initializing CARLA-VLM Environment for PHASE 3 (DYNAMIC TRAFFIC FINE-TUNING)...")
#     env = CarlaVLMEnv()
    
#     # 🧠 Paths for models
#     base_model_path = "sac_carla_phase2_with_obstacle"
#     new_model_path = "sac_carla_phase3_dynamic"
    
#     # Check if the Phase 2 model exists
#     if not os.path.exists(f"{base_model_path}.zip"):
#         print(f"❌ ERROR: Could not find Phase 2 model '{base_model_path}.zip'!")
#         print("Please ensure it is in the same directory.")
#         return

#     print(f"🧠 Loading pre-trained Phase 2 Brain: {base_model_path}.zip")
    
#     # -------------------------------------------------------------------
#     # 🚀 TRANSFER LEARNING: Load the Phase 2 model to continue training
#     # -------------------------------------------------------------------
#     model = SAC.load(
#         base_model_path, 
#         env=env,
#         device="cpu"  # Keep CPU to leave VRAM for Moondream/Carla
#     )
    
#     # Re-introduce exploration noise so the agent dares to press the gas pedal more
#     n_actions = env.action_space.shape[-1]
#     action_noise = NormalActionNoise(
#         mean=np.zeros(n_actions), 
#         sigma=0.1 * np.ones(n_actions) # Slightly lower noise, just enough to explore speed
#     )
#     model.action_noise = action_noise
    
#     # Lower learning rate for fine-tuning
#     model.learning_rate = 1e-4

#     print("✅ Transfer Learning Setup Complete! The agent remembers how to brake.")
#     print("🔥 Starting Phase 3 Fine-Tuning: Overcoming fear and driving confidently...")

#     # 15,000 steps is usually enough to un-learn the overly cautious behavior
#     total_timesteps = 50000
    
#     logger = ConvergenceLogCallbackPhase3(total_timesteps=total_timesteps)
#     start_time = time.time()
    
#     try:
#         model.learn(
#             total_timesteps=total_timesteps, 
#             callback=logger,
#             reset_num_timesteps=False 
#         )
#     except KeyboardInterrupt:
#         print("\n🛑 Training interrupted by user. Saving current progress...")
#     finally:
#         end_time = time.time()
#         elapsed_minutes = (end_time - start_time) / 60
#         print(f"\n⏱️ Phase 3 Session ended. Duration: {elapsed_minutes:.1f} minutes.")
        
#         # Save the new Phase 3 model
#         model.save(new_model_path)
#         print(f"💾 Phase 3 Model saved successfully as '{new_model_path}.zip'")
        
#         env.close()
        
#         # Generate the Convergence Plot
#         plot_convergence(log_file=logger.log_file)
        
#         print("🎉 Phase 3 Fine-Tuning Process Finished! The agent should now be braver in traffic.")

# if __name__ == "__main__":
#     main()

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from carla_vlm_env import CarlaVLMEnv

class ConvergenceLogCallbackPhase3(BaseCallback):
    """
    Custom callback to track episodic rewards, save them to JSON,
    and display training progress for Phase 3 Fine-Tuning.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ConvergenceLogCallbackPhase3, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_count = 0
        self.log_file = "training_log_phase3.json"
        
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

def plot_convergence(log_file="training_log_phase3.json", output_img="convergence_plot_phase3.png"):
    """
    Reads the Phase 3 JSON log and generates a smoothed convergence plot.
    """
    print(f"\n📊 Generating Phase 3 convergence plot from {log_file}...")
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
        
        # Plot raw rewards in light blue
        plt.plot(episodes, rewards, color='lightblue', alpha=0.6, label='Episodic Reward')
        
        # Plot moving average in dark blue
        if window > 1:
            plt.plot(episodes[(window-1):], moving_avg, color='darkblue', linewidth=2, label=f'{window}-Ep Moving Average')
        
        plt.title('Phase 3: Fine-Tuning for Confident Driving in Dynamic Traffic', fontsize=14, fontweight='bold')
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
    print("🚦 Initializing CARLA-VLM Environment for PHASE 3 (DYNAMIC TRAFFIC FINE-TUNING)...")
    env = CarlaVLMEnv()
    
    # 🧠 Paths for models
    base_model_path = "sac_carla_phase2_with_obstacle"
    new_model_path = "sac_carla_phase3_dynamic"
    
    # Check if the Phase 2 model exists
    if not os.path.exists(f"{base_model_path}.zip"):
        print(f"❌ ERROR: Could not find Phase 2 model '{base_model_path}.zip'!")
        print("Please ensure it is in the same directory.")
        return

    print(f"🧠 Loading pre-trained Phase 2 Brain: {base_model_path}.zip")
    
    # -------------------------------------------------------------------
    # 🚀 TRANSFER LEARNING: Load the Phase 2 model to continue training
    # -------------------------------------------------------------------
    model = SAC.load(
        base_model_path, 
        env=env,
        device="cpu",  # Keep CPU to leave VRAM for Moondream/Carla
        custom_objects={
            "learning_rate": 1e-4,     # Maintain a solid learning rate for un-freezing
            "ent_coef": "auto_0.1"     # 🔥 FORCE EXPLORATION to stop the agent from freezing
        }
    )
    
    # Re-introduce exploration noise so the agent dares to press the gas pedal more
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=0.15 * np.ones(n_actions) # Slightly increased noise to force gas/brake variety
    )
    model.action_noise = action_noise

    print("✅ Transfer Learning Setup Complete! The agent remembers how to brake.")
    print("🔥 Starting Phase 3 Fine-Tuning: Overcoming fear and driving confidently...")

    # 50,000 steps is a great starting point for dynamic traffic without overfitting
    total_timesteps = 50000
    
    logger = ConvergenceLogCallbackPhase3(total_timesteps=total_timesteps)
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=logger,
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving current progress...")
    finally:
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"\n⏱️ Phase 3 Session ended. Duration: {elapsed_minutes:.1f} minutes.")
        
        # Save the new Phase 3 model
        model.save(new_model_path)
        print(f"💾 Phase 3 Model saved successfully as '{new_model_path}.zip'")
        
        env.close()
        
        # Generate the Convergence Plot
        plot_convergence(log_file=logger.log_file)
        
        print("🎉 Phase 3 Fine-Tuning Process Finished! The agent should now be braver in traffic.")

if __name__ == "__main__":
    main()