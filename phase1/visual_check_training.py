import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from carla_vlm_env import CarlaVLMEnv

class VisualDebugCallback(BaseCallback):
    """
    A custom callback designed for Sanity Checks.
    It renders the live camera feed and telemetry during the training process.
    It automatically stops the training after a specified number of episodes.
    """
    def __init__(self, max_episodes=5, verbose=0):
        super(VisualDebugCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0
        self.episode_reward = 0.0

    def _on_step(self) -> bool:
        # Extract observations and info from the vectorized environment
        obs = self.locals.get("new_obs")[0]
        infos = self.locals.get("infos")[0]
        reward = self.locals.get("rewards")[0]
        
        self.episode_reward += reward
        
        # Parse observation vector
        speed = obs[0]
        cte = obs[1]
        vlm_signal = obs[5]
        
        # ---------------------------------------------------------
        # 📺 AI Visual Dashboard (Live Training Feed)
        # ---------------------------------------------------------
        if "image" in infos:
            img_data = base64.b64decode(infos["image"])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Overlay Training Telemetry
            cv2.putText(frame, f"TRAINING - Episode: {self.episode_count + 1}/{self.max_episodes}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"CTE (Error): {cte:.2f} m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
            cv2.putText(frame, f"VLM Signal: {vlm_signal:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vlm_color, 2)
            cv2.putText(frame, f"Current Reward: {self.episode_reward:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Render the frame
            cv2.imshow("Live Training Sanity Check", frame)
            
            # Essential for updating the cv2 window smoothly
            cv2.waitKey(1)
            
        # ---------------------------------------------------------
        # Episode Termination Logic
        # ---------------------------------------------------------
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            print(f"🏁 Episode {self.episode_count + 1} Ended. Reward: {self.episode_reward:.2f}")
            self.episode_count += 1
            self.episode_reward = 0.0
            
            if self.episode_count >= self.max_episodes:
                print(f"\n🛑 Sanity Check Complete! {self.max_episodes} episodes finished.")
                print("Closing windows and stopping training gracefully...")
                cv2.destroyAllWindows()
                return False  # Returning False completely stops the SAC.learn() process
                
        return True

def main():
    print("🚗 Initializing Carla-VLM Environment for VISUAL TRAINING CHECK...")
    env = CarlaVLMEnv()
    
    print("🧠 Creating Fresh SAC Agent...")
    model = SAC(
        policy="MlpPolicy",         
        env=env,
        verbose=0,
        learning_rate=0.0003,
        buffer_size=50000,          
        learning_starts=100,        
        batch_size=64
    )

    # Initialize the visual callback set to 5 episodes
    visual_callback = VisualDebugCallback(max_episodes=5)

    print("🔥 Starting Sanity Check Training (5 Episodes max)...")
    try:
        # A large timestep number, but it will be interrupted by the callback after 5 episodes
        model.learn(total_timesteps=100000, callback=visual_callback)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted manually.")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("✅ Visual check finished. If everything looked good, run 'train_agent_sac.py' for full training!")

if __name__ == "__main__":
    main()