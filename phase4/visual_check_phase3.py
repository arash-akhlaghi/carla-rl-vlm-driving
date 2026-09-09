import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    """
    Pre-training Visual Check Script for Phase 3.
    Loads the baseline Phase 2 model to observe its initial behavior
    in the new dynamic traffic environment before any fine-tuning begins.
    """
    print("🚗 Initializing CARLA-VLM Environment for PRE-TRAINING VISUAL CHECK...")
    env = CarlaVLMEnv()
    
    # 🧠 Load the Phase 2 model (which knows how to brake, but might be too cautious)
    model_path = "sac_carla_phase2_with_obstacle"
    print(f"🧠 Loading Phase 2 Model from {model_path}.zip...")
    
    try:
        # Load the model on CPU to leave VRAM for the VLM and CARLA simulator
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Phase 2 Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the .zip file is in the same directory.")
        return

    print("🔥 Starting Visual Check (Press 'q' in the video window to stop)...")
    print("⚠️ EXPECTATION: The agent will brake safely for moving cars, but might drive too slowly on clear roads.")
    
    # Run the test for 3 episodes to observe behavior across multiple spawns
    test_episodes = 3
    
    for ep in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # 🟢 CRITICAL: deterministic=True removes exploration noise.
            # We want to see the pure learned behavior of the Phase 2 model.
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute step in the environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # 📺 Visual Dashboard Update
            if "image" in info and info["image"]:
                # Decode base64 image from the environment
                img_data = base64.b64decode(info["image"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # Extract telemetry from the observation array
                speed = obs[0]
                cte = obs[1]
                vlm_signal = obs[5]
                steer_cmd = action[1]
                throttle_cmd = action[0]
                
                # Overlay Phase and Episode text
                cv2.putText(frame, f"PHASE 3 PRE-CHECK - Ep: {ep + 1}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Overlay Speed and Commands
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Throttle/Brake: {throttle_cmd:.2f}", 
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Overlay VLM Status (Green for SAFE, Red for DANGER)
                vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
                vlm_status = "SAFE (Path Clear)" if vlm_signal > 0.5 else "DANGER (Obstacle!)"
                cv2.putText(frame, f"VLM: {vlm_status}", 
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vlm_color, 2)
                
                # Overlay Rewards
                cv2.putText(frame, f"Step Reward: {reward:.1f}", 
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Total Reward: {episode_reward:.1f}", 
                            (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Flash warning if agent is being overly cautious (driving slow on clear road)
                if vlm_signal > 0.5 and speed < 15.0 and step_count > 20:
                    cv2.putText(frame, "⚠️ TOO CAUTIOUS! NEED FINE-TUNING", 
                                (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)
                
                # Flash warning if agent is stopped for no reason
                if vlm_signal > 0.5 and speed < 1.0 and step_count > 20:
                    cv2.putText(frame, "🛑 UNNECESSARY STOP!", 
                                (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                
                # Show the combined visual output
                cv2.imshow("CARLA Phase 3 - Pre-Train Visualizer", frame)
                
                # Graceful exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Manual stop requested by user.")
                    env.close()
                    cv2.destroyAllWindows()
                    return
        
        print(f"🏁 Episode {ep + 1} Ended | Total Reward: {episode_reward:.2f} | Steps: {step_count}")

    # Cleanup resources
    env.close()
    cv2.destroyAllWindows()
    print("✅ Pre-training visual check completed!")

if __name__ == "__main__":
    main()