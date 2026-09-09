import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    """
    Test script for Phase 3: Dynamic Traffic Inference.
    We are loading the HEALTHY Phase 2 model, paired with our newly
    bulletproofed VLM Semantic Parser, to drive in dynamic traffic.
    """
    print("🚗 Initializing CARLA-VLM Environment for PHASE 3 DYNAMIC TEST...")
    env = CarlaVLMEnv()
    
    # 🧠 FIX: Load the robust Phase 2 model, ditching the collapsed Phase 3 model.
    model_path = "sac_carla_phase3_dynamic"
    print(f"🧠 Loading healthy trained brain from {model_path}.zip...")
    
    try:
        # Load the model on CPU to leave VRAM for the VLM and CARLA simulator
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Phase 2 Model loaded successfully for Phase 3 testing!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure 'sac_carla_phase2_with_obstacle.zip' is in the current directory.")
        return

    print("🔥 Starting Autonomous Test Drive in Dynamic Traffic...")
    print("⚠️ Press 'q' in the video window to stop.")
    
    test_episodes = 3
    
    for ep in range(test_episodes):
        # Reset environment for a new episode
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # 🟢 CRITICAL: deterministic=True removes exploration noise.
            # We want the agent to use its pure learned policy.
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute the predicted action in the environment
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
                
                # Overlay Phase and Episode text
                cv2.putText(frame, f"PHASE 3 DYNAMIC TEST - Ep: {ep + 1}/{test_episodes}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Overlay Speed
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Overlay VLM Status (Green for SAFE, Red for DANGER)
                vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
                vlm_status = "SAFE (Path Clear)" if vlm_signal > 0.5 else "DANGER (Obstacle!)"
                cv2.putText(frame, f"VLM: {vlm_status}", 
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vlm_color, 2)
                
                # Overlay Rewards
                cv2.putText(frame, f"Step Reward: {reward:.1f}", 
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                cv2.putText(frame, f"Total Reward: {episode_reward:.1f}", 
                            (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Show the combined visual output
                cv2.imshow("Autonomous Driving - Dynamic Traffic", frame)
                
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
    print("✅ Phase 3 Testing Completed!")

if __name__ == "__main__":
    main()