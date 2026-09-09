import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    """
    Test script for Phase 5: Dynamic Intersections & Right-of-Way.
    Loads the newly initialized Phase 5 model to execute autonomous driving using VLM Perception.
    """
    print("🚗 Initializing CARLA-VLM Environment for PHASE 5 TEST...")
    env = CarlaVLMEnv()
    
    model_path = "sac_carla_phase5_init"
    print(f"🧠 Loading trained brain from {model_path}.zip...")
    
    try:
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Phase 5 Model loaded successfully for testing!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Make sure '{model_path}.zip' is in the current directory.")
        return

    print("🔥 Starting Autonomous Test Drive with VLM Perception...")
    print("⚠️ Press 'q' in the video window to stop.")
    
    test_episodes = 5
    
    for ep in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            if "image" in info and info["image"]:
                img_data = base64.b64decode(info["image"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                speed = obs[0]
                vlm_safety = obs[5] 
                radar_dist = obs[6]
                
                cv2.putText(frame, f"PHASE 5: VLM PERCEPTION - Ep: {ep + 1}/{test_episodes}", 
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(frame, f"Speed: {speed:.1f} km/h | Radar: {radar_dist:.1f}m", 
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if vlm_safety >= 0.8:
                    status_text = "SAFE (Path Clear)"
                    color = (0, 255, 0)
                elif vlm_safety >= 0.3:
                    status_text = "CAUTION (Yield/Slow)"
                    color = (0, 255, 255)  # Yellow
                else:
                    status_text = "DANGER (Stop!)"
                    color = (0, 0, 255)
                    
                cv2.putText(frame, f"VLM Status: {status_text}", 
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.putText(frame, f"Step Reward: {reward:.1f}", 
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
                cv2.imshow("Autonomous Driving - Dynamic Traffic", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Manual stop requested by user.")
                    env.close()
                    cv2.destroyAllWindows()
                    return
        
        print(f"🏁 Episode {ep + 1} Ended | Total Reward: {episode_reward:.2f} | Steps: {step_count}")

    env.close()
    cv2.destroyAllWindows()
    print("✅ Phase 5 Testing Completed!")

if __name__ == "__main__":
    main()
