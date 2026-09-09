import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    print("🚗 Initializing CARLA-VLM Environment for PRE-TRAINING VISUAL CHECK...")
    env = CarlaVLMEnv()
    
    # 🧠 Load the Phase 1 master brain (40,000 steps)
    model_path = "sac_carla_phase1_no_obstacle"
    print(f"🧠 Loading Phase 1 Model from {model_path}.zip...")
    
    try:
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the .zip file is in the same directory.")
        return

    print("🔥 Starting Visual Check (Press 'q' in the video window to stop)...")
    print("⚠️ EXPECTATION: The agent will drive smoothly but CRASH into the Van, because it hasn't learned to brake yet!")
    
    test_episodes = 3
    
    for ep in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # گرفتن اکشن خام و بدون فیلتر از مدل
            action, _states = model.predict(obs, deterministic=True)
            
            # ارسال مستقیم اکشن به محیط
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # 📺 Visual Dashboard
            if "image" in info and info["image"]:
                img_data = base64.b64decode(info["image"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                speed = obs[0]
                cte = obs[1]
                vlm_signal = obs[5]
                steer_cmd = action[1]
                
                # Overlay Text
                cv2.putText(frame, f"PRE-TRAINING CHECK - Ep: {ep + 1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Raw Steer: {steer_cmd:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Check VLM Status
                vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
                vlm_status = "SAFE (Path Clear)" if vlm_signal > 0.5 else "DANGER (Obstacle!)"
                cv2.putText(frame, f"VLM: {vlm_status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vlm_color, 2)
                
                cv2.putText(frame, f"Step Reward: {reward:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Total Reward: {episode_reward:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Flash warning if VLM sees obstacle but car is still moving fast
                if vlm_signal < 0.5 and speed > 1.0:
                    cv2.putText(frame, "IMMINENT COLLISION - IGNORING VLM!", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                
                cv2.imshow("CARLA Phase 2 - Pre-Train Visualizer", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return
        
        print(f"🏁 Episode {ep + 1} Ended | Total Reward: {episode_reward:.2f} | Steps: {step_count}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()