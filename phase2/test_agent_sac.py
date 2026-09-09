import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    print("🚗 Initializing CARLA-VLM Environment for PHASE 2 TEST DRIVE...")
    env = CarlaVLMEnv()
    
    # 🧠 Define the path to the newly trained Phase 2 model
    model_path = "sac_carla_phase2_with_obstacle"
    print(f"🧠 Loading Phase 2 trained brain from {model_path}.zip...")
    
    try:
        # Load the trained model onto the CPU to save VRAM for CARLA and VLM
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Phase 2 Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Make sure '{model_path}.zip' exists in the current directory!")
        return

    print("🔥 Starting Autonomous Test Drive (Press 'q' in the video window to stop)...")
    print("⚠️ Expectation: Testing raw agent actions (No Steering Filter) against the Van and VLM signals.")
    
    # Number of episodes to test the agent
    test_episodes = 5
    
    for ep in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # 🟢 CRITICAL: deterministic=True
            # Exploits the learned policy without any random exploration noise.
            # It will take the absolute best action it knows for the current state.
            action, _states = model.predict(obs, deterministic=True)
            
            # --- RAW ACTION (NO FILTER) ---
            # We extract the raw steering directly from the network output for display
            raw_steer = action[1]
            
            # Send the raw action directly to the CARLA environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 📺 Visual Dashboard: Render the camera feed and live telemetry
            if "image" in info and info["image"]:
                # Decode the base64 image string received from the ZMQ bridge
                img_data = base64.b64decode(info["image"])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # Extract telemetry data from the 6D observation vector
                speed = obs[0]
                cte = obs[1]
                vlm_signal = obs[5]
                
                # Overlay Telemetry Text on the OpenCV frame
                cv2.putText(frame, f"PHASE 2 TEST - Ep: {ep + 1}/{test_episodes}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display Raw Steer Cmd
                cv2.putText(frame, f"Raw Steer: {raw_steer:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Change VLM text color based on safety signal (Green for SAFE, Red for DANGER)
                vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
                vlm_status = "SAFE (Path Clear)" if vlm_signal > 0.5 else "DANGER (Obstacle!)"
                cv2.putText(frame, f"VLM: {vlm_status}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vlm_color, 2)
                
                # Check if agent is cheating (Speed is 0 while path is SAFE)
                if speed < 1.0 and vlm_signal > 0.5:
                    cv2.putText(frame, "⚠️ AGENT IS IDLING (REWARD FARMING?)", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    cv2.putText(frame, f"Step Reward: {reward:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                cv2.putText(frame, f"Total Reward: {episode_reward:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Display the frame in a window
                cv2.imshow("Autonomous Driving - Phase 2 Inference", frame)
                
                # Wait 1 ms and check if the user pressed 'q' to quit the test early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Manual stop requested by user.")
                    env.close()
                    cv2.destroyAllWindows()
                    return
        
        print(f"🏁 Episode {ep + 1} Ended | Total Reward: {episode_reward:.2f} | Steps Alive: {step_count}")

    # Clean up resources after testing is fully completed
    env.close()
    cv2.destroyAllWindows()
    print("✅ Phase 2 Inference completed!")

if __name__ == "__main__":
    main()