import cv2
import base64
import numpy as np
from stable_baselines3 import SAC
from carla_vlm_env import CarlaVLMEnv

def main():
    print("🚗 Initializing CARLA-VLM Environment for TEST DRIVE...")
    env = CarlaVLMEnv()
    
    # Define the path to the saved model from Phase 1
    model_path = "sac_carla_phase1_no_obstacle"
    print(f"🧠 Loading trained brain from {model_path}.zip...")
    
    try:
        # Load the trained model onto the CPU to save VRAM for CARLA and VLM
        model = SAC.load(model_path, env=env, device="cpu")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Make sure '{model_path}.zip' exists in the current directory!")
        return

    print("🔥 Starting Autonomous Test Drive (Press 'q' in the video window to stop)...")
    
    # Number of episodes to test the agent
    test_episodes = 5
    
    for ep in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # 🟢 CRITICAL: deterministic=True
            # This tells the agent to exploit its learned policy rather than exploring randomly.
            # It will take the absolute best action it knows for the current state.
            action, _states = model.predict(obs, deterministic=True)
            
            # Send the predicted action to the CARLA environment
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
                cv2.putText(frame, f"TEST DRIVE - Episode: {ep + 1}/{test_episodes}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"CTE (Error): {cte:.2f} m", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Change VLM text color based on safety signal (Green for SAFE, Red for DANGER)
                vlm_color = (0, 255, 0) if vlm_signal > 0.5 else (0, 0, 255)
                vlm_status = "SAFE" if vlm_signal > 0.5 else "DANGER"
                cv2.putText(frame, f"VLM Signal: {vlm_signal:.1f} ({vlm_status})", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vlm_color, 2)
                cv2.putText(frame, f"Total Reward: {episode_reward:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Display the frame in a window
                cv2.imshow("Autonomous Driving - Live AI", frame)
                
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
    print("✅ Inference completed! The car should have driven smoothly.")

if __name__ == "__main__":
    main()