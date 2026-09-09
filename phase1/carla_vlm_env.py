import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import requests
import time

class CarlaVLMEnv(gym.Env):
    """
    Advanced CARLA RL Environment:
    Coordinates path-tracking geometry with VLM safety interventions 
    and handles exact coordinate-based episode termination.
    """
    
    def __init__(self):
        super(CarlaVLMEnv, self).__init__()
        
        # ACTION SPACE: [Acceleration, Steering]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: 6D Vector
        # [Speed, Cross-Track Error, Heading Error, Distance to WP, Angle to WP, VLM Signal]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Connect to CARLA via ZMQ Bridge
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000  
        self.socket.connect("tcp://localhost:5555")
        print("✅ RL Environment Connected to CARLA Path-Planner!")
        
        self.step_count = 0
        self.last_vlm_signal = 1.0  # Assume path is safe initially

    def _reconnect_zmq(self):
        """Clears ZMQ socket state to prevent deadlocks."""
        print("🔄 Reconnecting ZMQ socket to clear broken state...")
        self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000
        self.socket.connect("tcp://localhost:5555")

    def reset(self, seed=None, options=None):
        """Resets the environment."""
        super().reset(seed=seed)
        self.step_count = 0
        self.last_vlm_signal = 1.0 
        
        self.socket.send_json({"command": "reset"})
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            print("⚠️ ZMQ Timeout during reset. Reconnecting...")
            self._reconnect_zmq()
            return np.zeros(6, dtype=np.float32), {}
        
        observation = np.array([
            state.get("speed", 0.0),
            state.get("cte", 0.0),
            state.get("heading_error", 0.0),
            state.get("dist_to_wp", 0.0),
            state.get("angle_to_wp", 0.0),
            self.last_vlm_signal
        ], dtype=np.float32)
        
        return observation, {"image": state.get("image", "")}

    def step(self, action):
        """
        Executes action, receives telemetry, checks VLM safety,
        and uses the original stable reward function.
        """
        accel_raw = float(action[0])
        steer_raw = float(action[1])
        
        # Map acceleration [-1, 1] to Throttle and Brake [0, 1]
        if accel_raw >= 0:
            throttle, brake = accel_raw, 0.0
        else:
            throttle, brake = 0.0, abs(accel_raw)
            
        # Send action to CARLA bridge via ZMQ
        self.socket.send_json({
            "command": "step",
            "throttle": throttle,
            "steer": steer_raw,
            "brake": brake
        })
        
        # Receive current state from CARLA
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            print("⚠️ ZMQ Timeout during step. Reconnecting...")
            self._reconnect_zmq()
            return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

        # Extract telemetry data
        speed = state.get("speed", 0.0)
        cte = state.get("cte", 0.0)
        heading_error = state.get("heading_error", 0.0)
        dist_to_wp = state.get("dist_to_wp", 0.0)
        angle_to_wp = state.get("angle_to_wp", 0.0)
        dist_to_target = state.get("dist_to_target", 999.0)  
        img_base64 = state.get("image", "")
        
        done = False
        info = {"image": img_base64}

        # -------------------------------------------------------------
        # 👁️ VLM Safety Check (Every 5 steps = 0.25s)
        # -------------------------------------------------------------
        vlm_immediate_penalty = 0.0
        if self.step_count % 5 == 0 and img_base64:
            vlm_reward, _, vlm_signal = self.get_vlm_feedback(img_base64)
            vlm_immediate_penalty = vlm_reward
            self.last_vlm_signal = vlm_signal
            
        reward = 0.0

        # ---------------------------------------------------------
        # 🎯 TERMINATION LOGIC (Simple 2.5m Radius)
        # ---------------------------------------------------------
        
        # 1. SUCCESS: Reached target area (Distance < 2.5 meters)
        if dist_to_target < 2.5:
            print("🎯 Target Coordinates Reached (Within 2.5m)! Episode Completed Successfully.")
            reward += 100.0  # Massive success bonus
            done = True
            return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

        # 2. FAILURE: Deviated too far from the green path (Off-road)
        if abs(cte) > 3.5:  
            print("❌ Vehicle Off-Lane (CTE > 3.5m)! Episode Terminated.")
            reward -= 50.0   # Failure penalty
            done = True      
            return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

        # 3. SAFETY TIMEOUT: Maximum step guard
        if self.step_count > 600:
            print("⏳ Max steps reached. Terminating.")
            done = True

        # ---------------------------------------------------------
        # 🧠 NORMAL REWARD COMPUTATION (Original Stable Formula)
        # ---------------------------------------------------------
        if self.last_vlm_signal == 1.0:
            # NORMAL DRIVING MODE (Path is SAFE)
            speed_reward = speed * 0.1 if speed < 25.0 else 2.5
            cte_penalty = abs(cte) * 0.5                             # Linear penalty (Stable!)
            heading_penalty = (abs(heading_error) / 180.0) * 2.0
            
            reward += (speed_reward - cte_penalty - heading_penalty)
            
            # Penalize the agent if it stops unnecessarily when the road is clear
            if speed < 1.0:
                reward -= 1.0 
        else:
            # EMERGENCY BRAKING MODE (VLM detected an obstacle)
            if speed > 1.0:
                reward -= 10.0  # Heavy penalty for moving while blocked
            else:
                reward += 2.0   # Reward for successfully stopping

        # Incorporate VLM-triggered immediate rewards/penalties
        reward += vlm_immediate_penalty
        self.step_count += 1
        
        return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

    def _get_obs(self, speed, cte, heading_error, dist_to_wp, angle_to_wp):
        """Constructs the 6D observation vector."""
        return np.array([
            speed, cte, heading_error, dist_to_wp, angle_to_wp, self.last_vlm_signal
        ], dtype=np.float32)

    def get_vlm_feedback(self, img_base64):
        """Semantic VLM Feedback via Moondream2."""
        prompt = "Look at the road ahead. Is the path clear, or is there a physical obstacle like a car or pedestrian blocking it?"
        
        try:
            payload = {
                "model": "moondream",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 30,  
                    "num_ctx": 1024
                }
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5.0)
            
            if response.status_code == 200:
                answer = response.json().get("response", "").strip().lower()
                print(f"\n👨‍🏫 VLM Semantic Teacher: '{answer}'")
                
                if not answer:
                    return 2.0, False, 1.0

                # 1. HARD FILTER FOR SIMULATOR HALLUCINATIONS
                hallucination_triggers = [
                    "green arrow", "white arrow", "yellow arrow", 
                    "green line", "yellow line", "dotted line", 
                    "grid pattern", "checkered pattern", "green square"
                ]
                
                if any(trigger in answer for trigger in hallucination_triggers):
                    print("✅ Semantic Parser: VLM hallucinated a ground marking. Forcing SAFE.")
                    return 2.0, False, 1.0

                # 2. NEGATION HANDLING
                negations = [
                    "no obstruction", "no visible obstruction", "no obstacles", "no visible obstacles",
                    "no cars", "empty of cars", "empty of traffic", "no traffic", "parked", 
                    "no person", "no pedestrians"
                ]
                if any(neg in answer for neg in negations):
                    print("✅ Semantic Parser: 'No Obstacle' detected. Path is Clear.")
                    return 2.0, False, 1.0

                # 3. SEMANTIC INTENT PARSING
                danger_indicators = ["blocked", "obstruction", "obstacle", "blocking", "red light"]
                safe_indicators = ["clear", "smooth", "safe", "open", "free", "empty"]

                if "stop" in answer and "when approaching" not in answer:
                    danger_indicators.append("stop")

                has_danger = any(word in answer for word in danger_indicators)
                has_safe = any(word in answer for word in safe_indicators)

                if has_danger and not has_safe:
                    print("🚨 Semantic Parser: Real Physical Obstacle Confirmed!")
                    return -5.0, False, 0.0  
                elif has_safe:
                    print("✅ Semantic Parser: Path is Clear.")
                    return 2.0, False, 1.0   
                else:
                    return 1.0, False, 1.0
                     
        except Exception as e:
            pass 
            
        return 0.0, False, self.last_vlm_signal