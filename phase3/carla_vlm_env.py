# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import zmq
# import requests
# import re

# class CarlaVLMEnv(gym.Env):
#     """
#     Advanced CARLA RL Environment:
#     Features a highly strict Semantic Parser using explicit Word Boundaries
#     and Context Filters to eliminate hallucinations regarding the ego-vehicle's hood, 
#     road markings, parked cars, and traffic rules.
#     Implements a robust Stop-and-Go reward system to prevent reward farming.
#     """
    
#     def __init__(self):
#         super(CarlaVLMEnv, self).__init__()
        
#         # ACTION SPACE: [Acceleration, Steering]
#         # Acceleration range: [-1.0 (Full Brake), 1.0 (Full Throttle)]
#         # Steering range: [-1.0 (Full Left), 1.0 (Full Right)]
#         self.action_space = spaces.Box(
#             low=np.array([-1.0, -1.0]), 
#             high=np.array([1.0, 1.0]), 
#             dtype=np.float32
#         )
        
#         # OBSERVATION SPACE: 6D Vector
#         # [Speed, Cross-Track Error, Heading Error, Dist to WP, Angle to WP, VLM Signal]
#         self.observation_space = spaces.Box(
#             low=-np.inf, 
#             high=np.inf, 
#             shape=(6,), 
#             dtype=np.float32
#         )
        
#         # Setup ZMQ connection to the CARLA path-planning server
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.REQ)
#         self.socket.RCVTIMEO = 30000  # 30-second timeout to prevent infinite hangs
#         self.socket.connect("tcp://localhost:5555")
#         print("✅ RL Environment Connected to CARLA Path-Planner via ZMQ!")
        
#         # Environment State Variables
#         self.step_count = 0
#         self.last_vlm_signal = 1.0        # 1.0 = SAFE, 0.0 = DANGER
#         self.danger_cooldown = 0          # Maintains danger state for stability
#         self.danger_violation_steps = 0   # Tracks consecutive steps of ignoring the brake

#     def _reconnect_zmq(self):
#         """Clears ZMQ socket state to prevent deadlocks upon timeout."""
#         print("🔄 Reconnecting ZMQ socket to clear broken state...")
#         self.socket.close()
#         self.socket = self.context.socket(zmq.REQ)
#         self.socket.RCVTIMEO = 30000
#         self.socket.connect("tcp://localhost:5555")

#     def reset(self, seed=None, options=None):
#         """Resets the environment and state variables for a new episode."""
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.last_vlm_signal = 1.0 
#         self.danger_cooldown = 0  
#         self.danger_violation_steps = 0 
        
#         # Request environment reset from CARLA
#         self.socket.send_json({"command": "reset"})
#         try:
#             state = self.socket.recv_json()
#         except zmq.error.Again:
#             print("⚠️ ZMQ Timeout during reset. Reconnecting...")
#             self._reconnect_zmq()
#             return np.zeros(6, dtype=np.float32), {}
        
#         # Construct initial observation
#         observation = self._get_obs(
#             state.get("speed", 0.0),
#             state.get("cte", 0.0),
#             state.get("heading_error", 0.0),
#             state.get("dist_to_wp", 0.0),
#             state.get("angle_to_wp", 0.0)
#         )
        
#         return observation, {"image": state.get("image", "")}

#     def step(self, action):
#         """Executes action, evaluates VLM safety, and calculates rewards."""
#         accel_raw = float(action[0])
#         steer_raw = float(action[1])
        
#         # Map raw acceleration [-1.0, 1.0] to throttle and brake [0.0, 1.0]
#         if accel_raw >= 0:
#             throttle, brake = accel_raw, 0.0
#         else:
#             throttle, brake = 0.0, abs(accel_raw)
            
#         # Send control commands to CARLA via ZMQ
#         self.socket.send_json({
#             "command": "step",
#             "throttle": throttle,
#             "steer": steer_raw,
#             "brake": brake
#         })
        
#         # Receive telemetry from CARLA
#         try:
#             state = self.socket.recv_json()
#         except zmq.error.Again:
#             self._reconnect_zmq()
#             return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

#         # Extract telemetry fields
#         speed = state.get("speed", 0.0)
#         cte = state.get("cte", 0.0)
#         heading_error = state.get("heading_error", 0.0)
#         dist_to_wp = state.get("dist_to_wp", 0.0)
#         angle_to_wp = state.get("angle_to_wp", 0.0)
#         dist_to_target = state.get("dist_to_target", 999.0)  
#         img_base64 = state.get("image", "")
        
#         done = False
#         info = {"image": img_base64}

#         vlm_immediate_penalty = 0.0
        
#         # Decrease memory cooldown for danger signal
#         if self.danger_cooldown > 0:
#             self.danger_cooldown -= 1

#         # Query the VLM every 5 steps (approx. every 0.25 seconds in simulation)
#         if self.step_count % 5 == 0 and img_base64:
#             vlm_reward, _, raw_vlm_signal = self.get_vlm_feedback(img_base64)
#             vlm_immediate_penalty = vlm_reward
            
#             # Apply memory to smooth out VLM inconsistencies
#             if raw_vlm_signal == 0.0:
#                 self.danger_cooldown = 20  
#                 self.last_vlm_signal = 0.0
#             elif self.danger_cooldown == 0:
#                 self.last_vlm_signal = 1.0
            
#         # ---------------------------------------------------------
#         # 🎯 TERMINATION LOGIC
#         # ---------------------------------------------------------
#         if dist_to_target < 2.5:
#             print("🎯 Target Coordinates Reached! Episode Completed Successfully.")
#             reward = 100.0 
#             done = True
#             return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

#         if abs(cte) > 3.5:  
#             print("❌ Vehicle Off-Lane (CTE > 3.5m)! Episode Terminated.")
#             reward = -50.0  
#             done = True      
#             return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

#         if self.step_count > 500:
#             print("⏳ Max steps reached. Terminating.")
#             done = True

#         # ---------------------------------------------------------
#         # 🧠 REWARD COMPUTATION
#         # ---------------------------------------------------------
#         reward = 0.0
        
#         if self.last_vlm_signal == 1.0:
#             # 🟢 SAFE MODE: The road is completely clear.
#             self.danger_violation_steps = 0 
            
#             # Boosted speed reward for Phase 3 to encourage faster driving
#             speed_reward = speed * 0.2 if speed < 25.0 else 3.0
#             cte_penalty = abs(cte) * 0.5                             
#             heading_penalty = (abs(heading_error) / 180.0) * 2.0
            
#             reward = speed_reward - cte_penalty - heading_penalty
            
#             # ❌ ANTI-FARMING PENALTY: 
#             # Penalize only if the car is practically stopped (speed < 1.0) on a clear road.
#             if speed < 1.0 and self.danger_cooldown == 0:
#                 reward -= 5.0  
#         else:
#             # 🔴 DANGER MODE: An obstacle is blocking the path.
#             if speed > 0.5:
#                 # ❌ PENALTY: Agent is ignoring the obstacle and moving.
#                 self.danger_violation_steps += 1
                
#                 # Progressive penalty ensures the agent learns to fear collision
#                 progressive_penalty = -10.0 - (self.danger_violation_steps * 5.0)
#                 progressive_penalty = max(progressive_penalty, -50.0) 
#                 reward += progressive_penalty
#             else:
#                 # ✅ REWARD: Agent successfully stopped for the obstacle.
#                 self.danger_violation_steps = 0 
#                 reward += 10.0   

#         reward += vlm_immediate_penalty
#         self.step_count += 1
        
#         return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

#     def _get_obs(self, speed, cte, heading_error, dist_to_wp, angle_to_wp):
#         """Constructs the observation vector for the RL agent."""
#         return np.array([
#             speed, cte, heading_error, dist_to_wp, angle_to_wp, self.last_vlm_signal
#         ], dtype=np.float32)

#     def get_vlm_feedback(self, img_base64):
#         """Queries Moondream2 and processes the semantic response."""
        
#         # Explicitly command VLM to ignore the car's own hood at the bottom of the image
#         prompt = "Look straight ahead at the road. IGNORE the hood of the vehicle at the bottom of the image. Is the immediate path ahead clear, or is there a physical obstacle blocking it?"
        
#         try:
#             payload = {
#                 "model": "moondream",
#                 "prompt": prompt,
#                 "images": [img_base64],
#                 "stream": False,
#                 "options": {
#                     "temperature": 0.1,
#                     "num_predict": 30,  
#                     "num_ctx": 1024
#                 }
#             }
            
#             # Timeout set to 5.0 seconds
#             response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5.0)
#             if response.status_code == 200:
#                 answer = response.json().get("response", "").strip().lower()
#                 print(f"\n👨‍🏫 VLM Semantic Teacher: '{answer}'")
                
#                 if not answer:
#                     return 0.0, False, 1.0

#                 # -------------------------------------------------------------
#                 # 🛑 BULLETPROOF FILTERING (Fixing Background & Rule Hallucinations)
#                 # -------------------------------------------------------------
                
#                 # 1. Ignore parked cars or background context
#                 safe_contexts = [
#                     r".*parked.*",
#                     r".*in the background.*",
#                     r".*in the distance.*",
#                     r".*on the side of the.*",
#                     r".*on the sidewalk.*",
#                     # Ignore vehicles in the opposite lane, left side, or behind
#                     r".*left side.*", 
#                     r".*opposite lane.*", 
#                     r".*opposite direction.*",
#                     r".*following behind.*", 
#                     r".*behind.*",
#                     # Ignore hallucinations where VLM describes the ego-vehicle itself
#                     r".*driving down the street.*",
#                     r".*driving down it.*",
#                     r".*bicycle on the right.*"
#                 ]
                
#                 clean_text = answer
#                 for context_pattern in safe_contexts:
#                     if re.search(context_pattern, clean_text):
#                         # Extended removal list to ensure no leftover words trigger the danger signal
#                         clean_text = re.sub(r"(cars?|vans?|vehicles?|trucks?|buses?|bicycles?|pedestrians?|people|person)", "", clean_text)
                
#                 # 2. Remove negations completely (e.g. "there are no cars, vans, or pedestrians")
#                 clean_text = re.sub(r"\bno\b[^.]*", "", clean_text)

#                 # 3. Remove confusing rule descriptions and ground markings
#                 noise_phrases = [
#                     "clear of", "empty of",
#                     "where cars", "where vehicles", "that cars", "that vehicles",
#                     "for cars", "for vehicles", "allow cars", "allow pedestrians",
#                     "pedestrians should", "cars should", "vehicles should",
#                     "green line", "yellow line", "dotted line", "green arrow", "yellow arrow",
#                     "crosswalk", "grid pattern"
#                 ]
                
#                 for noise in noise_phrases:
#                     clean_text = clean_text.replace(noise, "")
                
#                 # 4. 🚨 EXPLICIT OBSTACLE DETECTION
#                 # We specifically look for the presence of a single vehicle/person using strict word boundaries.
#                 obstacle_patterns = [
#                     r"\ba car\b", r"\bthe car\b", r"\bred car\b", r"\bwhite car\b", r"\bblack car\b", r"\bblue car\b", r"\bsilver car\b",
#                     r"\ba van\b", r"\bthe van\b", r"\bred van\b", r"\bwhite van\b", r"\bparked van\b",
#                     r"\ban suv\b", r"\bthe suv\b", 
#                     r"\ba truck\b", r"\bthe truck\b", 
#                     r"\ba bus\b", r"\bthe bus\b", 
#                     r"\ba pedestrian\b", r"\bthe pedestrian\b", r"\bpeople\b", r"\bperson\b"
#                 ]
                
#                 # Check if any strict obstacle pattern exists in the cleaned text
#                 obstacle_detected = any(re.search(pat, clean_text) for pat in obstacle_patterns)
                
#                 if obstacle_detected:
#                     print("🚨 Semantic Parser: Explicit Physical Obstacle DETECTED! Forcing DANGER.")
#                     return 0.0, False, 0.0  # Signal = 0.0 (DANGER)
                
#                 # 5. Default state if no explicit obstacle is found
#                 print("✅ Semantic Parser: Path appears Clear. Defaulting to SAFE.")
#                 return 0.0, False, 1.0   
                
#         except Exception as e:
#             pass 
            
#         return 0.0, False, self.last_vlm_signal



import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import requests
import re

class CarlaVLMEnv(gym.Env):
    """
    Advanced CARLA RL Environment:
    Features a highly strict Semantic Parser using explicit Word Boundaries
    and Context Filters to eliminate hallucinations regarding the ego-vehicle's hood, 
    road markings, parked cars, and traffic rules.
    Implements a robust Stop-and-Go reward system to prevent reward farming.
    """
    
    def __init__(self):
        super(CarlaVLMEnv, self).__init__()
        
        # ACTION SPACE: [Acceleration, Steering]
        # Acceleration range: [-1.0 (Full Brake), 1.0 (Full Throttle)]
        # Steering range: [-1.0 (Full Left), 1.0 (Full Right)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # OBSERVATION SPACE: 6D Vector
        # [Speed, Cross-Track Error, Heading Error, Dist to WP, Angle to WP, VLM Signal]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Setup ZMQ connection to the CARLA path-planning server
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000  # 30-second timeout to prevent infinite hangs
        self.socket.connect("tcp://localhost:5555")
        print("✅ RL Environment Connected to CARLA Path-Planner via ZMQ!")
        
        # Environment State Variables
        self.step_count = 0
        self.last_vlm_signal = 1.0        # 1.0 = SAFE, 0.0 = DANGER
        self.danger_cooldown = 0          # Maintains danger state for stability
        self.danger_violation_steps = 0   # Tracks consecutive steps of ignoring the brake

    def _reconnect_zmq(self):
        """Clears ZMQ socket state to prevent deadlocks upon timeout."""
        print("🔄 Reconnecting ZMQ socket to clear broken state...")
        self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000
        self.socket.connect("tcp://localhost:5555")

    def reset(self, seed=None, options=None):
        """Resets the environment and state variables for a new episode."""
        super().reset(seed=seed)
        self.step_count = 0
        self.last_vlm_signal = 1.0 
        self.danger_cooldown = 0  
        self.danger_violation_steps = 0 
        
        # Request environment reset from CARLA
        self.socket.send_json({"command": "reset"})
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            print("⚠️ ZMQ Timeout during reset. Reconnecting...")
            self._reconnect_zmq()
            return np.zeros(6, dtype=np.float32), {}
        
        # Construct initial observation
        observation = self._get_obs(
            state.get("speed", 0.0),
            state.get("cte", 0.0),
            state.get("heading_error", 0.0),
            state.get("dist_to_wp", 0.0),
            state.get("angle_to_wp", 0.0)
        )
        
        return observation, {"image": state.get("image", "")}

    def step(self, action):
        """Executes action, evaluates VLM safety, and calculates rewards."""
        accel_raw = float(action[0])
        steer_raw = float(action[1])
        
        # Map raw acceleration [-1.0, 1.0] to throttle and brake [0.0, 1.0]
        if accel_raw >= 0:
            throttle, brake = accel_raw, 0.0
        else:
            throttle, brake = 0.0, abs(accel_raw)
            
        # Send control commands to CARLA via ZMQ
        self.socket.send_json({
            "command": "step",
            "throttle": throttle,
            "steer": steer_raw,
            "brake": brake
        })
        
        # Receive telemetry from CARLA
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            self._reconnect_zmq()
            return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

        # Extract telemetry fields
        speed = state.get("speed", 0.0)
        cte = state.get("cte", 0.0)
        heading_error = state.get("heading_error", 0.0)
        dist_to_wp = state.get("dist_to_wp", 0.0)
        angle_to_wp = state.get("angle_to_wp", 0.0)
        dist_to_target = state.get("dist_to_target", 999.0)  
        img_base64 = state.get("image", "")
        
        done = False
        info = {"image": img_base64}

        vlm_immediate_penalty = 0.0
        
        # Decrease memory cooldown for danger signal
        if self.danger_cooldown > 0:
            self.danger_cooldown -= 1

        # Query the VLM every 5 steps (approx. every 0.25 seconds in simulation)
        if self.step_count % 5 == 0 and img_base64:
            vlm_reward, _, raw_vlm_signal = self.get_vlm_feedback(img_base64)
            vlm_immediate_penalty = vlm_reward
            
            # Apply memory to smooth out VLM inconsistencies
            if raw_vlm_signal == 0.0:
                self.danger_cooldown = 20  
                self.last_vlm_signal = 0.0
            elif self.danger_cooldown == 0:
                self.last_vlm_signal = 1.0
            
        # ---------------------------------------------------------
        # 🎯 TERMINATION LOGIC
        # ---------------------------------------------------------
        if dist_to_target < 2.5:
            print("🎯 Target Coordinates Reached! Episode Completed Successfully.")
            reward = 100.0 
            done = True
            return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

        if abs(cte) > 3.5:  
            print("❌ Vehicle Off-Lane (CTE > 3.5m)! Episode Terminated.")
            reward = -50.0  
            done = True      
            return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

        if self.step_count > 500:
            print("⏳ Max steps reached. Terminating.")
            done = True

        # ---------------------------------------------------------
        # 🧠 REWARD COMPUTATION
        # ---------------------------------------------------------
        reward = 0.0
        
        if self.last_vlm_signal == 1.0:
            # 🟢 SAFE MODE: The road is completely clear.
            self.danger_violation_steps = 0 
            
            # Boosted speed reward for Phase 3 to encourage faster driving
            speed_reward = speed * 0.5 if speed < 25.0 else 5.0
            cte_penalty = abs(cte) * 0.5                             
            heading_penalty = (abs(heading_error) / 180.0) * 2.0
            
            reward = speed_reward - cte_penalty - heading_penalty
            
            # ❌ ANTI-FARMING PENALTY: 
            # Penalize only if the car is practically stopped (speed < 1.0) on a clear road.
            if speed < 1.0 and self.danger_cooldown == 0:
                reward -= 20.0  
        else:
            # 🔴 DANGER MODE: An obstacle is blocking the path.
            if speed > 0.5:
                # ❌ PENALTY: Agent is ignoring the obstacle and moving.
                self.danger_violation_steps += 1
                
                # Progressive penalty ensures the agent learns to fear collision
                progressive_penalty = -10.0 - (self.danger_violation_steps * 5.0)
                progressive_penalty = max(progressive_penalty, -50.0) 
                reward += progressive_penalty
            else:
                # ✅ REWARD: Agent successfully stopped for the obstacle.
                self.danger_violation_steps = 0 
                reward += 10.0   

        reward += vlm_immediate_penalty
        self.step_count += 1
        
        return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp), reward, done, False, info

    def _get_obs(self, speed, cte, heading_error, dist_to_wp, angle_to_wp):
        """Constructs the observation vector for the RL agent."""
        return np.array([
            speed, cte, heading_error, dist_to_wp, angle_to_wp, self.last_vlm_signal
        ], dtype=np.float32)

    def get_vlm_feedback(self, img_base64):
        """Queries Moondream2 and processes the semantic response."""
        
        # Explicitly command VLM to ignore the car's own hood at the bottom of the image
        prompt = "Look straight ahead at the road. IGNORE the hood of the vehicle at the bottom of the image. Is the immediate path ahead clear, or is there a physical obstacle blocking it?"
        
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
            
            # Timeout set to 5.0 seconds
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5.0)
            if response.status_code == 200:
                answer = response.json().get("response", "").strip().lower()
                print(f"\n👨‍🏫 VLM Semantic Teacher: '{answer}'")
                
                if not answer:
                    return 0.0, False, 1.0

                # -------------------------------------------------------------
                # 🛑 BULLETPROOF FILTERING (Fixing Background & Rule Hallucinations)
                # -------------------------------------------------------------
                
                # 1. Ignore parked cars or background context
                safe_contexts = [
                    r".*parked.*",
                    r".*in the background.*",
                    r".*in the distance.*",
                    r".*on the side of the.*",
                    r".*on the sidewalk.*",
                    # Ignore vehicles in the opposite lane, left side, or behind
                    r".*left side.*", 
                    r".*opposite lane.*", 
                    r".*opposite direction.*",
                    r".*following behind.*", 
                    r".*behind.*",
                    # Ignore hallucinations where VLM describes the ego-vehicle itself
                    r".*driving down the street.*",
                    r".*driving down it.*",
                    r".*bicycle on the right.*"
                ]
                
                clean_text = answer
                for context_pattern in safe_contexts:
                    if re.search(context_pattern, clean_text):
                        # Extended removal list to ensure no leftover words trigger the danger signal
                        clean_text = re.sub(r"(cars?|vans?|vehicles?|trucks?|buses?|bicycles?|pedestrians?|people|person)", "", clean_text)
                
                # 2. Remove negations completely (e.g. "there are no cars, vans, or pedestrians")
                clean_text = re.sub(r"\bno\b[^.]*", "", clean_text)

                # 3. Remove confusing rule descriptions and ground markings
                noise_phrases = [
                    "clear of", "empty of",
                    "where cars", "where vehicles", "that cars", "that vehicles",
                    "for cars", "for vehicles", "allow cars", "allow pedestrians",
                    "pedestrians should", "cars should", "vehicles should",
                    "green line", "yellow line", "dotted line", "green arrow", "yellow arrow",
                    "crosswalk", "grid pattern"
                ]
                
                for noise in noise_phrases:
                    clean_text = clean_text.replace(noise, "")
                
                # 4. 🚨 EXPLICIT OBSTACLE DETECTION
                # We specifically look for the presence of a single vehicle/person using strict word boundaries.
                obstacle_patterns = [
                    r"\ba car\b", r"\bthe car\b", r"\bred car\b", r"\bwhite car\b", r"\bblack car\b", r"\bblue car\b", r"\bsilver car\b",
                    r"\ba van\b", r"\bthe van\b", r"\bred van\b", r"\bwhite van\b", r"\bparked van\b",
                    r"\ban suv\b", r"\bthe suv\b", 
                    r"\ba truck\b", r"\bthe truck\b", 
                    r"\ba bus\b", r"\bthe bus\b", 
                    r"\ba pedestrian\b", r"\bthe pedestrian\b", r"\bpeople\b", r"\bperson\b"
                ]
                
                # Check if any strict obstacle pattern exists in the cleaned text
                obstacle_detected = any(re.search(pat, clean_text) for pat in obstacle_patterns)
                
                if obstacle_detected:
                    print("🚨 Semantic Parser: Explicit Physical Obstacle DETECTED! Forcing DANGER.")
                    return 0.0, False, 0.0  # Signal = 0.0 (DANGER)
                
                # 5. Default state if no explicit obstacle is found
                print("✅ Semantic Parser: Path appears Clear. Defaulting to SAFE.")
                return 0.0, False, 1.0   
                
        except Exception as e:
            pass 
            
        return 0.0, False, self.last_vlm_signal