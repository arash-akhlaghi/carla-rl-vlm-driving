import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import requests
import json as json_lib

class CarlaVLMEnv(gym.Env):
    """
    Advanced CARLA RL Environment for Phase 5.
    Implements Right-of-Way aware Sensor Fusion with structured VLM prompts.
    Uses CARLA ground-truth (traffic light, junction) + VLM semantic analysis + Radar.
    Added safety override for blind‑spot collisions using crossing_threat.
    """
    
    def __init__(self):
        super(CarlaVLMEnv, self).__init__()
        
        # Action space: [Acceleration, Steering]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        # Observation space is strictly 10D with backward compatible first 6 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000 
        self.socket.connect("tcp://localhost:5555")
        print("✅ RL Environment Connected to CARLA Path-Planner via ZMQ!")
        
        self.step_count = 0
        self.last_vlm_signal = 1.0        
        self.last_vlm_reason = "Init"
        self.current_scenario_id = 1
        self.crossing_threat = False
        self.crossing_threat_distance = 999.0
        # 🔒 Danger latch: once VLM reports stop/slow_down, hold that reading for
        # this many VLM-query cycles before trusting a "proceed" reading again.
        self.vlm_hold_steps = 5
        self.vlm_hold_counter = 0
        # Junction context from CARLA ground truth
        self.current_junction_ctx = {
            "in_junction": False,
            "approaching_junction": False,
            "junction_distance": 999.0,
            "traffic_light": "none",
            "vehicles_in_junction": 0
        }

    def _reconnect_zmq(self):
        """Safely re-establishes ZMQ connection if a timeout occurs."""
        self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000
        self.socket.connect("tcp://localhost:5555")

    def reset(self, seed=None, options=None):
        """Resets the RL episode."""
        super().reset(seed=seed)
        self.step_count = 0
        self.last_vlm_signal = 1.0 
        self.last_vlm_reason = "Init"
        self.vlm_hold_counter = 0
        self.crossing_threat = False
        self.crossing_threat_distance = 999.0
        self.current_junction_ctx = {
            "in_junction": False,
            "approaching_junction": False,
            "junction_distance": 999.0,
            "traffic_light": "none",
            "vehicles_in_junction": 0
        }
        
        self.socket.send_json({"command": "reset", "need_image": True})
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            self._reconnect_zmq()
            return np.zeros(10, dtype=np.float32), {}
        
        # Extract junction context from CARLA ground truth
        self.current_junction_ctx = {
            "in_junction": state.get("in_junction", False),
            "approaching_junction": state.get("approaching_junction", False),
            "junction_distance": state.get("junction_distance", 999.0),
            "traffic_light": state.get("traffic_light", "none"),
            "vehicles_in_junction": state.get("vehicles_in_junction", 0)
        }
        self.current_scenario_id = state.get("scenario_id", 1)
        self.crossing_threat = state.get("crossing_threat", False)
        self.crossing_threat_distance = state.get("crossing_threat_distance", 999.0)
        
        radar_dist = state.get("radar_distance", 30.0)
        
        obs = self._get_obs(
            state.get("speed", 0.0), 
            state.get("cte", 0.0), 
            state.get("heading_error", 0.0), 
            state.get("dist_to_wp", 0.0), 
            state.get("angle_to_wp", 0.0),
            radar_dist,
            self.current_junction_ctx.get("traffic_light", "none"),
            self.current_junction_ctx.get("in_junction", False),
            self.current_junction_ctx.get("vehicles_in_junction", 0)
        )
        return obs, {"image": state.get("image", "")}

    def step(self, action):
        """Executes one simulation step and calculates rewards."""
        accel_raw, steer_raw = float(action[0]), float(action[1])
        throttle, brake = (accel_raw, 0.0) if accel_raw >= 0 else (0.0, abs(accel_raw))
            
        # Sample the VLM more often right where right-of-way conflicts actually
        # happen, and stay coarse elsewhere to limit wall-clock cost.
        near_junction = (
            self.current_junction_ctx.get("in_junction", False)
            or self.current_junction_ctx.get("approaching_junction", False)
        )
        vlm_query_stride = 5 if not near_junction else 3
        need_image = (self.step_count % vlm_query_stride == 0)
        self.socket.send_json({
            "command": "step", 
            "throttle": throttle, 
            "steer": steer_raw, 
            "brake": brake,
            "need_image": need_image
        })
        
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            self._reconnect_zmq()
            return np.zeros(10, dtype=np.float32), 0.0, True, False, {}

        speed = state.get("speed", 0.0)
        cte = state.get("cte", 0.0)
        heading_error = state.get("heading_error", 0.0)
        dist_to_wp = state.get("dist_to_wp", 0.0)
        angle_to_wp = state.get("angle_to_wp", 0.0)
        radar_dist = state.get("radar_distance", 30.0)
        img_base64 = state.get("image", "")
        dist_to_target = state.get("dist_to_target", 999.0)
        collision = state.get("collision", False)
        
        # Update junction context from CARLA ground truth
        self.current_junction_ctx = {
            "in_junction": state.get("in_junction", False),
            "approaching_junction": state.get("approaching_junction", False),
            "junction_distance": state.get("junction_distance", 999.0),
            "traffic_light": state.get("traffic_light", "none"),
            "vehicles_in_junction": state.get("vehicles_in_junction", 0)
        }
        self.current_scenario_id = state.get("scenario_id", self.current_scenario_id)
        self.crossing_threat = bool(state.get("crossing_threat", False))
        self.crossing_threat_distance = float(state.get("crossing_threat_distance", 999.0))
        
        # ────────────────────────────────────────────────────────────────
        # SAFETY OVERRIDE FOR BLIND‑SPOT COLLISIONS
        # If a cross vehicle is close (< 12m), force a stop regardless of VLM/Radar
        # This ensures the ego doesn't get hit from directions not covered by sensors.
        # ────────────────────────────────────────────────────────────────
        if self.crossing_threat and self.crossing_threat_distance < 12.0:
            self.last_vlm_signal = 0.0
            self.vlm_hold_counter = self.vlm_hold_steps
            self.last_vlm_reason = "Safety override: cross vehicle too close"
        else:
            # Normal VLM perception with danger latch (only when not overridden)
            if self.step_count % vlm_query_stride == 0 and img_base64:
                _, _, raw_vlm_signal = self.get_vlm_feedback(img_base64)

                # Apply the same geometry validator to every scenario that contains a
                # cross-traffic vehicle. Scenarios 6 and 7 are intentionally excluded.
                cross_vehicle_scenarios = {1, 2, 3, 4, 5, 8}
                if self.current_scenario_id in cross_vehicle_scenarios:
                    if self.crossing_threat:
                        if raw_vlm_signal < 1.0:
                            self.last_vlm_signal = raw_vlm_signal
                            self.vlm_hold_counter = self.vlm_hold_steps
                        elif self.vlm_hold_counter > 0:
                            self.vlm_hold_counter -= 1
                        else:
                            self.last_vlm_signal = raw_vlm_signal
                    else:
                        # No genuine cross-traffic threat -> clear any stale VLM latch
                        # immediately, so irrelevant vehicles cannot force a STOP.
                        self.last_vlm_signal = 1.0
                        self.vlm_hold_counter = 0
                        self.last_vlm_reason = "Cross-vehicle gate: crossing path clear"
                else:
                    if raw_vlm_signal < 1.0:
                        # Danger/slow-down detected: latch it and (re)start the hold timer.
                        self.last_vlm_signal = raw_vlm_signal
                        self.vlm_hold_counter = self.vlm_hold_steps
                    elif self.vlm_hold_counter > 0:
                        # Still inside the post-danger hold window.
                        self.vlm_hold_counter -= 1
                    else:
                        self.last_vlm_signal = raw_vlm_signal

        # --- TERMINAL LOGGING ---
        if self.step_count % vlm_query_stride == 0:
            tl = self.current_junction_ctx["traffic_light"]
            in_j = self.current_junction_ctx["in_junction"]
            appr_j = self.current_junction_ctx["approaching_junction"]
            v_count = self.current_junction_ctx["vehicles_in_junction"]
            
            if radar_dist <= 3.0:
                radar_limit_kmh = 0.0
            elif radar_dist <= 8.0:
                radar_limit_kmh = ((radar_dist - 3.0) / 5.0) * 5.0
            else:
                radar_limit_kmh = 5.0 + ((min(radar_dist, 25.0) - 8.0) / 17.0) * 20.0
            vlm_limit_kmh = self.last_vlm_signal * 25.0
            
            if tl == "red":
                target_speed = 0.0
            elif tl == "yellow":
                target_speed = 10.0
            else:
                target_speed = min(radar_limit_kmh, vlm_limit_kmh)
            
            print(f"\n📊 --- SENSOR FUSION DASHBOARD (Step {self.step_count}) | Scenario: {self.current_scenario_id} ---")
            print(f"📡 RADAR : {radar_dist:.1f}m → Limit: {radar_limit_kmh:.1f} km/h")
            print(f"👁️  VLM   : {self.last_vlm_signal:.2f} → Limit: {vlm_limit_kmh:.1f} km/h")
            print(f"🧠 REASON : {getattr(self, 'last_vlm_reason', 'Init')}")
            print(f"🚦 LIGHT : {tl.upper()}")
            print(f"🔀 JUNCTION: {'IN' if in_j else 'APPROACHING' if appr_j else 'NONE'} | Vehicles: {v_count}")
            if self.current_scenario_id in {1, 2, 3, 4, 5, 8}:
                print(f"🛡️  CROSS-VEHICLE GATE: {'THREAT' if self.crossing_threat else 'CLEAR'} | Distance: {self.crossing_threat_distance:.1f}m")
            if tl == "red": print(f"🔴 ACTION: Red Light — MUST STOP!")
            print(f"🎯 TARGET: {target_speed:.1f} km/h")
            print(f"🚗 ACTUAL: {speed:.1f} km/h")
            print(f"--------------------------------------------------")

        # --- TERMINATION LOGIC ---
        tl_state = self.current_junction_ctx.get("traffic_light", "none")
        in_junction = self.current_junction_ctx.get("in_junction", False)
        v_count = self.current_junction_ctx.get("vehicles_in_junction", 0)

        if collision:
            print("💥 COLLISION DETECTED! Episode terminated.")
            return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count), -100.0, True, False, {"image": img_base64}
        if dist_to_target < 2.5: return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count), 400.0, True, False, {"image": img_base64}
        if abs(cte) > 3.5: return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count), -300.0, True, False, {"image": img_base64}
        if self.step_count > 500: return self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count), 0.0, True, False, {"image": img_base64}

        # ═══════════════════════════════════════════════════════════════════
        # DETERMINISTIC TARGET SPEED & REWARD
        # ═══════════════════════════════════════════════════════════════════
        if radar_dist <= 3.0:
            radar_target = 0.0
        elif radar_dist <= 8.0:
            radar_target = ((radar_dist - 3.0) / 5.0) * 5.0
        else:
            radar_target = 5.0 + ((min(radar_dist, 25.0) - 8.0) / 17.0) * 20.0
            
        vlm_target = self.last_vlm_signal * 25.0
        
        if tl_state == "red":
            target_speed = 0.0
        elif tl_state == "yellow":
            target_speed = 10.0
        else:
            target_speed = min(radar_target, vlm_target, 25.0)

        speed_error = abs(speed - target_speed)
        reward = -0.5 * speed_error
        reward -= 0.5 * abs(cte)
        reward -= 2.0 * (abs(heading_error) / 180.0)

        # Extra penalty for moving fast while a cross vehicle is close (blind spot)
        if self.crossing_threat and self.crossing_threat_distance < 12.0 and speed > 5.0:
            reward -= 10.0   # strongly discourage moving fast near a crossing threat

        self.step_count += 1
        
        obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
        return obs, reward, False, False, {"image": img_base64, "vlm_reason": getattr(self, 'last_vlm_reason', '')}

    def _get_obs(self, speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, traffic_light, in_junction, vehicles_in_junction):
        """Constructs the observation vector sent to the Stable Baselines3 model."""
        in_junction_encoded = 1.0 if in_junction else 0.0
        
        tl = traffic_light.lower() if isinstance(traffic_light, str) else "none"
        if tl == "red":
            traffic_light_encoded = 0.0
        elif tl == "yellow":
            traffic_light_encoded = 0.5
        elif tl == "green":
            traffic_light_encoded = 1.0
        else:
            traffic_light_encoded = -1.0
            
        return np.array([
            speed, 
            cte, 
            heading_error, 
            dist_to_wp, 
            angle_to_wp, 
            self.last_vlm_signal,
            radar_dist,
            traffic_light_encoded,
            in_junction_encoded,
            float(vehicles_in_junction)
        ], dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # VLM PROMPT BUILDER — Context-Aware, Right-of-Way Focused
    # ═══════════════════════════════════════════════════════════════════
    def _build_vlm_prompt(self):
        """Build a purely visual prompt to guide VLM reasoning (no ground truth injected)."""
        prompt = (
            "Analyze only the visible physical area immediately ahead of the ego vehicle.\n"
            "Ignore:\n"
            "- traffic lights\n"
            "- road signs\n"
            "- lane markings\n"
            "- parked vehicles outside the driving path\n"
            "- vehicles in opposing lanes\n"
            "- distant objects\n"
            "Determine only whether a physical obstacle is blocking or entering the ego vehicle's driving path.\n"
            "Respond ONLY with a valid JSON object.\n"
            "{\n"
            '  "action": "stop" | "slow_down" | "proceed",\n'
            '  "reason": "..."\n'
            "}"
        )
        return prompt

    # ═══════════════════════════════════════════════════════════════════
    # VLM JSON PARSER — Clean, reliable, with fallback
    # ═══════════════════════════════════════════════════════════════════
    def _parse_vlm_json(self, answer):
        """Parse structured JSON from VLM. Returns signal float (0.0, 0.5, 1.0)."""
        try:
            clean = answer.strip()
            if "```" in clean:
                parts = clean.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        clean = part[4:].strip()
                        break
                    elif part.startswith("{"):
                        clean = part
                        break

            parsed = json_lib.loads(clean)
            action = parsed.get("action", "").lower()
            reason = parsed.get("reason", "")
            
            if action == "stop": return 0.0, reason
            if action == "slow_down": return 0.5, reason
            if action == "proceed": return 1.0, reason
            
            return self._fallback_keyword_parse(answer)
            
        except json_lib.JSONDecodeError:
            return self._fallback_keyword_parse(answer)

    def _fallback_keyword_parse(self, text):
        """Two-step fallback: strictly detect and remove negations, then search for obstacles."""
        text_lower = text.lower()
        negated_phrases = [
            "no pedestrian", "no pedestrians", "no person", "no cyclist", "no cyclists",
            "no vehicle", "no car", "no obstacle", "no danger", "clear path",
            "no emergency vehicle", "not blocking"
        ]
        for phrase in negated_phrases:
            text_lower = text_lower.replace(phrase, "")
        danger_keywords = ["pedestrian", "person", "cyclist", "motorcycle", "car blocking", "emergency", "ambulance", "firetruck"]
        for kw in danger_keywords:
            if kw in text_lower:
                return 0.0, f"Fallback: detected {kw}"
        if "proceed" in text_lower or "safe" in text_lower or "clear" in text_lower:
            return 1.0, "Fallback Parsing"
        return self.last_vlm_signal, "Fallback Parsing"

    # ═══════════════════════════════════════════════════════════════════
    # MAIN VLM FEEDBACK FUNCTION
    # ═══════════════════════════════════════════════════════════════════
    def get_vlm_feedback(self, img_base64):
        """
        Sends the camera image to the local VLM purely for visual perception.
        Returns: (vlm_penalty, done_flag, signal_float)
        """
        prompt = self._build_vlm_prompt()
        try:
            payload = {
                "model": "qwen2.5vl:7b", 
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 40
            }
            print("\n[VLM] 📡 Sending request to Qwen2.5-VL...")
            response = requests.post("http://localhost:11434/v1/chat/completions", json=payload, timeout=20.0)
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"].strip()
                print(f"[VLM] 🧠 Raw Output: '{answer}'")
                if not answer or len(answer) < 5:
                    print("[VLM] ⚠️ Output too short or invalid. Defaulting to previous signal.")
                    return 0.0, False, self.last_vlm_signal
                raw_signal, reason = self._parse_vlm_json(answer)
                self.last_vlm_signal = raw_signal
                self.last_vlm_reason = reason
                return 0.0, False, raw_signal
            else:
                print(f"[VLM] ❌ HTTP Error {response.status_code}: {response.text}")
        except requests.exceptions.Timeout:
            print("[VLM] ❌ Timeout: The VLM took longer than 20 seconds to respond.")
        except requests.exceptions.ConnectionError:
            print("[VLM] ❌ Connection Error: Could not connect to the VLM server on port 11434.")
        except Exception as e:
            print(f"[VLM] ❌ Unexpected Exception: {e}") 
        return 0.0, False, self.last_vlm_signal
