import json as json_lib
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import zmq


class CarlaVLMEnv(gym.Env):
    """
    CARLA + SAC + Qwen2.5-VL Reinforcement Learning Environment (Phase 5).
    
    Key Features:
    1. VLM-integrated spatial reasoning with temporal latching.
    2. Softened VLM STOP logic on isolated detections to prevent abrupt stalls.
    3. Anti-parking linear reward decay with a 5-second hard cutoff.
    4. Enhanced cruise motivation and controlled intersection motion bonus.
    5. Calibrated safe-stop distance thresholds (4.5m radar / 4.0m junction boundary).
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Action space: [acceleration (-1.0 to 1.0), steering (-1.0 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: 10-dimensional feature vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )

        # ZeroMQ server connection to CARLA bridge
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000
        self.socket.connect("tcp://localhost:5555")
        print("✅ RL Environment Connected to CARLA Server via ZMQ!")

        # VLM client configuration
        self.http = requests.Session()
        self.vlm_url = "http://localhost:11434/v1/chat/completions"
        self.vlm_model = "qwen2.5vl:7b"

        # General tracking variables
        self.step_count = 0
        self.last_vlm_signal = 1.0
        self.last_vlm_reason = "Init"
        self.current_scenario_id = 1
        self.last_radar_distance = 30.0

        # Temporal latch and anti-jitter parameters
        self.vlm_stop_hold_steps = 5
        self.vlm_slow_hold_steps = 4
        self.vlm_hold_counter = 0
        self.vlm_hold_level = 1.0

        # Isolated STOP softening counter
        self.vlm_consecutive_stop_count = 0
        self.vlm_stop_confirmation_steps = 2

        # Anti-parking counter
        self.consecutive_stop_steps = 0

        # Junction contextual information
        self.current_junction_ctx = {
            "in_junction": False,
            "approaching_junction": False,
            "junction_distance": 999.0,
            "traffic_light": "none",
            "vehicles_in_junction": 0,
        }

    def _reconnect_zmq(self):
        """Safely reconnects ZeroMQ socket upon timeout."""
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = 30000
        self.socket.connect("tcp://localhost:5555")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_vlm_signal = 1.0
        self.last_vlm_reason = "Init"
        self.vlm_hold_counter = 0
        self.vlm_hold_level = 1.0
        self.vlm_consecutive_stop_count = 0
        self.consecutive_stop_steps = 0
        self.current_scenario_id = 1
        self.last_radar_distance = 30.0
        self.current_junction_ctx = {
            "in_junction": False,
            "approaching_junction": False,
            "junction_distance": 999.0,
            "traffic_light": "none",
            "vehicles_in_junction": 0,
        }

        self.socket.send_json({"command": "reset", "need_image": True})
        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            print("❌ ZMQ timeout during reset. Reconnecting...")
            self._reconnect_zmq()
            return np.zeros(10, dtype=np.float32), {}

        self.current_junction_ctx = {
            "in_junction": state.get("in_junction", False),
            "approaching_junction": state.get("approaching_junction", False),
            "junction_distance": state.get("junction_distance", 999.0),
            "traffic_light": state.get("traffic_light", "none"),
            "vehicles_in_junction": state.get("vehicles_in_junction", 0),
        }
        self.current_scenario_id = state.get("scenario_id", 1)
        radar_dist = float(state.get("radar_distance", 30.0))
        self.last_radar_distance = radar_dist

        obs = self._get_obs(
            speed=state.get("speed", 0.0),
            cte=state.get("cte", 0.0),
            heading_error=state.get("heading_error", 0.0),
            dist_to_wp=state.get("dist_to_wp", 0.0),
            angle_to_wp=state.get("angle_to_wp", 0.0),
            radar_dist=radar_dist,
            traffic_light=self.current_junction_ctx.get("traffic_light", "none"),
            in_junction=self.current_junction_ctx.get("in_junction", False),
            vehicles_in_junction=self.current_junction_ctx.get("vehicles_in_junction", 0),
        )
        return obs, {"image": state.get("image", ""), "vlm_reason": self.last_vlm_reason}

    def _get_vlm_query_stride(self, radar_dist):
        """Dynamically adjusts VLM query frequency based on proximity to hazards."""
        in_junction = self.current_junction_ctx.get("in_junction", False)
        approaching_junction = self.current_junction_ctx.get("approaching_junction", False)
        near_junction = in_junction or approaching_junction

        if radar_dist <= 8.0:
            return 1
        if near_junction:
            return 2
        if radar_dist <= 20.0:
            return 3
        return 5

    def _process_vlm_signal(self, raw_vlm_signal, reason):
        """Processes and filters raw VLM output to avoid false-positive stops."""
        if raw_vlm_signal == 1.0:
            self.vlm_consecutive_stop_count = 0
            if self.vlm_hold_counter > 0:
                self.vlm_hold_counter -= 1
            else:
                self.last_vlm_signal = 1.0
                self.vlm_hold_level = 1.0
                self.last_vlm_reason = reason
            return

        if raw_vlm_signal == 0.5:
            self.vlm_consecutive_stop_count = 0
            if self.vlm_hold_level == 0.0 and self.vlm_hold_counter > 0:
                self.vlm_hold_counter -= 1
                self.last_vlm_reason = reason
            else:
                self.last_vlm_signal = 0.5
                self.vlm_hold_counter = self.vlm_slow_hold_steps
                self.vlm_hold_level = 0.5
                self.last_vlm_reason = reason
            return

        if raw_vlm_signal == 0.0:
            self.vlm_consecutive_stop_count += 1
            radar_supports_stop = self.last_radar_distance <= 7.0
            hard_stop = (
                radar_supports_stop
                or (self.vlm_consecutive_stop_count >= self.vlm_stop_confirmation_steps)
            )
            if hard_stop:
                self.last_vlm_signal = 0.0
                self.vlm_hold_counter = self.vlm_stop_hold_steps
                self.vlm_hold_level = 0.0
                self.last_vlm_reason = reason
                if radar_supports_stop:
                    print(
                        f"[VLM] 🛑 STOP CONFIRMED by radar | "
                        f"radar={self.last_radar_distance:.1f}m | "
                        f"reason=\"{reason}\""
                    )
                else:
                    print(
                        f"[VLM] 🛑 STOP CONFIRMED by repeated VLM | "
                        f"count={self.vlm_consecutive_stop_count} | "
                        f"reason=\"{reason}\""
                    )
            else:
                self.last_vlm_signal = 0.5
                self.vlm_hold_counter = self.vlm_slow_hold_steps
                self.vlm_hold_level = 0.5
                self.last_vlm_reason = f"VLM STOP->CAUTION: {reason}"
                print(
                    f"[VLM] ⚠️ Isolated STOP -> CAUTION | "
                    f"radar={self.last_radar_distance:.1f}m | "
                    f"reason=\"{reason}\""
                )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        accel_raw = float(np.clip(action[0], -1.0, 1.0))
        steer_raw = float(np.clip(action[1], -1.0, 1.0))

        if accel_raw >= 0.0:
            throttle = accel_raw
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(accel_raw)

        vlm_query_stride = self._get_vlm_query_stride(self.last_radar_distance)
        need_image = (self.step_count % vlm_query_stride == 0)

        self.socket.send_json({
            "command": "step",
            "throttle": throttle,
            "steer": steer_raw,
            "brake": brake,
            "need_image": need_image,
        })

        try:
            state = self.socket.recv_json()
        except zmq.error.Again:
            print("❌ ZMQ timeout during step. Reconnecting...")
            self._reconnect_zmq()
            return np.zeros(10, dtype=np.float32), 0.0, True, False, {}

        speed = float(state.get("speed", 0.0))
        cte = float(state.get("cte", 0.0))
        heading_error = float(state.get("heading_error", 0.0))
        dist_to_wp = float(state.get("dist_to_wp", 0.0))
        angle_to_wp = float(state.get("angle_to_wp", 0.0))
        radar_dist = float(state.get("radar_distance", 30.0))
        self.last_radar_distance = radar_dist
        img_base64 = state.get("image", "")
        dist_to_target = float(state.get("dist_to_target", 999.0))
        collision = bool(state.get("collision", False))

        self.current_junction_ctx = {
            "in_junction": state.get("in_junction", False),
            "approaching_junction": state.get("approaching_junction", False),
            "junction_distance": state.get("junction_distance", 999.0),
            "traffic_light": state.get("traffic_light", "none"),
            "vehicles_in_junction": state.get("vehicles_in_junction", 0),
        }
        self.current_scenario_id = state.get("scenario_id", self.current_scenario_id)

        # Periodic VLM perception inference
        if need_image and img_base64:
            raw_vlm_signal, reason = self.get_vlm_feedback(
                img_base64=img_base64,
                default_signal=self.last_vlm_signal,
            )
            self._process_vlm_signal(raw_vlm_signal, reason)
        else:
            if self.vlm_hold_counter > 0:
                self.vlm_hold_counter -= 1
            elif self.last_vlm_signal < 1.0:
                self.last_vlm_signal = 1.0
                self.vlm_hold_level = 1.0
                self.last_vlm_reason = "Latch expired"
                self.vlm_consecutive_stop_count = 0

        tl_state = self.current_junction_ctx.get("traffic_light", "none")
        in_junction = self.current_junction_ctx.get("in_junction", False)
        v_count = int(self.current_junction_ctx.get("vehicles_in_junction", 0))

        # Episode termination checks
        if collision:
            print("💥 COLLISION DETECTED! Terminating episode.")
            obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
            return obs, -100.0, True, False, {"image": img_base64, "vlm_reason": self.last_vlm_reason, "vlm_signal": self.last_vlm_signal}

        if dist_to_target < 2.5:
            print("🎯 Target Coordinates Reached! Completed successfully.")
            obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
            return obs, 100.0, True, False, {"image": img_base64, "vlm_reason": self.last_vlm_reason, "vlm_signal": self.last_vlm_signal}

        if abs(cte) > 3.5:
            print("❌ Vehicle Off-Lane (CTE > 3.5m)! Terminating episode.")
            obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
            return obs, -50.0, True, False, {"image": img_base64, "vlm_reason": self.last_vlm_reason, "vlm_signal": self.last_vlm_signal}

        if self.step_count > 500:
            print("⏳ Maximum episode steps reached (TRUNCATED, not terminated).")
            obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
            # Truncation preserves bootstrap value targets in SAC
            return obs, 0.0, False, True, {"image": img_base64, "vlm_reason": self.last_vlm_reason, "vlm_signal": self.last_vlm_signal}

        # ─────────────────────────────────────────────────────────────
        # Hierarchical Reward Formulation
        # ─────────────────────────────────────────────────────────────
        is_radar_danger = (radar_dist <= 4.0)
        is_danger = (
            (self.last_vlm_signal < 1.0)
            or is_radar_danger
            or (tl_state in ["red", "yellow"])
        )

        if not is_danger:
            # NORMAL / SAFE MODE: Higher cruise motivation
            self.consecutive_stop_steps = 0
            speed_reward = min(speed, 20.0) * 0.22
            cte_penalty = abs(cte) * 0.5
            heading_penalty = (abs(heading_error) / 180.0) * 2.0
            reward = speed_reward - cte_penalty - heading_penalty
            if speed < 1.0:
                reward -= 5.0
        else:
            # DANGER MODE: Calibrated approach and junction yielding
            radar_allowed = float(np.clip(((radar_dist - 3.0) / 12.0) * 8.0, 0.0, 8.0))

            if self.last_vlm_signal == 0.0:
                junction_dist = self.current_junction_ctx.get("junction_distance", 999.0)
                # Calibrated thresholds: Approach within 4.5m radar and 4.0m junction boundary
                if radar_dist > 4.5 and junction_dist > 4.0:
                    vlm_allowed = 4.0
                else:
                    vlm_allowed = 0.0
            elif self.last_vlm_signal == 0.5:
                vlm_allowed = 5.0
            else:
                vlm_allowed = 8.0

            cautious_allowed_speed = min(radar_allowed, vlm_allowed)
            if tl_state == "red":
                cautious_allowed_speed = 0.0
            elif tl_state == "yellow":
                cautious_allowed_speed = min(cautious_allowed_speed, 4.0)

            # Danger reward calculation with linear anti-parking decay
            if cautious_allowed_speed == 0.0:
                if speed <= 0.5:
                    self.consecutive_stop_steps += 1
                    reward = max(0.0, 5.0 - 0.05 * self.consecutive_stop_steps)
                else:
                    self.consecutive_stop_steps = 0
                    reward = -2.0 - (1.5 * speed)
            else:
                self.consecutive_stop_steps = 0
                if speed <= cautious_allowed_speed:
                    reward = 1.0 + (speed * 0.1)
                    # Controlled motion bonus inside junction to prevent stalling
                    if in_junction:
                        reward += min(speed, 8.0) * 0.15
                else:
                    excess_speed = speed - cautious_allowed_speed
                    reward = -2.0 - (1.5 * excess_speed)

            reward -= abs(cte) * 0.5

        self.step_count += 1
        obs = self._get_obs(speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, tl_state, in_junction, v_count)
        info = {
            "image": img_base64,
            "vlm_reason": self.last_vlm_reason,
            "vlm_signal": self.last_vlm_signal,
            "vlm_hold_counter": self.vlm_hold_counter,
            "vlm_consecutive_stop_count": self.vlm_consecutive_stop_count,
            "consecutive_stop_steps": self.consecutive_stop_steps,
            "scenario_id": self.current_scenario_id,
        }
        return obs, reward, False, False, info

    def _get_obs(self, speed, cte, heading_error, dist_to_wp, angle_to_wp, radar_dist, traffic_light, in_junction, vehicles_in_junction):
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
            speed, cte, heading_error, dist_to_wp, angle_to_wp,
            self.last_vlm_signal, radar_dist, traffic_light_encoded,
            in_junction_encoded, float(vehicles_in_junction),
        ], dtype=np.float32)

    def _build_vlm_prompt(self):
        return (
            "You are a collision-risk detector for an autonomous vehicle.\n"
            "Inspect the asphalt driving path ahead. "
            "Ignore the vehicle hood at the bottom edge.\n"
            "Classify immediate hazards "
            "(vehicles, pedestrians, cyclists, obstacles):\n"
            '- "stop": Path is directly blocked or an obstacle is entering our lane.\n'
            '- "slow_down": An obstacle is near the lane edge or entering a crosswalk.\n'
            '- "proceed": Path ahead is completely clear.\n'
            "Ignore traffic lights and distant signs.\n"
            'Respond ONLY in valid JSON:\n'
            '{"action": "stop" | "slow_down" | "proceed", "reason": "3-5 words"}'
        )

    def _parse_vlm_json(self, answer, default_signal=1.0):
        if not answer:
            return default_signal, "Empty VLM output"
        clean = answer.strip()
        if "```" in clean:
            clean = clean.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

        try:
            parsed = json_lib.loads(clean)
            action = str(parsed.get("action", "")).strip().lower()
            reason = str(parsed.get("reason", "")).strip()
            if action == "stop":
                return 0.0, reason or "VLM: stop"
            if action in ["slow", "slow_down"]:
                return 0.5, reason or "VLM: slow_down"
            if action == "proceed":
                return 1.0, reason or "VLM: proceed"
        except Exception:
            pass

        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json_lib.loads(clean[start:end + 1])
                action = str(parsed.get("action", "")).strip().lower()
                reason = str(parsed.get("reason", "")).strip()
                if action == "stop":
                    return 0.0, reason or "VLM: stop"
                if action in ["slow", "slow_down"]:
                    return 0.5, reason or "VLM: slow_down"
                if action == "proceed":
                    return 1.0, reason or "VLM: proceed"
            except Exception:
                pass

        return self._fallback_keyword_parse(clean, default_signal)

    def _fallback_keyword_parse(self, text, default_signal=1.0):
        text_lower = str(text).lower()
        stop_patterns = ['"action":"stop"', '"action": "stop"', "'action':'stop'", "'action': 'stop'", "action: stop"]
        for pattern in stop_patterns:
            if pattern in text_lower:
                return 0.0, "Fallback: stop"

        slow_patterns = [
            '"action":"slow_down"', '"action": "slow_down"',
            '"action":"slow"', '"action": "slow"',
            "'action':'slow_down'", "'action': 'slow_down'",
            "'action':'slow'", "'action': 'slow'",
            "action: slow_down", "action: slow",
        ]
        for pattern in slow_patterns:
            if pattern in text_lower:
                return 0.5, "Fallback: slow_down"

        proceed_patterns = ['"action":"proceed"', '"action": "proceed"', "'action':'proceed'", "'action': 'proceed'", "action: proceed"]
        for pattern in proceed_patterns:
            if pattern in text_lower:
                return 1.0, "Fallback: proceed"

        return default_signal, "Fallback: preserved previous signal"

    def get_vlm_feedback(self, img_base64, default_signal=1.0):
        """Dispatches captured scene image to local Ollama Qwen2.5-VL endpoint."""
        if not img_base64:
            print(f"[VLM] ⚠️ No image received. Preserving signal={default_signal}")
            return default_signal, "No image"

        prompt = self._build_vlm_prompt()
        payload = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + img_base64},
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": 40,
            "keep_alive": -1,
        }

        start_time = time.perf_counter()
        print(
            f"[VLM] 🧠 Querying model={self.vlm_model} | "
            f"scenario={self.current_scenario_id} | "
            f"step={self.step_count} | "
            f"image_bytes≈{len(img_base64)} chars"
        )

        try:
            response = self.http.post(self.vlm_url, json=payload, timeout=5.0)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code != 200:
                print(f"[VLM] ❌ HTTP Error {response.status_code} | latency={latency_ms:.1f} ms | response={response.text[:300]}")
                return default_signal, "VLM HTTP error"

            try:
                response_data = response.json()
            except ValueError as e:
                print(f"[VLM] ❌ Invalid JSON response | latency={latency_ms:.1f} ms | error={e}")
                return default_signal, "VLM invalid API JSON"

            choices = response_data.get("choices", [])
            if not choices:
                print(f"[VLM] ❌ No choices returned | latency={latency_ms:.1f} ms")
                return default_signal, "VLM: no choices"

            message = choices[0].get("message", {})
            answer = str(message.get("content", "")).strip()
            if not answer:
                print(f"[VLM] ❌ Empty model output | latency={latency_ms:.1f} ms")
                return default_signal, "VLM: empty output"

            print(f"[VLM] 📥 Raw output: {answer}")
            raw_signal, reason = self._parse_vlm_json(answer, default_signal)

            action_name = "STOP" if raw_signal == 0.0 else ("SLOW_DOWN" if raw_signal == 0.5 else "PROCEED")
            print(f"[VLM] ✅ Result: action={action_name} | signal={raw_signal:.1f} | reason=\"{reason}\" | latency={latency_ms:.1f} ms")
            return raw_signal, reason

        except requests.Timeout:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[VLM] ⚠️ Inference timeout | latency={latency_ms:.1f} ms | preserving signal={default_signal}")
            return default_signal, "VLM timeout"
        except requests.RequestException as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[VLM] ❌ Request error | latency={latency_ms:.1f} ms | error={e}")
            return default_signal, "VLM request error"
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            print(f"[VLM] ❌ Unexpected error | latency={latency_ms:.1f} ms | error={e}")
            return default_signal, "VLM unexpected error"

    def close(self):
        try:
            self.socket.close(linger=0)
        except Exception:
            pass
        try:
            self.http.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass
        super().close()