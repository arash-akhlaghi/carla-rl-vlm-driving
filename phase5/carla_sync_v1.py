import carla
import zmq
import base64
import numpy as np
import cv2
import time
import math
import random

# ==========================================
# 🛠️ TESTING CONFIGURATION
# Set to None for real training (all scenarios)
FORCE_SCENARIO = None
# ==========================================

def noisy_dist(base, noise_range=3.0):
    """Add random noise to spawn distances to prevent RL overfitting on fixed positions."""
    return max(3.0, base + random.uniform(-noise_range, noise_range))

# ==========================================================================
# RIGHT-OF-WAY CROSS-TRAFFIC DIVERSITY
# ==========================================================================
CROSS_TRAFFIC_CLEAR_PROB = 0.0
CRITICAL_CROSS_DISTANCE_RANGE = (2.0, 5.0)
CRITICAL_CROSS_SPEED_RANGE = (45.0, 52.0)
CLEAR_CROSS_DISTANCE_RANGE = (15.0, 25.0)
CLEAR_CROSS_SPEED_RANGE = (20.0, 25.0)

def sample_cross_traffic_profile():
    """Sample one cross-traffic urgency profile for a training episode."""
    if random.random() < CROSS_TRAFFIC_CLEAR_PROB:
        distance = random.uniform(*CLEAR_CROSS_DISTANCE_RANGE)
        speed_kmh = random.uniform(*CLEAR_CROSS_SPEED_RANGE)
        return {
            "label": "CLEAR",
            "distance": float(distance),
            "speed_kmh": float(speed_kmh),
            "speed_diff": 0.0,
        }
    distance = random.uniform(*CRITICAL_CROSS_DISTANCE_RANGE)
    speed_kmh = random.uniform(*CRITICAL_CROSS_SPEED_RANGE)
    return {
        "label": "CRITICAL",
        "distance": float(distance),
        "speed_kmh": float(speed_kmh),
        "speed_diff": -35.0,
    }

def find_lead_wp_ahead(vehicle, route, min_ahead=6.0, max_ahead=11.0):
    """Pick a lead-convoy waypoint safely ahead along the route."""
    if not route or len(route) < 12:
        return None, 0.0

    closest_idx = 0
    if vehicle is not None:
        loc = vehicle.get_location()
        if abs(loc.x) > 0.1 or abs(loc.y) > 0.1:
            closest_idx = min(range(len(route)), key=lambda i: route[i].transform.location.distance(loc))

    target_offset = int(random.uniform(min_ahead, max_ahead))
    target_idx = closest_idx + target_offset

    if target_idx >= len(route):
        target_idx = len(route) - 2

    if target_idx <= closest_idx:
        return None, 0.0

    actual_dist = route[target_idx].transform.location.distance(route[closest_idx].transform.location)
    return route[target_idx], float(actual_dist)

# ==========================================================================
# 🧭 HELPER FUNCTIONS FOR CROSS-VEHICLE SPAWNING
# ==========================================================================
def find_cross_conflict_wp(ego_junction_wps, cross_start_wp, cross_end_wp):
    """Find the cross-road waypoint closest to the ego's junction trajectory."""
    if not ego_junction_wps:
        return cross_start_wp
    current_wp = cross_start_wp
    best_wp = cross_start_wp
    best_dist = float('inf')
    max_iter = int(cross_start_wp.transform.location.distance(cross_end_wp.transform.location)) + 3
    for _ in range(max_iter):
        d = min(current_wp.transform.location.distance(e_wp.transform.location) for e_wp in ego_junction_wps)
        if d < best_dist:
            best_dist = d
            best_wp = current_wp
        if current_wp.transform.location.distance(cross_end_wp.transform.location) < 1.5:
            break
        next_wps = current_wp.next(1.0)
        if not next_wps:
            break
        current_wp = min(next_wps, key=lambda w: w.transform.location.distance(cross_end_wp.transform.location))
    return best_wp

def spawn_cross_vehicle_at_distance(world, adv_bp, adv_start_wp, spawn_dist, z_offset=1.0):
    """Spawn a vehicle at a specific distance upstream of adv_start_wp."""
    prev_wps = adv_start_wp.previous(spawn_dist)
    if not prev_wps:
        return None, None
    spawn_wp = prev_wps[0]
    transform = spawn_wp.transform
    transform.location.z += z_offset
    adv = world.try_spawn_actor(adv_bp, transform)
    return adv, spawn_wp

def push_cross_vehicle_initial(adv, speed_kmh):
    """Give freshly spawned cross traffic immediate forward velocity."""
    fwd = adv.get_transform().get_forward_vector()
    speed_ms = max(0.0, float(speed_kmh)) / 3.6
    adv.set_target_velocity(carla.Vector3D(fwd.x * speed_ms, fwd.y * speed_ms, 0.0))

def set_forward_tm_path(tm, vehicle, spawn_wp, start_wp, end_wp):
    """Give Traffic Manager a strictly forward path from the actual spawn point."""
    try:
        next_after = end_wp.next(20.0)
        locations = [spawn_wp.transform.location, start_wp.transform.location, end_wp.transform.location]
        if next_after:
            locations.append(next_after[0].transform.location)
        if hasattr(tm, 'set_path'):
            tm.set_path(vehicle, locations)
            return True
    except Exception:
        pass
    return False

def random_weather():
    """Generate random weather/lighting conditions for domain randomization."""
    presets = [
        carla.WeatherParameters.ClearNoon,
        carla.WeatherParameters.CloudyNoon,
        carla.WeatherParameters.WetNoon,
        carla.WeatherParameters.SoftRainNoon,
        carla.WeatherParameters.HardRainNoon,
        carla.WeatherParameters.ClearSunset,
        carla.WeatherParameters.ClearNight,
        carla.WeatherParameters.WetNight,
        carla.WeatherParameters.SoftRainNight,
        carla.WeatherParameters.HardRainNight,
    ]
    if random.random() < 0.3:
        return carla.WeatherParameters(
            cloudiness=random.uniform(0, 100),
            precipitation=random.uniform(0, 100),
            precipitation_deposits=random.uniform(0, 100),
            wind_intensity=random.uniform(0, 100),
            sun_altitude_angle=random.uniform(-90, 90),
            fog_density=random.uniform(0, 50),
            fog_distance=random.uniform(0, 100),
            wetness=random.uniform(0, 100),
        )
    else:
        return random.choice(presets)

def normalize_angle(angle):
    """Normalizes an angle to the [-180, 180] degree range."""
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def draw_route(world, route, life_time=0.1):
    """Draws the generated route. Yellow for intersections, Green for straight roads."""
    for wp in route:
        loc = wp.transform.location
        loc.z += 0.5
        if wp.is_junction:
            dot_color = carla.Color(r=255, g=255, b=0)
        else:
            dot_color = carla.Color(r=0, g=255, b=0)
        world.debug.draw_point(loc, size=0.15, color=dot_color, life_time=life_time)

def draw_crosswalk(world, center_wp, lane_width=None, life_time=60.0):
    """Draw visible zebra crosswalk stripes on the road at the given waypoint."""
    if lane_width is None:
        lane_width = center_wp.lane_width if center_wp.lane_width > 0 else 3.5
    fwd = center_wp.transform.get_forward_vector()
    right = center_wp.transform.get_right_vector()
    center_loc = center_wp.transform.location
    half_width = lane_width / 2.0 + 1.0
    num_stripes = 8
    stripe_spacing = 0.6
    stripe_thickness = 0.12
    white = carla.Color(r=255, g=255, b=255)
    for i in range(num_stripes):
        offset = (i - num_stripes / 2.0 + 0.5) * stripe_spacing
        stripe_center_x = center_loc.x + fwd.x * offset
        stripe_center_y = center_loc.y + fwd.y * offset
        stripe_center_z = center_loc.z + 0.03
        start = carla.Location(x=stripe_center_x - right.x * half_width, y=stripe_center_y - right.y * half_width, z=stripe_center_z)
        end = carla.Location(x=stripe_center_x + right.x * half_width, y=stripe_center_y + right.y * half_width, z=stripe_center_z)
        world.debug.draw_line(start, end, thickness=stripe_thickness, color=white, life_time=life_time)
    border_offset = (num_stripes / 2.0) * stripe_spacing
    for sign in [-1, 1]:
        bx = center_loc.x + fwd.x * sign * border_offset
        by = center_loc.y + fwd.y * sign * border_offset
        bz = center_loc.z + 0.03
        b_start = carla.Location(x=bx - right.x * half_width, y=by - right.y * half_width, z=bz)
        b_end = carla.Location(x=bx + right.x * half_width, y=by + right.y * half_width, z=bz)
        world.debug.draw_line(b_start, b_end, thickness=0.15, color=carla.Color(r=255, g=200, b=0), life_time=life_time)
    return center_loc, right, fwd, half_width

def generate_safe_route(start_waypoint, target_distance=250.0, step_size=1.0, max_junctions=1, post_junction_runway=0.0):
    """Generates a route that covers max_junctions junction(s) + post_junction_runway meters."""
    route = [start_waypoint]
    current_wp = start_waypoint
    accumulated_dist = 0.0
    junction_count = 0
    was_in_junction = False
    post_junction_dist = 0.0
    while accumulated_dist < target_distance:
        next_wps = current_wp.next(step_size)
        if not next_wps: break
        valid_wps = [wp for wp in next_wps if wp.lane_type == carla.LaneType.Driving]
        if not valid_wps: break
        if len(valid_wps) == 1:
            current_wp = valid_wps[0]
        else:
            current_yaw = current_wp.transform.rotation.yaw
            current_wp = min(valid_wps, key=lambda w: abs(normalize_angle(w.transform.rotation.yaw - current_yaw)))
        route.append(current_wp)
        accumulated_dist += step_size
        if current_wp.is_junction:
            was_in_junction = True
        elif was_in_junction:
            was_in_junction = False
            junction_count += 1
            if junction_count >= max_junctions and post_junction_runway <= 0.0:
                route.pop()
                break
        if junction_count >= max_junctions:
            if current_wp.is_junction:
                break
            post_junction_dist += step_size
            if post_junction_dist >= post_junction_runway:
                break
    return route

def get_route_errors(vehicle, route):
    """Calculates Cross-Track Error (CTE) and Heading Error for the RL Agent."""
    if not route or len(route) < 2:
        return 5.0, 0.0, 0.0, 0.0, 999.0
    vehicle_loc = vehicle.get_location()
    vehicle_yaw = vehicle.get_transform().rotation.yaw
    min_dist = float('inf')
    closest_idx = 0
    for i, wp in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    target_idx = min(closest_idx + 2, len(route) - 1)
    target_wp = route[target_idx]
    target_loc = target_wp.transform.location
    target_yaw = target_wp.transform.rotation.yaw
    dx = target_loc.x - vehicle_loc.x
    dy = target_loc.y - vehicle_loc.y
    dist_to_wp = math.sqrt(dx**2 + dy**2)
    target_vector_yaw = math.degrees(math.atan2(dy, dx))
    angle_to_wp = normalize_angle(target_vector_yaw - vehicle_yaw)
    heading_error = normalize_angle(target_yaw - vehicle_yaw)
    cte = dist_to_wp * math.sin(math.radians(angle_to_wp))
    final_destination = route[-1].transform.location
    dx_target = final_destination.x - vehicle_loc.x
    dy_target = final_destination.y - vehicle_loc.y
    dist_to_target = math.sqrt(dx_target**2 + dy_target**2)
    return cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target

def _tl_state_to_str(state):
    if state == carla.TrafficLightState.Red:
        return "red"
    elif state == carla.TrafficLightState.Yellow:
        return "yellow"
    elif state == carla.TrafficLightState.Green:
        return "green"
    return "none"

def get_junction_context(vehicle, world, route):
    """Extract traffic light state, junction proximity, and nearby vehicles for right-of-way reasoning."""
    map_ = world.get_map()
    ego_wp = map_.get_waypoint(vehicle.get_location())
    in_junction = ego_wp.is_junction
    closest_idx = 0
    min_dist = float('inf')
    vehicle_loc = vehicle.get_location()
    for i, wp in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    future_route = route[closest_idx : closest_idx + 50]
    approaching_junction = False
    junction_dist = 999.0
    for wp in future_route:
        if wp.is_junction:
            approaching_junction = True
            junction_dist = wp.transform.location.distance(vehicle_loc)
            break
    tl_state = "none"
    if vehicle.is_at_traffic_light():
        tl = vehicle.get_traffic_light()
        if tl:
            tl_state = _tl_state_to_str(tl.get_state())
    if tl_state == "none" and (approaching_junction or in_junction):
        ego_loc = vehicle.get_location()
        ego_fwd = vehicle.get_transform().get_forward_vector()
        all_tls = world.get_actors().filter('traffic.traffic_light')
        best_tl = None
        best_score = float('inf')
        for tl_actor in all_tls:
            tl_loc = tl_actor.get_location()
            dist = tl_loc.distance(ego_loc)
            if dist > 20.0 or dist < 1.0:
                continue
            dx = tl_loc.x - ego_loc.x
            dy = tl_loc.y - ego_loc.y
            forward_proj = ego_fwd.x * dx + ego_fwd.y * dy
            if forward_proj <= 0:
                continue
            is_on_route = False
            for stop_wp in tl_actor.get_stop_waypoints():
                for wp in future_route:
                    if stop_wp.road_id == wp.road_id and stop_wp.lane_id == wp.lane_id:
                        if stop_wp.transform.location.distance(wp.transform.location) < 4.0:
                            is_on_route = True
                            break
                if is_on_route:
                    break
            if is_on_route and dist < best_score:
                best_score = dist
                best_tl = tl_actor
        if best_tl:
            tl_state = _tl_state_to_str(best_tl.get_state())
    vehicles_in_junction = 0
    if in_junction or approaching_junction:
        all_vehicles = world.get_actors().filter('vehicle.*')
        for v in all_vehicles:
            if v.id == vehicle.id:
                continue
            v_wp = map_.get_waypoint(v.get_location())
            if v_wp.is_junction:
                dist = v.get_location().distance(vehicle.get_location())
                if dist < 25.0:
                    vehicles_in_junction += 1
    return {
        "in_junction": in_junction,
        "approaching_junction": approaching_junction,
        "junction_distance": round(junction_dist, 1),
        "traffic_light": tl_state,
        "vehicles_in_junction": vehicles_in_junction
    }

# ==========================================================================
# 🛡️ NPC SAFETY SHIELD — last-resort collision prevention
# ==========================================================================
# DESIGN PHILOSOPHY:
#   * LAST-RESORT safety net to keep training episodes clean.
#   * NOT a right-of-way / yield mechanism.
#   * NPCs still cross the conflict point BEFORE Ego whenever possible.
#   * Ego -> NPC collisions (ego's fault): NOT prevented.
#   * NPC -> Ego collisions from the blind sides (REAR / LEFT / RIGHT):
#     prevented, because ego's sensors cannot see them.
#   * NPC <-> NPC collisions: prevented from ANY direction.
#
# SCOPE:
#   * Active only for scenarios 1-5 (main call site gates by scenario_id).
#   * Scenarios 6, 7, 8 are intentionally NOT shielded:
#       - 6/7 place an obstacle directly on Ego's path; perception + RL must
#         handle it.
#       - 8 is a multi-threat scenario (pedestrian + vehicle); the vehicle
#         arrives after the pedestrian triggers, so the challenge must remain.
# ==========================================================================

def _actor_speed_kmh(actor):
    """Return actor speed in km/h."""
    if actor is None or not actor.is_alive:
        return 0.0
    v = actor.get_velocity()
    return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _set_npc_target_speed(npc, target_kmh, shield_meta):
    """Set an NPC's target speed via Traffic Manager without disabling autopilot."""
    if npc is None or not npc.is_alive:
        return
    speed_limit_kmh = 50.0
    diff = (speed_limit_kmh - max(0.0, float(target_kmh))) / speed_limit_kmh * 100.0
    diff = float(np.clip(diff, -80.0, 100.0))
    tm.vehicle_percentage_speed_difference(npc, diff)
    if shield_meta is not None:
        shield_meta["active"] = True


def _restore_npc_speed(npc, shield_meta):
    """Restore an NPC's original TM speed difference after the shield releases."""
    if npc is None or not npc.is_alive:
        return
    if shield_meta is None or not shield_meta.get("active", False):
        return
    orig_diff = shield_meta.get("orig_diff", 0.0)
    tm.vehicle_percentage_speed_difference(npc, orig_diff)
    shield_meta["active"] = False


def _closest_approach_metrics(a, b):
    """
    Return (tcpa, cpa_distance, closing_speed).

    TCPA is the time until closest point of approach; CPA is the minimum
    distance between the two actors assuming they keep their current
    velocities. A small CPA indicates a genuine collision course, even if
    the current straight-line distance is still large.

    Handles the perpendicular T-bone case that simple distance-based rules
    cannot detect.
    """
    a_loc = a.get_location()
    b_loc = b.get_location()
    a_v = a.get_velocity()
    b_v = b.get_velocity()

    dx = b_loc.x - a_loc.x
    dy = b_loc.y - a_loc.y
    dvx = b_v.x - a_v.x
    dvy = b_v.y - a_v.y

    dv_dot_dv = dvx * dvx + dvy * dvy
    if dv_dot_dv < 0.05:
        dist = math.sqrt(dx * dx + dy * dy)
        return 999.0, dist, 0.0

    t_cpa = -(dx * dvx + dy * dvy) / dv_dot_dv
    if t_cpa < 0.0:
        dist = math.sqrt(dx * dx + dy * dy)
        return 0.0, dist, 0.0
    if t_cpa > 6.0:
        return t_cpa, 999.0, 0.0

    cpa_x = dx + dvx * t_cpa
    cpa_y = dy + dvy * t_cpa
    cpa_dist = math.sqrt(cpa_x * cpa_x + cpa_y * cpa_y)

    dist_now = math.sqrt(dx * dx + dy * dy)
    if dist_now < 0.05:
        closing = 0.0
    else:
        ux, uy = dx / dist_now, dy / dist_now
        closing = -(dvx * ux + dvy * uy)

    return t_cpa, cpa_dist, closing


def _npc_ego_zone(ego, npc):
    """Return the position of NPC in ego's local frame: FRONT / REAR / LEFT / RIGHT."""
    ego_loc = ego.get_location()
    ego_fwd = ego.get_transform().get_forward_vector()
    ego_right = ego.get_transform().get_right_vector()
    npc_loc = npc.get_location()

    dx = npc_loc.x - ego_loc.x
    dy = npc_loc.y - ego_loc.y

    long_proj = ego_fwd.x * dx + ego_fwd.y * dy
    lat_proj = ego_right.x * dx + ego_right.y * dy

    if abs(long_proj) >= abs(lat_proj):
        return "FRONT" if long_proj > 0 else "REAR"
    return "RIGHT" if lat_proj > 0 else "LEFT"


def npc_safety_shield(ego_vehicle, npcs, shield_meta):
    """
    Unified, minimal-intervention safety net for cross-traffic NPCs.

    Runs BEFORE world.tick(), so speed adjustments take effect on this tick.
    Uses TM speed control (autopilot stays ON), so both steering and the
    built-in TM collision checks remain functional.

    Rules (in order of priority):
      0. EMERGENCY BRAKE: for any side/rear NPC, if TCPA < 1.2 s AND CPA < 1.5 m,
         force target speed to 0. At that point the NPC cannot clear the
         intersection and the only safe action is to let Ego pass.
      1. NPC <-> NPC: any direction, TCPA < 1.5 s and CPA < 4 m -> slow the
         faster one.
      2. NPC -> Ego from LEFT/RIGHT: TCPA < 2.0 s and CPA < 4 m -> speed the
         NPC up so it clears the intersection faster. If already at max
         speed, slow it down as a fallback.
      3. NPC -> Ego from REAR: TCPA < 2.0 s and CPA < 3 m -> slow the NPC
         down to avoid a rear-end / merge collision.
      4. NPC -> Ego from FRONT: never touched (ego's responsibility).
    """
    if not npcs:
        return

    live = [n for n in npcs if n is not None and n.is_alive]

    # --------------------------------------------------------------
    # 0. EMERGENCY BRAKE — must run first, overrides everything else.
    # --------------------------------------------------------------
    for npc in live:
        meta = shield_meta.get(npc.id)
        if meta is None:
            continue

        zone = _npc_ego_zone(ego_vehicle, npc)
        if zone == "FRONT":
            continue

        tcpa, cpa_dist, closing = _closest_approach_metrics(npc, ego_vehicle)
        if tcpa < 1.2 and cpa_dist < 1.5 and closing >= 0.3:
            _set_npc_target_speed(npc, 0.0, meta)
            print(
                f"   🛑 NPC emergency brake (id={npc.id}, {zone}): "
                f"TCPA={tcpa:.2f}s CPA={cpa_dist:.2f}m → target=0"
            )

    # --------------------------------------------------------------
    # 1. NPC <-> NPC (any direction)
    # --------------------------------------------------------------
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a = live[i]
            b = live[j]

            meta_a = shield_meta.get(a.id)
            meta_b = shield_meta.get(b.id)

            tcpa, cpa_dist, closing = _closest_approach_metrics(a, b)
            if tcpa > 1.5 or cpa_dist > 4.0 or closing < 0.5:
                continue

            a_spd = _actor_speed_kmh(a)
            b_spd = _actor_speed_kmh(b)

            if a_spd >= b_spd and meta_a is not None:
                _set_npc_target_speed(a, max(0.0, a_spd * 0.30), meta_a)
            elif meta_b is not None:
                _set_npc_target_speed(b, max(0.0, b_spd * 0.30), meta_b)

            print(
                f"   🛡️ NPC↔NPC shield: id={a.id} vs id={b.id} "
                f"| TCPA={tcpa:.2f}s CPA={cpa_dist:.2f}m closing={closing*3.6:.1f}km/h"
            )

    # --------------------------------------------------------------
    # 2-4. NPC -> Ego (blind sides only)
    # --------------------------------------------------------------
    for npc in live:
        meta = shield_meta.get(npc.id)
        if meta is None:
            continue

        zone = _npc_ego_zone(ego_vehicle, npc)

        if zone == "FRONT":
            continue

        tcpa, cpa_dist, closing = _closest_approach_metrics(npc, ego_vehicle)
        if tcpa > 2.0 or cpa_dist > 4.0 or closing < 0.5:
            continue

        if tcpa < 1.2 and cpa_dist < 1.5:
            continue

        npc_spd = _actor_speed_kmh(npc)

        if zone == "REAR":
            if cpa_dist < 1.0 or tcpa < 0.5:
                target = 0.0
            else:
                target = max(0.0, npc_spd * 0.30)
            _set_npc_target_speed(npc, target, meta)
            print(
                f"   🛡️ NPC→Ego REAR shield: id={npc.id} "
                f"| TCPA={tcpa:.2f}s CPA={cpa_dist:.2f}m target={target:.0f}km/h"
            )
        else:
            if npc_spd < 55.0:
                target = min(80.0, npc_spd + 30.0)
            else:
                target = max(0.0, npc_spd * 0.50)
            _set_npc_target_speed(npc, target, meta)
            print(
                f"   🛡️ NPC→Ego {zone} shield: id={npc.id} "
                f"| TCPA={tcpa:.2f}s CPA={cpa_dist:.2f}m target={target:.0f}km/h"
            )

    # --------------------------------------------------------------
    # Release: if an NPC is no longer a threat, restore its original speed.
    # --------------------------------------------------------------
    for npc in live:
        meta = shield_meta.get(npc.id)
        if meta is None or not meta.get("active", False):
            continue

        on_course = False

        for other in live:
            if other.id == npc.id:
                continue
            tcpa, cpa_dist, closing = _closest_approach_metrics(npc, other)
            if tcpa <= 1.5 and cpa_dist <= 4.0 and closing >= 0.5:
                on_course = True
                break

        if not on_course:
            zone = _npc_ego_zone(ego_vehicle, npc)
            if zone != "FRONT":
                tcpa, cpa_dist, closing = _closest_approach_metrics(npc, ego_vehicle)
                if tcpa <= 2.0 and cpa_dist <= 4.0 and closing >= 0.5:
                    on_course = True

        if not on_course:
            _restore_npc_speed(npc, meta)


# ==========================================================================
# 🎯 CROSS-VEHICLE SPAWNING WITH FIXED SPEED
# ==========================================================================
def spawn_cross_vehicle_fixed(world, adv_bp, adv_start_wp, adv_end_wp,
                              ego_vehicle, ego_junction_wps,
                              arrival_offset_seconds,
                              spawn_dist_range=(8.0, 15.0),
                              min_speed_kmh=50.0, max_speed_kmh=80.0,
                              z_offset=1.0):
    """
    Spawn cross-traffic vehicle with a speed calibrated to the ego conflict ETA.

    Velocity handling:
      * The desired arrival time floor is 2.0 s.
      * After spawning with autopilot ON and the correct TM speed difference,
        a short manual-velocity boost (~0.3 s / 6 ticks) is applied so the
        vehicle actually reaches its cruise speed before TM takes over.
        TM does NOT provide instant acceleration, so without this step a
        "54 km/h" target NPC was observed crawling at 27 km/h and arriving
        late at the conflict point.
    """
    conflict_wp = find_cross_conflict_wp(ego_junction_wps, adv_start_wp, adv_end_wp)

    spawn_dist = random.uniform(*spawn_dist_range)
    for dist in [spawn_dist, spawn_dist * 0.7, spawn_dist * 0.5, 8.0, 5.0]:
        prev_wps = adv_start_wp.previous(dist)
        if not prev_wps:
            continue
        spawn_wp = prev_wps[0]

        if spawn_wp.lane_type != carla.LaneType.Driving:
            continue

        vec_to_start = adv_start_wp.transform.location - spawn_wp.transform.location
        fwd = spawn_wp.transform.get_forward_vector()
        dot = fwd.x * vec_to_start.x + fwd.y * vec_to_start.y + fwd.z * vec_to_start.z
        if dot <= 0:
            continue

        transform = spawn_wp.transform
        transform.location.z += z_offset
        adv = world.try_spawn_actor(adv_bp, transform)
        if adv is not None:
            ego_loc = ego_vehicle.get_location()
            ego_dist = ego_loc.distance(conflict_wp.transform.location)
            ego_v = ego_vehicle.get_velocity()
            ego_speed_kmh = 3.6 * math.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2)
            if ego_speed_kmh < 5.0:
                ego_speed_kmh = 15.0
            ego_eta = ego_dist / (ego_speed_kmh / 3.6)

            desired_arrival_s = max(2.0, ego_eta + arrival_offset_seconds)
            start_to_conflict = adv_start_wp.transform.location.distance(conflict_wp.transform.location)
            total_adv_dist = dist + start_to_conflict
            required_speed_kmh = (total_adv_dist / desired_arrival_s) * 3.6
            required_speed_kmh = float(np.clip(required_speed_kmh, min_speed_kmh, max_speed_kmh))

            speed_limit_kmh = 50.0
            diff_percent = (speed_limit_kmh - required_speed_kmh) / speed_limit_kmh * 100.0
            diff_percent = float(np.clip(diff_percent, -80.0, 80.0))

            # ----------------------------------------------------------
            # Phase 1: manual velocity boost so the NPC actually reaches
            # cruise speed before TM takes over. Without this, TM ramps up
            # from 0 m/s and the NPC arrives at the conflict point 1-2 s
            # late, causing late T-bone collisions in Scenario 1.
            # ----------------------------------------------------------
            adv.set_autopilot(False)
            for _ in range(6):  # ~0.30 s at fixed_delta_seconds=0.05
                push_cross_vehicle_initial(adv, required_speed_kmh)
                world.tick()

            # ----------------------------------------------------------
            # Phase 2: hand control back to TM with the correct target.
            # ----------------------------------------------------------
            adv.set_autopilot(True, tm.get_port())
            tm.ignore_lights_percentage(adv, 100)
            tm.ignore_signs_percentage(adv, 100)
            tm.ignore_vehicles_percentage(adv, 0)
            tm.vehicle_percentage_speed_difference(adv, diff_percent)
            set_forward_tm_path(tm, adv, spawn_wp, adv_start_wp, adv_end_wp)

            # One extra settle tick so TM picks up the correct state.
            world.tick()

            return adv, spawn_wp, conflict_wp, diff_percent

    return None, None, None, 0.0


def get_crossing_threat(vehicle, route, crossing_vehicles, max_ego_distance=22.0, max_route_gap=6.0):
    """
    Return whether a live crossing vehicle is genuinely approaching/crossing ego's route.
    """
    if vehicle is None or not route or not crossing_vehicles:
        return False, 999.0
    ego_loc = vehicle.get_location()
    ego_fwd = vehicle.get_transform().get_forward_vector()
    closest_idx = min(range(len(route)), key=lambda i: route[i].transform.location.distance(ego_loc))
    future_route = route[max(0, closest_idx - 2): closest_idx + 55]
    best_threat_dist = 999.0
    for adv in crossing_vehicles:
        try:
            if adv is None or not adv.is_alive:
                continue
            adv_loc = adv.get_location()
            adv_fwd = adv.get_transform().get_forward_vector()
            dx = adv_loc.x - ego_loc.x
            dy = adv_loc.y - ego_loc.y
            ego_longitudinal = ego_fwd.x * dx + ego_fwd.y * dy
            if ego_longitudinal < -3.0:
                continue
            dot = max(-1.0, min(1.0, ego_fwd.x * adv_fwd.x + ego_fwd.y * adv_fwd.y))
            angle = math.degrees(math.acos(dot))
            if not (45.0 <= angle <= 135.0):
                continue
            ego_to_adv = ego_loc.distance(adv_loc)
            if ego_to_adv > max_ego_distance:
                continue
            route_gap = min(math.hypot(adv_loc.x - wp.transform.location.x, adv_loc.y - wp.transform.location.y) for wp in future_route)
            if route_gap <= max_route_gap:
                best_threat_dist = min(best_threat_dist, ego_to_adv)
        except Exception:
            continue
    return best_threat_dist < 999.0, float(best_threat_dist)


def main():
    global tm
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("🧹 Cleaning up leftover actors from previous sessions...")
    for actor in world.get_actors().filter('vehicle.*'):
        actor.destroy()
    for actor in world.get_actors().filter('walker.*'):
        actor.destroy()
    for actor in world.get_actors().filter('controller.ai.walker'):
        actor.destroy()

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    adv_bps = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
    emergency_bps = [bp for bp in adv_bps if 'ambulance' in bp.id.lower() or 'firetruck' in bp.id.lower()]
    if not emergency_bps: emergency_bps = adv_bps
    walker_bps = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    cyclist_bps = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 2]
    if not cyclist_bps: cyclist_bps = adv_bps

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    vehicle = None
    camera = None
    radar = None
    collision_sensor = None
    collision_flag = False
    current_image = None
    current_image_raw = None
    local_step_count = 0
    current_radar_distance = 30.0
    current_route = []
    adversary_vehicles = []
    crossing_adversary_vehicles = []
    # Per-NPC shield metadata: {"orig_diff": float, "active": bool}
    npc_shield_meta = {}
    walker_list = []
    moving_cyclists = []
    current_scenario_id = 1
    test_scenario_counter = FORCE_SCENARIO if FORCE_SCENARIO else 1

    print("🗺️ Parsing map topology to find safe junction entrances...")
    topology = world.get_map().get_topology()
    junction_entries = []
    for segment in topology:
        end_wp = segment[1]
        if end_wp.lane_type == carla.LaneType.Driving:
            next_wps = end_wp.next(1.0)
            if next_wps and next_wps[0].is_junction:
                junction_entries.append(end_wp)
    junction_entries = list({(wp.transform.location.x, wp.transform.location.y): wp for wp in junction_entries}.values())
    print(f"✅ Found {len(junction_entries)} valid junction entrances!")

    def camera_callback(image):
        nonlocal current_image_raw
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        current_image_raw = array[:, :, :3].copy()

    def clear_adversaries():
        nonlocal adversary_vehicles, walker_list
        for adv in adversary_vehicles:
            try:
                if adv is not None and adv.is_alive:
                    adv.destroy()
            except Exception:
                pass
        adversary_vehicles.clear()
        for item in walker_list:
            ctrl = item[0]
            walker = item[1]
            try:
                if ctrl is not None and ctrl.is_alive:
                    ctrl.stop()
                    ctrl.destroy()
            except Exception:
                pass
            try:
                if walker is not None and walker.is_alive:
                    walker.destroy()
            except Exception:
                pass
        walker_list.clear()

    def reset_environment():
        nonlocal vehicle, camera, radar, collision_sensor, collision_flag, current_image, current_image_raw, current_radar_distance, current_route, adversary_vehicles, crossing_adversary_vehicles, test_scenario_counter, current_scenario_id
        if collision_sensor: collision_sensor.destroy(); collision_sensor = None
        if camera: camera.destroy(); camera = None
        if radar: radar.destroy(); radar = None
        if vehicle: vehicle.destroy(); vehicle = None
        for tl in world.get_actors().filter('traffic.traffic_light'):
            tl.freeze(False)
        clear_adversaries()
        crossing_adversary_vehicles.clear()
        npc_shield_meta.clear()
        moving_cyclists.clear()
        current_radar_distance = 30.0
        collision_flag = False

        weather = random_weather()
        world.set_weather(weather)
        print(f"🌦️ Weather set: sun_alt={weather.sun_altitude_angle:.0f}°, rain={weather.precipitation:.0f}%, fog={weather.fog_density:.0f}%")

        spawn_wp = None
        max_attempts = 40
        target_crosswalk_wp = None
        for attempt in range(max_attempts):
            spawn_wp = None
            if test_scenario_counter in [6, 8]:
                crosswalks = world.get_map().get_crosswalks()
                if crosswalks:
                    for _ in range(10):
                        cw = random.choice(crosswalks)
                        cw_wp = world.get_map().get_waypoint(cw, project_to_road=True, lane_type=carla.LaneType.Driving)
                        if cw_wp:
                            prev_wps = cw_wp.previous(15.0)
                            if prev_wps:
                                spawn_wp = prev_wps[0]
                                target_crosswalk_wp = cw_wp
                                break
            if spawn_wp is None:
                if junction_entries:
                    candidate_wp = random.choice(junction_entries)
                    ego_pre_junction_dist = 10.0 if test_scenario_counter == 3 else 5.0
                    prev_wps = candidate_wp.previous(ego_pre_junction_dist)
                    spawn_wp = prev_wps[0] if prev_wps else candidate_wp
                else:
                    spawn_points = world.get_map().get_spawn_points()
                    spawn_wp = world.get_map().get_waypoint(random.choice(spawn_points).location)
            test_route = generate_safe_route(spawn_wp, target_distance=250.0, step_size=1.0, max_junctions=1, post_junction_runway=0.0)
            if len(test_route) > 20:
                if test_scenario_counter in [1, 2, 3, 4, 5, 8]:
                    ego_j_wps = [wp for wp in test_route if wp.is_junction]
                    ego_j = ego_j_wps[0].get_junction() if ego_j_wps else None
                    temp_cross_paths = []
                    if ego_j and ego_j_wps:
                        ego_start_loc = ego_j_wps[0].transform.location
                        j_paths = ego_j.get_waypoints(carla.LaneType.Driving)
                        for path in j_paths:
                            start_wp, end_wp = path[0], path[1]
                            if start_wp.transform.location.distance(ego_start_loc) > 5.0:
                                dist = start_wp.transform.location.distance(end_wp.transform.location)
                                if dist < 2.0: continue
                                conflict = False
                                current_wp = start_wp
                                for _ in range(int(dist)):
                                    next_wps = current_wp.next(1.0)
                                    if not next_wps: break
                                    current_wp = min(next_wps, key=lambda w: w.transform.location.distance(end_wp.transform.location))
                                    for e_wp in ego_j_wps:
                                        if current_wp.transform.location.distance(e_wp.transform.location) < 3.5:
                                            conflict = True
                                            break
                                    if conflict: break
                                if conflict:
                                    temp_cross_paths.append((start_wp, end_wp))
                    if attempt < max_attempts - 5:
                        if not temp_cross_paths:
                            continue
                        if test_scenario_counter == 2 and len(temp_cross_paths) < 2:
                            continue
                final_transform = spawn_wp.transform
                final_transform.location.z += 1.0
                vehicle = world.try_spawn_actor(vehicle_bp, final_transform)
                if vehicle is not None:
                    current_route = test_route
                    junction_wps = sum(1 for wp in test_route if wp.is_junction)
                    print(f"📏 Route: {len(test_route)}m total | {junction_wps}m in junction(s)")
                    break
        if vehicle is None:
            raise RuntimeError("Failed to spawn Ego vehicle.")
        else:
            print("   🚗 Ego start position: ~5.0m before junction (10.0m for scenario 3) for non-crosswalk scenarios.")

        current_scenario_id = test_scenario_counter
        scenario_id = test_scenario_counter
        print(f"\n====================================================")
        print(f"🎬 BUILDING SCENARIO {scenario_id}")
        ego_junction_wps = [wp for wp in current_route if wp.is_junction]
        ego_junction = ego_junction_wps[0].get_junction() if ego_junction_wps else None
        valid_cross_paths = []
        if ego_junction and ego_junction_wps:
            ego_start_loc = ego_junction_wps[0].transform.location
            j_paths = ego_junction.get_waypoints(carla.LaneType.Driving)
            for path in j_paths:
                start_wp, end_wp = path[0], path[1]
                if start_wp.transform.location.distance(ego_start_loc) > 5.0:
                    dist = start_wp.transform.location.distance(end_wp.transform.location)
                    if dist < 2.0: continue
                    conflict = False
                    current_wp = start_wp
                    for _ in range(int(dist)):
                        next_wps = current_wp.next(1.0)
                        if not next_wps: break
                        current_wp = min(next_wps, key=lambda w: w.transform.location.distance(end_wp.transform.location))
                        for e_wp in ego_junction_wps:
                            if current_wp.transform.location.distance(e_wp.transform.location) < 3.5:
                                conflict = True
                                break
                        if conflict: break
                    if conflict:
                        valid_cross_paths.append((start_wp, end_wp))

        # ================= SCENARIO BUILDING =================
        if scenario_id == 1:
            print("🚑 Target: Emergency Vehicle Ambush (Fixed Speed)")
            print("   🛡️ NPC safety shield handles blind-side NPC→Ego and any NPC↔NPC collision course.")
            print("   🚦 NPCs still cross BEFORE Ego on their original schedule — no forced yielding.")
            if valid_cross_paths:
                chosen_path = random.choice(valid_cross_paths)
                adv_start_wp, adv_end_wp = chosen_path
                arrival_offset = random.uniform(-6.0, -4.0)
                adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                    world, random.choice(emergency_bps), adv_start_wp, adv_end_wp,
                    vehicle, ego_junction_wps, arrival_offset
                )
                if adv:
                    adversary_vehicles.append(adv)
                    crossing_adversary_vehicles.append(adv)
                    npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                    print(f"   ✅ Emergency vehicle: offset {arrival_offset:+.1f}s")

                other_paths = [p for p in valid_cross_paths if p != chosen_path]
                if other_paths:
                    adv_start_wp2, adv_end_wp2 = random.choice(other_paths)
                    arrival_offset2 = min(-2.5, arrival_offset + random.uniform(3.0, 3.6))
                    adv2, spawn_wp2, conflict_wp2, orig_diff2 = spawn_cross_vehicle_fixed(
                        world, random.choice(emergency_bps), adv_start_wp2, adv_end_wp2,
                        vehicle, ego_junction_wps, arrival_offset2
                    )
                    if adv2:
                        adversary_vehicles.append(adv2)
                        crossing_adversary_vehicles.append(adv2)
                        npc_shield_meta[adv2.id] = {"orig_diff": orig_diff2, "active": False}
                        print(f"   ✅ Secondary emergency: offset {arrival_offset2:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        elif scenario_id == 2:
            print("🔀 Target: Double Cross Traffic (Fixed Speed)")
            if valid_cross_paths:
                if len(valid_cross_paths) >= 2:
                    selected_paths = random.sample(valid_cross_paths, 2)
                    same_path = False
                else:
                    selected_paths = [valid_cross_paths[0], valid_cross_paths[0]]
                    same_path = True

                if not same_path:
                    offsets = [random.uniform(-7.0, -5.0), random.uniform(-5.0, -3.0)]
                    for i, path in enumerate(selected_paths):
                        adv_start_wp, adv_end_wp = path
                        adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                            world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                            vehicle, ego_junction_wps, offsets[i]
                        )
                        if adv:
                            adversary_vehicles.append(adv)
                            crossing_adversary_vehicles.append(adv)
                            npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                            print(f"   ✅ Cross vehicle {i+1}: offset {offsets[i]:+.1f}s")
                else:
                    adv_start_wp, adv_end_wp = selected_paths[0]
                    offset1 = random.uniform(-6.0, -3.0)
                    adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                        world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                        vehicle, ego_junction_wps, offset1
                    )
                    if adv:
                        adversary_vehicles.append(adv)
                        crossing_adversary_vehicles.append(adv)
                        npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                        print(f"   ✅ Cross vehicle (same path, only one): offset {offset1:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        elif scenario_id == 3:
            print("🚙🔀 Target: Convoy + Cross Threat (Fixed Speed)")
            adv_wp, lead_distance = find_lead_wp_ahead(vehicle, current_route, min_ahead=6.0, max_ahead=11.0)

            if adv_wp is not None:
                adv_transform = adv_wp.transform
                adv_transform.location.z += 1.0
                adv = world.try_spawn_actor(random.choice(adv_bps), adv_transform)
                if adv:
                    adv.set_autopilot(True, tm.get_port())
                    tm.ignore_lights_percentage(adv, 100)
                    tm.vehicle_percentage_speed_difference(adv, 50.0)

                    fwd = adv_transform.get_forward_vector()
                    adv.set_target_velocity(carla.Vector3D(fwd.x * 3.5, fwd.y * 3.5, 0.0))

                    adversary_vehicles.append(adv)
                    print(f"   ✅ Slow lead vehicle spawned {lead_distance:.1f}m ahead (Safe Gap)")
                else:
                    print("   ⚠️ Lead vehicle spawn blocked by collision")
            else:
                print("   ⚠️ Could not spawn lead vehicle (Route too short)")
            if valid_cross_paths:
                chosen_path = random.choice(valid_cross_paths)
                adv_start_wp, adv_end_wp = chosen_path
                arrival_offset = random.uniform(-5.0, -3.0)
                adv2, spawn_wp2, conflict_wp2, orig_diff = spawn_cross_vehicle_fixed(
                    world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                    vehicle, ego_junction_wps, arrival_offset
                )
                if adv2:
                    adversary_vehicles.append(adv2)
                    crossing_adversary_vehicles.append(adv2)
                    npc_shield_meta[adv2.id] = {"orig_diff": orig_diff, "active": False}
                    print(f"   ✅ Cross traffic: offset {arrival_offset:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        elif scenario_id == 4:
            print("🏎️ Target: Aggressive Multi-Vehicle Right-of-Way (Fixed Speed)")
            if valid_cross_paths:
                if len(valid_cross_paths) >= 2:
                    selected_paths = random.sample(valid_cross_paths, 2)
                    same_path = False
                else:
                    selected_paths = [valid_cross_paths[0], valid_cross_paths[0]]
                    same_path = True

                if not same_path:
                    offsets = [random.uniform(-7.0, -5.0), random.uniform(-5.0, -3.0)]
                    for i, path in enumerate(selected_paths):
                        adv_start_wp, adv_end_wp = path
                        adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                            world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                            vehicle, ego_junction_wps, offsets[i]
                        )
                        if adv:
                            adversary_vehicles.append(adv)
                            crossing_adversary_vehicles.append(adv)
                            npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                            print(f"   ✅ Aggressive vehicle {i+1}: offset {offsets[i]:+.1f}s")
                else:
                    adv_start_wp, adv_end_wp = selected_paths[0]
                    offset1 = random.uniform(-7.0, -4.0)
                    adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                        world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                        vehicle, ego_junction_wps, offset1
                    )
                    if adv:
                        adversary_vehicles.append(adv)
                        crossing_adversary_vehicles.append(adv)
                        npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                        print(f"   ✅ Aggressive vehicle (same path, only one): offset {offset1:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        elif scenario_id == 5:
            print("🔄 Target: Staggered Yield Chain (Fixed Speed)")
            if valid_cross_paths:
                static_path = random.choice(valid_cross_paths)
                cross_start_wp = static_path[0]

                # -----------------------------------------------------------------
                # STATIC OCCLUDER PLACEMENT
                # The occluder must sit clearly on the cross road, BEFORE the
                # junction, and away from ego's lane corridor.
                #
                # Two safety measures:
                #   1. Take a waypoint 7-10 m back from the junction entry and
                #      pick the FURTHEST one in the returned list (previously we
                #      took [0], the closest at ~1 m, which put the vehicle in
                #      ego's lane corridor at the junction entrance).
                #   2. If the resulting position still lands on ego's lane,
                #      push the occluder further back along the cross road
                #      until it is clear.
                # -----------------------------------------------------------------
                backward_dist = random.uniform(7.0, 10.0)
                prev_wps = cross_start_wp.previous(backward_dist)
                if prev_wps:
                    static_wp = max(
                        prev_wps,
                        key=lambda w: w.transform.location.distance(cross_start_wp.transform.location),
                    )
                else:
                    static_wp = cross_start_wp

                # -----------------------------------------------------------------
                # Validate the occluder is not on ego's lane. If it is, move
                # further back along the cross road until it is clear.
                # -----------------------------------------------------------------
                def _is_on_ego_lane(test_loc):
                    try:
                        ego_wp = world.get_map().get_waypoint(
                            vehicle.get_location(),
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        test_wp = world.get_map().get_waypoint(
                            test_loc,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving,
                        )
                        if ego_wp is None or test_wp is None:
                            return False
                        return (
                            ego_wp.road_id == test_wp.road_id
                            and ego_wp.lane_id == test_wp.lane_id
                        )
                    except Exception:
                        return False

                max_push_attempts = 6
                for attempt in range(max_push_attempts):
                    if not _is_on_ego_lane(static_wp.transform.location):
                        break
                    extra = 3.0
                    more_prev = static_wp.previous(extra)
                    if not more_prev:
                        break
                    static_wp = max(
                        more_prev,
                        key=lambda w: w.transform.location.distance(cross_start_wp.transform.location),
                    )
                    print(f"   ↻ Static occluder pushed back (still on ego's lane, attempt {attempt+1})")

                # -----------------------------------------------------------------
                # Lateral offset: shift the occluder sideways on the cross road
                # so it partially blocks the cross lane but leaves room for the
                # moving vehicle to pass. Use the cross road's right vector.
                # -----------------------------------------------------------------
                right_vec = static_wp.transform.get_right_vector()
                lane_w = static_wp.lane_width if static_wp.lane_width > 0 else 3.5
                side_offset = lane_w * 0.5 + 2.0   # ~3.75 m for a 3.5 m lane

                static_transform = static_wp.transform
                static_transform.location.z += 1.0
                static_transform.location.x += right_vec.x * side_offset
                static_transform.location.y += right_vec.y * side_offset

                static_adv = world.try_spawn_actor(random.choice(adv_bps), static_transform)
                if static_adv:
                    static_adv.set_autopilot(False)
                    adversary_vehicles.append(static_adv)
                    print(
                        f"   ✅ Static occluder spawned {side_offset:.1f}m off the cross lane "
                        f"({backward_dist:.1f}m before junction)"
                    )
                else:
                    print("   ⚠️ Static occluder spawn blocked, skipping")

                # -----------------------------------------------------------------
                # Moving cross vehicle (unchanged)
                # -----------------------------------------------------------------
                moving_paths = [p for p in valid_cross_paths if p != static_path]
                if not moving_paths:
                    moving_paths = valid_cross_paths
                chosen_path = random.choice(moving_paths)
                adv_start_wp, adv_end_wp = chosen_path
                arrival_offset = random.uniform(-7.0, -4.0)
                moving_adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                    world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                    vehicle, ego_junction_wps, arrival_offset
                )
                if moving_adv:
                    adversary_vehicles.append(moving_adv)
                    crossing_adversary_vehicles.append(moving_adv)
                    npc_shield_meta[moving_adv.id] = {"orig_diff": orig_diff, "active": False}
                    print(f"   ✅ Moving cross vehicle: offset {arrival_offset:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")
                
        elif scenario_id == 6:
            print("🚶 Target: Pedestrian Crosswalk Crossing (Real Crosswalk, Tuned)")
            if target_crosswalk_wp is not None:
                crosswalk_wp = target_crosswalk_wp
                cw_center = crosswalk_wp.transform.location
                cw_right = crosswalk_wp.transform.get_right_vector()
                cw_fwd = crosswalk_wp.transform.get_forward_vector()
                cw_half_w = crosswalk_wp.lane_width / 2.0 + 1.0
                print(f"   🦓 REAL Crosswalk found!")

                side = random.choice([-1, 1])
                sidewalk_offset = cw_half_w + 0.5

                spawn_loc = carla.Location(
                    x=cw_center.x + cw_right.x * side * sidewalk_offset,
                    y=cw_center.y + cw_right.y * side * sidewalk_offset,
                    z=cw_center.z + 1.0
                )
                target_loc = carla.Location(
                    x=cw_center.x - cw_right.x * side * sidewalk_offset,
                    y=cw_center.y - cw_right.y * side * sidewalk_offset,
                    z=cw_center.z
                )
                cross_yaw = math.degrees(math.atan2(-cw_right.y * side, -cw_right.x * side))
                walker_bp = random.choice(list(walker_bps))
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc, carla.Rotation(yaw=cross_yaw)))

                trigger_dist = 22.0
                if walker:
                    speed = 1.3 + random.uniform(-0.1, 0.2)
                    walker_list.append((None, walker, target_loc, speed, trigger_dist, cw_center))
                    print(f"   ✅ Pedestrian on {'right' if side > 0 else 'left'} sidewalk")

                if random.random() < 0.4:
                    spawn_loc2 = carla.Location(
                        x=cw_center.x - cw_right.x * side * sidewalk_offset + cw_fwd.x * 1.8,
                        y=cw_center.y - cw_right.y * side * sidewalk_offset + cw_fwd.y * 1.8,
                        z=cw_center.z + 1.0
                    )
                    target_loc2 = carla.Location(
                        x=cw_center.x + cw_right.x * side * sidewalk_offset + cw_fwd.x * 1.8,
                        y=cw_center.y + cw_right.y * side * sidewalk_offset + cw_fwd.y * 1.8,
                        z=cw_center.z
                    )
                    cross_yaw2 = math.degrees(math.atan2(cw_right.y * side, cw_right.x * side))
                    walker_bp2 = random.choice(list(walker_bps))
                    if walker_bp2.has_attribute('is_invincible'):
                        walker_bp2.set_attribute('is_invincible', 'false')
                    walker2 = world.try_spawn_actor(walker_bp2, carla.Transform(spawn_loc2, carla.Rotation(yaw=cross_yaw2)))
                    if walker2:
                        speed2 = 1.2
                        walker_list.append((None, walker2, target_loc2, speed2, trigger_dist, cw_center))
                        print(f"   ✅ Second pedestrian (Offset parallel line)")
            else:
                print(f"   ⚠️ No real crosswalks in map. Skipping scenario 6 pedestrian.")

            if ego_junction_wps:
                all_tls = world.get_actors().filter('traffic.traffic_light')
                for tl_actor in all_tls:
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        elif scenario_id == 7:
            print("🚴 Target: Cyclist in Path (Slow/Stationary Obstacle)")
            cyclist_idx = min(int(noisy_dist(11, 3)), len(current_route) - 2)
            cyclist_wp = current_route[cyclist_idx]
            cyclist_transform = cyclist_wp.transform
            cyclist_transform.location.z += 1.0
            cyclist = world.try_spawn_actor(random.choice(cyclist_bps), cyclist_transform)
            if cyclist:
                cyclist.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(cyclist, 70.0)
                fwd = cyclist_transform.get_forward_vector()
                cyclist.set_target_velocity(carla.Vector3D(fwd.x * 4.0, fwd.y * 4.0, 0.0))
                moving_cyclists.append(cyclist)
                adversary_vehicles.append(cyclist)
                print(f"   ✅ Slow cyclist spawned at {cyclist_idx}m")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

            if random.random() < 0.4:
                ped_idx2 = max(11, cyclist_idx - random.randint(5, 7))
                ped_wp2 = current_route[ped_idx2]
                right_vec2 = ped_wp2.transform.get_right_vector()
                ped_loc2 = ped_wp2.transform.location
                side2 = random.choice([-1, 1])

                spawn_loc_ped = carla.Location(
                    x=ped_loc2.x + right_vec2.x * side2 * 4.5,
                    y=ped_loc2.y + right_vec2.y * side2 * 4.5,
                    z=ped_loc2.z + 1.0
                )
                target_loc_ped = carla.Location(
                    x=ped_loc2.x - right_vec2.x * side2 * 4.5,
                    y=ped_loc2.y - right_vec2.y * side2 * 4.5,
                    z=ped_loc2.z
                )
                cross_yaw = math.degrees(math.atan2(-right_vec2.y * side2, -right_vec2.x * side2))
                walker_bp_bonus = random.choice(list(walker_bps))
                if walker_bp_bonus.has_attribute('is_invincible'):
                    walker_bp_bonus.set_attribute('is_invincible', 'false')

                walker_bonus = world.try_spawn_actor(walker_bp_bonus, carla.Transform(spawn_loc_ped, carla.Rotation(yaw=cross_yaw)))
                if walker_bonus:
                    speed_ped = 1.1
                    trigger_dist = 11.0
                    walker_list.append((None, walker_bonus, target_loc_ped, speed_ped, trigger_dist, ped_loc2))
                    print(f"   ✅ Occasional pedestrian spawned behind cyclist at {ped_idx2}m (Trigger at {trigger_dist}m)")

        elif scenario_id == 8:
            print("🚶🚗 Target: Pedestrian Crosswalk + Vehicle Combo (Synchronized)")

            if target_crosswalk_wp is not None:
                crosswalk_wp = target_crosswalk_wp
                cw_center = crosswalk_wp.transform.location
                cw_right = crosswalk_wp.transform.get_right_vector()
                cw_fwd = crosswalk_wp.transform.get_forward_vector()
                cw_half_w = crosswalk_wp.lane_width / 2.0 + 1.0
                print(f"   🦓 REAL Crosswalk found near junction!")

                side = random.choice([-1, 1])
                sidewalk_offset = cw_half_w + 0.5
                spawn_loc = carla.Location(
                    x=cw_center.x + cw_right.x * side * sidewalk_offset,
                    y=cw_center.y + cw_right.y * side * sidewalk_offset,
                    z=cw_center.z + 1.0
                )
                target_loc = carla.Location(
                    x=cw_center.x - cw_right.x * side * sidewalk_offset,
                    y=cw_center.y - cw_right.y * side * sidewalk_offset,
                    z=cw_center.z
                )
                cross_yaw = math.degrees(math.atan2(-cw_right.y * side, -cw_right.x * side))
                walker_bp = random.choice(list(walker_bps))
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')

                walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc, carla.Rotation(yaw=cross_yaw)))
                if walker:
                    speed = 1.25
                    trigger_dist = 16.0
                    walker_list.append((None, walker, target_loc, speed, trigger_dist, cw_center))
                    print(f"   ✅ Pedestrian active on {'right' if side > 0 else 'left'} sidewalk")
            else:
                print(f"   ⚠️ No real crosswalks near junction. Skipping scenario 8 pedestrian.")

            if valid_cross_paths and ego_junction_wps:
                ego_exit_wp = ego_junction_wps[-1]
                true_crossing_paths = [p for p in valid_cross_paths if p[1].transform.location.distance(ego_exit_wp.transform.location) > 5.0]
                if not true_crossing_paths:
                    true_crossing_paths = valid_cross_paths

                chosen_path = random.choice(true_crossing_paths)
                adv_start_wp, adv_end_wp = chosen_path

                arrival_offset = random.uniform(1.5, 3.5)

                adv, spawn_wp, conflict_wp, orig_diff = spawn_cross_vehicle_fixed(
                    world, random.choice(adv_bps), adv_start_wp, adv_end_wp,
                    vehicle, ego_junction_wps, arrival_offset
                )
                if adv:
                    adversary_vehicles.append(adv)
                    crossing_adversary_vehicles.append(adv)
                    npc_shield_meta[adv.id] = {"orig_diff": orig_diff, "active": False}
                    print(f"   ✅ Cross-traffic synced with ped-delay: offset {arrival_offset:+.1f}s")

            if ego_junction_wps:
                for tl_actor in world.get_actors().filter('traffic.traffic_light'):
                    if tl_actor.get_location().distance(ego_junction_wps[0].transform.location) < 45.0:
                        tl_actor.set_state(carla.TrafficLightState.Green)
                        tl_actor.freeze(True)
                print("   🚦 Forced nearby Traffic Lights to GREEN for ego vehicle.")

        print(f"====================================================\n")
        if FORCE_SCENARIO is None:
            test_scenario_counter = random.randint(1, 8)
        else:
            test_scenario_counter = FORCE_SCENARIO

        for _ in range(5):
            world.tick()

        spectator = world.get_spectator()
        spectator_loc = final_transform.location + carla.Location(z=10.0) - final_transform.get_forward_vector() * 15.0
        spectator.set_transform(carla.Transform(spectator_loc, carla.Rotation(pitch=-25.0, yaw=final_transform.rotation.yaw)))
        draw_route(world, current_route, life_time=0.1)

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(lambda image: camera_callback(image))

        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '45')
        radar_bp.set_attribute('vertical_fov', '15')
        radar_bp.set_attribute('range', '30')
        radar_bp.set_attribute('points_per_second', '5000')
        radar_transform = carla.Transform(carla.Location(x=2.5, z=1.2))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

        def radar_callback(radar_data):
            nonlocal current_radar_distance, current_route
            min_dist = 30.0
            sensor_origin = radar_data.transform.location
            forward_vec = radar_data.transform.get_forward_vector()
            right_vec = radar_data.transform.get_right_vector()
            up_vec = radar_data.transform.get_up_vector()
            lookahead_wps = []
            if current_route:
                closest_dist = float('inf')
                closest_idx = 0
                for i, wp in enumerate(current_route):
                    dist_to_car = wp.transform.location.distance(sensor_origin)
                    if dist_to_car < closest_dist:
                        closest_dist = dist_to_car
                        closest_idx = i
                lookahead_wps = current_route[closest_idx : closest_idx+30]
            if lookahead_wps and len(radar_data) > 0:
                azi = np.array([detect.azimuth for detect in radar_data])
                alt = np.array([detect.altitude for detect in radar_data])
                dist = np.array([detect.depth for detect in radar_data])
                cos_alt = np.cos(alt)
                x = dist * cos_alt * np.cos(azi)
                y = dist * cos_alt * np.sin(azi)
                z = dist * np.sin(alt)
                pts_x = sensor_origin.x + forward_vec.x * x + right_vec.x * y + up_vec.x * z
                pts_y = sensor_origin.y + forward_vec.y * x + right_vec.y * y + up_vec.y * z
                pts_z = sensor_origin.z + forward_vec.z * x + right_vec.z * y + up_vec.z * z
                wp_x = np.array([wp.transform.location.x for wp in lookahead_wps])
                wp_y = np.array([wp.transform.location.y for wp in lookahead_wps])
                wp_z = np.array([wp.transform.location.z for wp in lookahead_wps])
                wp_radii = np.array([2.5 if wp.is_junction else 1.8 for wp in lookahead_wps])
                dx = pts_x[:, np.newaxis] - wp_x
                dy = pts_y[:, np.newaxis] - wp_y
                dist_sq = dx**2 + dy**2
                in_radius = dist_sq < (wp_radii**2)
                dz = pts_z[:, np.newaxis] - wp_z
                valid_height = (dz > 0.4) & (dz < 2.5)
                is_on_path = np.any(in_radius & valid_height, axis=1)
                if np.any(is_on_path):
                    min_dist = np.min(dist[is_on_path])
                    current_radar_distance = float(min_dist)
                    valid_indices = np.where(is_on_path)[0]
                    for idx in valid_indices:
                        world.debug.draw_point(
                            carla.Location(x=float(pts_x[idx]), y=float(pts_y[idx]), z=float(pts_z[idx])),
                            size=0.08, color=carla.Color(255, 0, 0), life_time=0.06
                        )
            current_radar_distance = min_dist
        radar.listen(lambda data: radar_callback(data))

        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

        def collision_callback(event):
            """
            Instrumented collision reporter.

            Prints the relative position of the colliding actor with respect
            to Ego (REAR / FRONT / LEFT / RIGHT), the 3-D separation distance,
            Ego's speed, the other actor's speed, and the signed closing speed
            projected onto Ego's forward axis.
            """
            nonlocal collision_flag
            other = event.other_actor
            try:
                ego_loc = vehicle.get_location()
                ego_fwd = vehicle.get_transform().get_forward_vector()
                ego_right = vehicle.get_transform().get_right_vector()
                other_loc = other.get_location()

                dx = other_loc.x - ego_loc.x
                dy = other_loc.y - ego_loc.y
                dz = other_loc.z - ego_loc.z

                long_proj = ego_fwd.x * dx + ego_fwd.y * dy
                lat_proj = ego_right.x * dx + ego_right.y * dy
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if abs(lat_proj) > abs(long_proj):
                    relative = "RIGHT" if lat_proj > 0 else "LEFT"
                else:
                    relative = "FRONT" if long_proj > 0 else "REAR"

                ego_v = vehicle.get_velocity()
                ego_speed_kmh = 3.6 * math.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2)

                other_speed_kmh = 0.0
                closing_kmh = 0.0
                try:
                    other_v = other.get_velocity()
                    other_speed_kmh = 3.6 * math.sqrt(other_v.x**2 + other_v.y**2 + other_v.z**2)
                    closing_ms = (other_v.x - ego_v.x) * ego_fwd.x + (other_v.y - ego_v.y) * ego_fwd.y
                    closing_kmh = closing_ms * 3.6
                except Exception:
                    pass

                print(f"💥 COLLISION with {other.type_id} (id={other.id})")
                print(f"   📍 Relative: {relative}")
                print(f"   📏 Distance: {distance:.2f}m")
                print(f"   🚗 Ego speed: {ego_speed_kmh:.1f} km/h")
                print(f"   🚙 Adv speed: {other_speed_kmh:.1f} km/h")
                print(f"   📈 Closing:  {closing_kmh:+.1f} km/h")
            except Exception as e:
                print(f"💥 COLLISION with {other.type_id} (id={other.id}) [report error: {e}]")
            collision_flag = True

        collision_sensor.listen(lambda event: collision_callback(event))

        world.tick()
        time.sleep(0.5)

    reset_environment()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("🔌 Scenario Bridge Ready on Port 5555. Waiting for RL Agent...")

    try:
        while True:
            msg = socket.recv_json()
            command = msg.get("command")
            if command == "reset":
                local_step_count = 0
                reset_environment()
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)
                junction_ctx = get_junction_context(vehicle, world, current_route)
                current_image = ""
                need_image = msg.get("need_image", True)
                if need_image and current_image_raw is not None:
                    _, buffer = cv2.imencode('.jpg', current_image_raw, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    current_image = base64.b64encode(buffer).decode('utf-8')
                socket.send_json({
                    "speed": 0.0,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target,
                    "image": current_image if current_image else "",
                    "radar_distance": current_radar_distance,
                    "collision": False,
                    "in_junction": junction_ctx["in_junction"],
                    "approaching_junction": junction_ctx["approaching_junction"],
                    "junction_distance": junction_ctx["junction_distance"],
                    "traffic_light": junction_ctx["traffic_light"],
                    "vehicles_in_junction": junction_ctx["vehicles_in_junction"],
                    "scenario_id": current_scenario_id,
                    "crossing_threat": False,
                    "crossing_threat_distance": 999.0
                })
            elif command == "step":
                local_step_count += 1
                throttle = msg.get("throttle", 0.0)
                steer = msg.get("steer", 0.0)
                brake = msg.get("brake", 0.0)
                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                vehicle.apply_control(control)

                # ==========================================================
                # NPC SAFETY SHIELD — active ONLY for scenarios 1-5
                # ----------------------------------------------------------
                # Scenarios 6, 7, 8 are intentionally NOT shielded:
                #   - 6/7 place an obstacle directly on Ego's path; the RL
                #     policy must perceive and respond on its own.
                #   - 8 is a multi-threat scenario (pedestrian + vehicle);
                #     the challenge must remain and the vehicle arrives after
                #     the pedestrian triggers.
                # The shield runs BEFORE world.tick() so adjustments take
                # effect on this tick.
                # ==========================================================
                if crossing_adversary_vehicles and current_scenario_id in [1, 2, 3, 4, 5]:
                    npc_safety_shield(
                        vehicle,
                        crossing_adversary_vehicles,
                        npc_shield_meta,
                    )

                world.tick()
                draw_route(world, current_route, life_time=0.1)

                to_remove = []
                for idx, item in enumerate(walker_list):
                    ctrl = item[0]
                    walker = item[1]
                    target = item[2]
                    speed = item[3] if len(item) > 3 else 1.5
                    trigger_dist = item[4] if len(item) > 4 else 999.0
                    trigger_ref = item[5] if len(item) > 5 else walker.get_location()
                    if walker is not None and walker.is_alive:
                        w_loc = walker.get_location()
                        dist_to_target = w_loc.distance(target)
                        ego_fwd = vehicle.get_transform().get_forward_vector()
                        ego_loc = vehicle.get_location()
                        dot = ego_fwd.x * (w_loc.x - ego_loc.x) + ego_fwd.y * (w_loc.y - ego_loc.y)
                        if dist_to_target < 1.5 or dot < -8.0:
                            to_remove.append(idx)
                            try:
                                if ctrl:
                                    ctrl.stop()
                                    ctrl.destroy()
                                walker.destroy()
                                print("   ✅ Pedestrian crossed the crosswalk -> removed.")
                            except Exception:
                                pass
                        elif trigger_ref.distance(ego_loc) > trigger_dist:
                            pass
                        elif ctrl is None:
                            direction = carla.Vector3D(target.x - w_loc.x, target.y - w_loc.y, 0.0)
                            mag = math.sqrt(direction.x**2 + direction.y**2)
                            if mag > 0:
                                direction.x /= mag
                                direction.y /= mag
                            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed, jump=False))
                        elif ctrl is not None:
                            try:
                                ctrl.go_to_location(target)
                            except Exception:
                                pass
                for idx in reversed(to_remove):
                    walker_list.pop(idx)

                for cyc in moving_cyclists[:]:
                    try:
                        if cyc is None or not cyc.is_alive:
                            moving_cyclists.remove(cyc)
                            continue
                        cv = cyc.get_velocity()
                        c_speed = math.sqrt(cv.x**2 + cv.y**2 + cv.z**2)
                        if c_speed < 1.0:
                            cf = cyc.get_transform().get_forward_vector()
                            cyc.set_target_velocity(carla.Vector3D(cf.x * 4.0, cf.y * 4.0, 0.0))
                    except Exception:
                        pass

                v = vehicle.get_velocity()
                speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)
                junction_ctx = get_junction_context(vehicle, world, current_route)
                crossing_threat, crossing_threat_distance = get_crossing_threat(
                    vehicle, current_route, crossing_adversary_vehicles
                )
                hit = collision_flag
                collision_flag = False
                current_image = ""
                need_image = msg.get("need_image", False)
                if need_image and current_image_raw is not None:
                    _, buffer = cv2.imencode('.jpg', current_image_raw, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    current_image = base64.b64encode(buffer).decode('utf-8')
                socket.send_json({
                    "speed": speed,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target,
                    "image": current_image if current_image else "",
                    "radar_distance": current_radar_distance,
                    "collision": hit,
                    "in_junction": junction_ctx["in_junction"],
                    "approaching_junction": junction_ctx["approaching_junction"],
                    "junction_distance": junction_ctx["junction_distance"],
                    "traffic_light": junction_ctx["traffic_light"],
                    "vehicles_in_junction": junction_ctx["vehicles_in_junction"],
                    "scenario_id": current_scenario_id,
                    "crossing_threat": crossing_threat,
                    "crossing_threat_distance": crossing_threat_distance
                })
    except KeyboardInterrupt:
        pass
    finally:
        print("🛑 Shutting down CARLA safely...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        clear_adversaries()
        if collision_sensor: collision_sensor.destroy()
        if camera: camera.destroy()
        if radar: radar.destroy()
        if vehicle: vehicle.destroy()

if __name__ == '__main__':
    main()