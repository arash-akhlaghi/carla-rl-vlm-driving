# import carla
# import zmq
# import base64
# import numpy as np
# import cv2
# import time
# import math
# import random

# def normalize_angle(angle):
#     """Normalizes an angle to the [-180, 180] range."""
#     while angle > 180: angle -= 360
#     while angle < -180: angle += 360
#     return angle

# def get_leftmost_lane(waypoint):
#     """Finds the leftmost driving lane for a given waypoint."""
#     current_wp = waypoint
#     while True:
#         left_wp = current_wp.get_left_lane()
#         if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
#             current_wp = left_wp
#         else:
#             break
#     return current_wp

# def generate_safe_route(start_waypoint, target_distance=100.0, step_size=1.0):
#     """
#     Dynamically generates a safe and valid forward route from the start waypoint.
#     """
#     route = [start_waypoint]
#     current_wp = start_waypoint
#     accumulated_dist = 0.0

#     while accumulated_dist < target_distance:
#         next_wps = current_wp.next(step_size)
#         if not next_wps:
#             break
        
#         valid_wps = [wp for wp in next_wps if wp.lane_type == carla.LaneType.Driving]
#         if not valid_wps:
#             break
            
#         current_wp = valid_wps[0] 
#         route.append(current_wp)
#         accumulated_dist += step_size
        
#     return route

# def draw_route(world, route):
#     """Draws the target green route points in the CARLA simulator."""
#     for wp in route:
#         loc = wp.transform.location
#         loc.z += 0.5  # Lift slightly above road surface
#         world.debug.draw_point(loc, size=0.15, color=carla.Color(r=0, g=255, b=0), life_time=500.0)

# def get_route_errors(vehicle, route):
#     """
#     Calculates Cross-Track Error (CTE), Heading Error, and Distances to targets.
#     """
#     if not route or len(route) < 2: 
#         return 5.0, 0.0, 0.0, 0.0, 999.0

#     vehicle_loc = vehicle.get_location()
#     vehicle_yaw = vehicle.get_transform().rotation.yaw

#     min_dist = float('inf')
#     closest_idx = 0
#     for i, wp in enumerate(route):
#         dist = wp.transform.location.distance(vehicle_loc)
#         if dist < min_dist:
#             min_dist = dist
#             closest_idx = i

#     target_idx = min(closest_idx + 2, len(route) - 1)
#     target_wp = route[target_idx]

#     target_loc = target_wp.transform.location
#     target_yaw = target_wp.transform.rotation.yaw

#     dx = target_loc.x - vehicle_loc.x
#     dy = target_loc.y - vehicle_loc.y
#     dist_to_wp = math.sqrt(dx**2 + dy**2)

#     target_vector_yaw = math.degrees(math.atan2(dy, dx))
#     angle_to_wp = normalize_angle(target_vector_yaw - vehicle_yaw)
#     heading_error = normalize_angle(target_yaw - vehicle_yaw)
#     cte = dist_to_wp * math.sin(math.radians(angle_to_wp))

#     final_destination = route[-1].transform.location
#     dx_target = final_destination.x - vehicle_loc.x
#     dy_target = final_destination.y - vehicle_loc.y
#     dist_to_target = math.sqrt(dx_target**2 + dy_target**2)

#     return cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target

# def is_spawn_point_clear(world, location, radius=3.0):
#     """Checks if a given location is free of other vehicles."""
#     vehicles = world.get_actors().filter('*vehicle*')
#     for v in vehicles:
#         if v.get_location().distance(location) < radius:
#             return False # Space is occupied
#     return True

# def main():
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(20.0) # Increased timeout for stability
#     world = client.get_world()

#     # Enable synchronous mode
#     settings = world.get_settings()
#     settings.synchronous_mode = True
#     settings.fixed_delta_seconds = 0.05
#     world.apply_settings(settings)

#     blueprint_library = world.get_blueprint_library()
#     vehicle_bp = blueprint_library.filter('model3')[0]

#     camera_bp = blueprint_library.find('sensor.camera.rgb')
#     camera_bp.set_attribute('image_size_x', '800')
#     camera_bp.set_attribute('image_size_y', '600')
#     camera_bp.set_attribute('fov', '90')

#     vehicle = None
#     camera = None
#     current_image = None
#     current_route = [] 

#     # ---------------------------------------------------------
#     # 🗺️ 1. Find all valid junction entrances ONCE at startup
#     # ---------------------------------------------------------
#     print("🗺️ Parsing map topology to find junction entrances...")
#     topology = world.get_map().get_topology()
#     junction_entries = []
    
#     for segment in topology:
#         end_wp = segment[1] 
#         if end_wp.lane_type == carla.LaneType.Driving:
#             next_wps = end_wp.next(1.0)
#             if next_wps and next_wps[0].is_junction:
#                 junction_entries.append(end_wp)

#     junction_entries = list({(wp.transform.location.x, wp.transform.location.y): wp for wp in junction_entries}.values())
#     print(f"✅ Found {len(junction_entries)} valid junction entrances!")

#     # ---------------------------------------------------------
#     # 📌 2. Select ONE fixed intersection for this execution session
#     # ---------------------------------------------------------
#     selected_spawn_wp = None
#     random.shuffle(junction_entries)
    
#     for candidate_wp in junction_entries:
#         prev_wps = candidate_wp.previous(3.0)
#         spawn_candidate = prev_wps[0] if prev_wps else candidate_wp
        
#         # Fast route logic test (no spawning required)
#         test_route = generate_safe_route(spawn_candidate, target_distance=50.0, step_size=1.0)
#         if len(test_route) > 20:
#             selected_spawn_wp = spawn_candidate
#             print(f"📌 Fixed Intersection Selected at Location: ({spawn_candidate.transform.location.x:.1f}, {spawn_candidate.transform.location.y:.1f})")
#             break

#     if selected_spawn_wp is None:
#         spawn_points = world.get_map().get_spawn_points()
#         selected_spawn_wp = world.get_map().get_waypoint(spawn_points[0].location)
#         print("⚠️ Fallback: Defaulting to first map spawn point.")

#     # ---------------------------------------------------------
#     # 🔁 Reset function that reuses the SAME fixed intersection
#     # ---------------------------------------------------------
#     def camera_callback(image):
#         nonlocal current_image
#         array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#         array = np.reshape(array, (image.height, image.width, 4))
#         array = array[:, :, :3]
#         _, buffer = cv2.imencode('.jpg', array)
#         current_image = base64.b64encode(buffer).decode('utf-8')

#     def destroy_actors():
#         """Safely destroy sensors and vehicles to prevent CARLA core dumps."""
#         nonlocal camera, vehicle
#         if camera is not None:
#             camera.stop()
#             camera.destroy()
#             camera = None
#         if vehicle is not None:
#             vehicle.destroy()
#             vehicle = None
#         # Tick to let CARLA clean up memory
#         world.tick()

#     def reset_environment():
#         nonlocal vehicle, camera, current_image, current_route
        
#         # Safely clean up previous state
#         destroy_actors()

#         final_transform = selected_spawn_wp.transform
#         final_transform.location.z += 1.0  # Safe drop height
        
#         # Clear traffic if they block our fixed intersection
#         while not is_spawn_point_clear(world, final_transform.location, radius=4.0):
#             print("⏳ Traffic is blocking the spawn point. Waiting/Clearing...")
#             # Optional: aggressively destroy blocking traffic
#             for actor in world.get_actors().filter('*vehicle*'):
#                 if actor.get_location().distance(final_transform.location) < 4.0:
#                     actor.destroy()
#             world.tick()
            
#         # Spawn new ego vehicle
#         vehicle = world.spawn_actor(vehicle_bp, final_transform)

#         # Set Spectator view
#         spectator = world.get_spectator()
#         spectator_loc = final_transform.location + carla.Location(z=10.0) - final_transform.get_forward_vector() * 15.0
#         spectator.set_transform(carla.Transform(spectator_loc, carla.Rotation(pitch=-25.0, yaw=final_transform.rotation.yaw)))

#         # Generate and draw route
#         current_route = generate_safe_route(selected_spawn_wp, target_distance=100.0, step_size=1.0)
#         draw_route(world, current_route)

#         # Spawn Camera
#         camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
#         camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
#         camera.listen(lambda image: camera_callback(image))

#         # Initial ticks to stabilize physics
#         for _ in range(5):
#             world.tick()
#         time.sleep(0.1)

#     # Initialize environment
#     reset_environment()

#     # ---------------------------------------------------------
#     # 🔌 ZMQ Server Setup
#     # ---------------------------------------------------------
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.bind("tcp://*:5555")
#     print("🔌 ZMQ Bridge Ready on Port 5555. Waiting for RL Agent...")

#     try:
#         while True:
#             msg = socket.recv_json()
#             command = msg.get("command")

#             if command == "reset":
#                 reset_environment()
#                 cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)
#                 socket.send_json({
#                     "speed": 0.0,
#                     "cte": cte,
#                     "heading_error": heading_error,
#                     "dist_to_wp": dist_to_wp,
#                     "angle_to_wp": angle_to_wp,
#                     "dist_to_target": dist_to_target,
#                     "image": current_image if current_image else ""
#                 })

#             elif command == "step":
#                 throttle = msg.get("throttle", 0.0)
#                 steer = msg.get("steer", 0.0)
#                 brake = msg.get("brake", 0.0)

#                 control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
#                 vehicle.apply_control(control)

#                 world.tick()

#                 v = vehicle.get_velocity()
#                 speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
#                 cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)

#                 socket.send_json({
#                     "speed": speed,
#                     "cte": cte,
#                     "heading_error": heading_error,
#                     "dist_to_wp": dist_to_wp,
#                     "angle_to_wp": angle_to_wp,
#                     "dist_to_target": dist_to_target,
#                     "image": current_image if current_image else ""
#                 })

#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Crucial final cleanup
#         settings.synchronous_mode = False
#         world.apply_settings(settings)
#         destroy_actors()

# if __name__ == '__main__':
#     main()

import carla
import zmq
import base64
import numpy as np
import cv2
import time
import math
import random

def normalize_angle(angle):
    """Normalizes an angle to the [-180, 180] range."""
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def draw_route(world, route, life_time=0.1):
    """
    Draws the generated route in the CARLA simulator using green dots.
    The short life_time (0.1s) ensures the dots disappear instantly when the episode resets.
    """
    for wp in route:
        loc = wp.transform.location
        loc.z += 0.5  # Lift slightly above the road surface to prevent clipping
        world.debug.draw_point(loc, size=0.15, color=carla.Color(r=0, g=255, b=0), life_time=life_time)

def generate_safe_route(start_waypoint, target_distance=100.0, step_size=1.0):
    """
    Dynamically generates a safe and valid route forward.
    It picks the longest valid trajectory through the intersection 
    to avoid instant CTE (Cross-Track Error) failures.
    """
    route = [start_waypoint]
    current_wp = start_waypoint
    accumulated_dist = 0.0

    while accumulated_dist < target_distance:
        next_wps = current_wp.next(step_size)
        if not next_wps:
            # Reached a dead end
            break
        
        # If multiple choices exist (e.g., inside an intersection), 
        # just pick the first valid driving lane to keep the route going.
        valid_wps = [wp for wp in next_wps if wp.lane_type == carla.LaneType.Driving]
        if not valid_wps:
            break
            
        # Arbitrarily pick the first valid option to ensure we have a path
        current_wp = valid_wps[0] 
        route.append(current_wp)
        accumulated_dist += step_size
        
    return route

def get_route_errors(vehicle, route):
    """
    Calculates Cross-Track Error (CTE), Heading Error, and Distances.
    If the route is empty, it returns a massive failure state.
    """
    if not route or len(route) < 2: 
        return 5.0, 0.0, 0.0, 0.0, 999.0 # Force CTE failure if route is broken

    vehicle_loc = vehicle.get_location()
    vehicle_yaw = vehicle.get_transform().rotation.yaw

    # Find the closest waypoint to the vehicle
    min_dist = float('inf')
    closest_idx = 0
    for i, wp in enumerate(route):
        dist = wp.transform.location.distance(vehicle_loc)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Look slightly ahead (lookahead distance) for stable tracking
    target_idx = min(closest_idx + 2, len(route) - 1)
    target_wp = route[target_idx]

    target_loc = target_wp.transform.location
    target_yaw = target_wp.transform.rotation.yaw

    # Vector to target
    dx = target_loc.x - vehicle_loc.x
    dy = target_loc.y - vehicle_loc.y
    dist_to_wp = math.sqrt(dx**2 + dy**2)

    # Errors
    target_vector_yaw = math.degrees(math.atan2(dy, dx))
    angle_to_wp = normalize_angle(target_vector_yaw - vehicle_yaw)
    heading_error = normalize_angle(target_yaw - vehicle_yaw)
    cte = dist_to_wp * math.sin(math.radians(angle_to_wp))

    # Distance to the very end of the green route (Target Coordinate)
    final_destination = route[-1].transform.location
    dx_target = final_destination.x - vehicle_loc.x
    dy_target = final_destination.y - vehicle_loc.y
    dist_to_target = math.sqrt(dx_target**2 + dy_target**2)

    return cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    vehicle = None
    camera = None
    current_image = None
    current_route = [] 

    # ---------------------------------------------------------
    # 1. Cache Intersection Entrances ONCE at startup
    # ---------------------------------------------------------
    print("🗺️ Parsing map topology to find safe junction entrances...")
    topology = world.get_map().get_topology()
    junction_entries = []
    
    for segment in topology:
        # segment[0] is start, segment[1] is end of a lane segment
        end_wp = segment[1] 
        if end_wp.lane_type == carla.LaneType.Driving:
            next_wps = end_wp.next(1.0)
            if next_wps and next_wps[0].is_junction:
                junction_entries.append(end_wp)

    # Remove duplicates based on coordinates
    junction_entries = list({(wp.transform.location.x, wp.transform.location.y): wp for wp in junction_entries}.values())
    print(f"✅ Found {len(junction_entries)} valid junction entrances!")

    def camera_callback(image):
        """Processes the camera image and encodes it to Base64 for the VLM."""
        nonlocal current_image
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        _, buffer = cv2.imencode('.jpg', array)
        current_image = base64.b64encode(buffer).decode('utf-8')

    def reset_environment():
        """Resets the vehicle at a new safe junction and generates a new route."""
        nonlocal vehicle, camera, current_image, current_route
        
        # Cleanup previous episode actors
        if camera: camera.destroy(); camera = None
        if vehicle: vehicle.destroy(); vehicle = None

        spawn_wp = None
        
        # ---------------------------------------------------------
        # 2. Randomly select an intersection and attempt Safe Spawn
        # ---------------------------------------------------------
        max_attempts = 20
        for _ in range(max_attempts):
            if junction_entries:
                candidate_wp = random.choice(junction_entries)
                
                # Move the spawn point ~3 meters BEFORE the junction
                prev_wps = candidate_wp.previous(3.0)
                spawn_wp = prev_wps[0] if prev_wps else candidate_wp
            else:
                # Fallback if no junctions found
                spawn_points = world.get_map().get_spawn_points()
                spawn_wp = world.get_map().get_waypoint(random.choice(spawn_points).location)

            # Test if this spawn point yields a decent length route (at least 20 meters)
            test_route = generate_safe_route(spawn_wp, target_distance=50.0, step_size=1.0)
            
            if len(test_route) > 20:
                # Try to spawn the vehicle HERE
                final_transform = spawn_wp.transform
                final_transform.location.z += 1.0  # Increased Z to avoid ground collision completely
                
                # use try_spawn_actor to prevent crashes if the spot is blocked
                vehicle = world.try_spawn_actor(vehicle_bp, final_transform)
                if vehicle is not None:
                    current_route = test_route
                    break # Found a good spawn point and successfully spawned!
                else:
                    print("⚠️ Spawn collision detected (Spot blocked). Retrying...")
            else:
                print("⚠️ Selected junction led to a dead-end. Retrying...")

        if vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts. Check map geometry.")

        # Set Spectator Camera for top-down debugging view
        spectator = world.get_spectator()
        spectator_loc = final_transform.location + carla.Location(z=10.0) - final_transform.get_forward_vector() * 15.0
        spectator.set_transform(carla.Transform(spectator_loc, carla.Rotation(pitch=-25.0, yaw=final_transform.rotation.yaw)))

        # Draw the validated route initially
        draw_route(world, current_route, life_time=0.1)

        # Attach RGB Camera to the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(lambda image: camera_callback(image))

        # Warm-up ticks to initialize physics and camera sensor
        world.tick()
        time.sleep(0.5)

    # Initialize first episode
    reset_environment()

    # ---------------------------------------------------------
    # 🔌 ZMQ Server Setup (Communicates with RL Agent)
    # ---------------------------------------------------------
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("🔌 ZMQ Bridge Ready on Port 5555. Waiting for RL Agent...")

    try:
        while True:
            # Wait for command from RL Agent (CarlaVLMEnv)
            msg = socket.recv_json()
            command = msg.get("command")

            if command == "reset":
                reset_environment()
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)
                socket.send_json({
                    "speed": 0.0,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target,
                    "image": current_image if current_image else ""
                })

            elif command == "step":
                throttle = msg.get("throttle", 0.0)
                steer = msg.get("steer", 0.0)
                brake = msg.get("brake", 0.0)

                # Apply RL controls to the ego vehicle
                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                vehicle.apply_control(control)

                # Step the simulator physics forward
                world.tick()

                # Refresh the route dots continuously. 
                # Since lifetime is 0.1s, they will vanish cleanly on reset.
                draw_route(world, current_route, life_time=0.1)

                # Calculate Current State
                v = vehicle.get_velocity()
                speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)

                # Send telemetry back to RL Agent
                socket.send_json({
                    "speed": speed,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target,
                    "image": current_image if current_image else ""
                })

    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup safely and restore synchronous mode
        print("🛑 Shutting down CARLA safely...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()

if __name__ == '__main__':
    main()