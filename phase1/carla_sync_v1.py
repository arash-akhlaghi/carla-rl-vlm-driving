import carla
import zmq
import base64
import numpy as np
import cv2
import time
import math

def normalize_angle(angle):
    """Normalizes an angle to the [-180, 180] range."""
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def get_leftmost_lane(waypoint):
    """Finds the leftmost driving lane for the given waypoint."""
    current_wp = waypoint
    while True:
        left_wp = current_wp.get_left_lane()
        if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
            current_wp = left_wp
        else:
            break
    return current_wp

def generate_target_route(start_waypoint, distance=100.0, step_size=1.0, turn_direction="left"):
    """Generates a route of waypoints for the vehicle to follow."""
    if turn_direction == "left":
        start_waypoint = get_leftmost_lane(start_waypoint)

    route = [start_waypoint]
    current_wp = start_waypoint
    accumulated_dist = 0.0

    while accumulated_dist < distance:
        next_wps = current_wp.next(step_size)
        if not next_wps:
            break
        
        if len(next_wps) > 1:
            curr_transform = current_wp.transform
            curr_loc = curr_transform.location
            curr_yaw = math.radians(curr_transform.rotation.yaw)

            def get_local_y(wp):
                dx = wp.transform.location.x - curr_loc.x
                dy = wp.transform.location.y - curr_loc.y
                return -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

            if turn_direction == "left":
                current_wp = min(next_wps, key=get_local_y)
            elif turn_direction == "right":
                current_wp = max(next_wps, key=get_local_y)
            else:
                current_wp = min(next_wps, key=lambda wp: abs(get_local_y(wp)))
        else:
            current_wp = next_wps[0]

        route.append(current_wp)
        accumulated_dist += step_size
        
    return route

def draw_route(world, route):
    """Draws the route in the CARLA simulator using green debug points."""
    for wp in route:
        loc = wp.transform.location
        loc.z += 0.5 
        world.debug.draw_point(loc, size=0.15, color=carla.Color(r=0, g=255, b=0), life_time=500.0)

def get_route_errors(vehicle, route):
    """
    Calculates Cross Track Error (CTE), Heading Error, and Distance to Target.
    Returns 5 values: cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target.
    """
    if not route: 
        return 0.0, 0.0, 0.0, 0.0, 999.0

    vehicle_loc = vehicle.get_location()
    vehicle_yaw = vehicle.get_transform().rotation.yaw

    # --- Find the closest waypoint for CTE calculation ---
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

    # --- 🎯 Calculate the distance to the final target (end of route) ---
    final_destination = route[-1].transform.location
    dx_target = final_destination.x - vehicle_loc.x
    dy_target = final_destination.y - vehicle_loc.y
    dist_to_target = math.sqrt(dx_target**2 + dy_target**2)

    return cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Enable Synchronous Mode for precise step execution
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    
    # Setup Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    vehicle = None
    camera = None
    current_image = None
    current_route = [] 

    def camera_callback(image):
        nonlocal current_image
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        _, buffer = cv2.imencode('.jpg', array)
        current_image = base64.b64encode(buffer).decode('utf-8')

    def reset_environment():
        nonlocal vehicle, camera, current_image, current_route
        
        if camera: camera.destroy(); camera = None
        if vehicle: vehicle.destroy(); vehicle = None

        # Determine spawn point (Town03 - reliable intersection)
        spawn_points = world.get_map().get_spawn_points()
        safe_transform = spawn_points[50] if len(spawn_points) > 50 else spawn_points[0]
        
        # Shift to the leftmost lane
        spawn_wp = world.get_map().get_waypoint(safe_transform.location)
        spawn_wp = get_leftmost_lane(spawn_wp)
        
        # 🟢 MOVE THE CAR CLOSER TO THE INTERSECTION (Exactly 2 Meters before)
        current_wp = spawn_wp
        dist_to_junction = 0.0
        while not current_wp.is_junction and dist_to_junction < 50.0:
            next_wps = current_wp.next(1.0)
            if not next_wps:
                break
            current_wp = next_wps[0]
            dist_to_junction += 1.0

        if dist_to_junction > 2.0:
            advance_dist = dist_to_junction - 2.0  # Stops 2 meters before junction
            spawn_wp = spawn_wp.next(advance_dist)[0]
        
        # Place directly on the ground to prevent floating
        final_transform = spawn_wp.transform
        final_transform.location.z += 0.5
        
        vehicle = world.spawn_actor(vehicle_bp, final_transform)

        # Set spectator camera
        spectator = world.get_spectator()
        spectator_loc = final_transform.location + carla.Location(z=10.0) - final_transform.get_forward_vector() * 15.0
        spectator.set_transform(carla.Transform(spectator_loc, carla.Rotation(pitch=-25.0, yaw=final_transform.rotation.yaw)))

        # Generate and draw the route
        start_wp = world.get_map().get_waypoint(vehicle.get_transform().location)
        current_route = generate_target_route(start_wp, distance=100.0, step_size=1.0, turn_direction="left")
        draw_route(world, current_route)

        # Spawn ego camera
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        camera.listen(lambda image: camera_callback(image))

        world.tick()
        time.sleep(0.5)

    reset_environment()

    # ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    print("🔌 ZMQ Bridge Ready on Port 5555. Waiting for RL Agent...")

    try:
        while True:
            msg = socket.recv_json()
            command = msg.get("command")

            if command == "reset":
                reset_environment()
                # 🟢 Extract 5 variables including dist_to_target
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)
                socket.send_json({
                    "speed": 0.0,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target, # Send to RL
                    "image": current_image if current_image else ""
                })

            elif command == "step":
                throttle = msg.get("throttle", 0.0)
                steer = msg.get("steer", 0.0)
                brake = msg.get("brake", 0.0)

                control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
                vehicle.apply_control(control)

                world.tick()

                v = vehicle.get_velocity()
                speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                
                # 🟢 Extract 5 variables including dist_to_target
                cte, heading_error, dist_to_wp, angle_to_wp, dist_to_target = get_route_errors(vehicle, current_route)

                socket.send_json({
                    "speed": speed,
                    "cte": cte,
                    "heading_error": heading_error,
                    "dist_to_wp": dist_to_wp,
                    "angle_to_wp": angle_to_wp,
                    "dist_to_target": dist_to_target, # Send to RL
                    "image": current_image if current_image else ""
                })

    except KeyboardInterrupt:
        pass
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        if camera: camera.destroy()
        if vehicle: vehicle.destroy()

if __name__ == '__main__':
    main()