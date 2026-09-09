import carla
import random
import time
import logging

def main():
    """
    Spawns dynamic traffic with ANTI-DEADLOCK settings.
    Ensures vehicles keep moving and don't get permanently stuck in intersections.
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    vehicles_list = []
    walkers_list = []
    all_actors = []
    
    logging.info("Connecting to CARLA server...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
    except Exception as e:
        logging.error(f"Failed to connect to CARLA: {e}")
        return

    # 2. Setup Traffic Manager with Anti-Deadlock Rules
    traffic_manager = client.get_trafficmanager(8000)
  #  traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0) # Keep cars closer
    traffic_manager.global_percentage_speed_difference(-20.0) # Force cars to drive 20% FASTER than limit
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*') if int(bp.get_attribute('number_of_wheels')) == 4]
    pedestrian_bps = blueprint_library.filter('walker.pedestrian.*')
    
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    # We slightly reduce density so they don't jam the intersections too badly
    NUM_VEHICLES = 30 
    NUM_PEDESTRIANS = 20
    
    logging.info(f"Spawning {NUM_VEHICLES} autonomous vehicles...")
    
    for i in range(min(NUM_VEHICLES, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        spawn_point = spawn_points[i]
        
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            
            # 🟢 ANTI-DEADLOCK RULES 🟢
            # Ignore traffic lights 15% of the time to break deadlocks
            traffic_manager.ignore_lights_percentage(vehicle, 15)
            # Allow lane changes to avoid stuck cars
            traffic_manager.random_left_lanechange_percentage(vehicle, 10)
            traffic_manager.random_right_lanechange_percentage(vehicle, 10)
            
            vehicles_list.append(vehicle)
            all_actors.append(vehicle)

    logging.info(f"Spawning {NUM_PEDESTRIANS} pedestrians...")
    
    for i in range(NUM_PEDESTRIANS):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            walker_bp = random.choice(pedestrian_bps)
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker is not None:
                walkers_list.append(walker)
                all_actors.append(walker)
                
    logging.info("✅ Dynamic Traffic spawned successfully! Cars will not deadlock.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("\n🧹 Destroying all spawned traffic...")
        client.apply_batch([carla.command.DestroyActor(x) for x in all_actors])
        logging.info("✅ Clean up complete!")

if __name__ == '__main__':
    main()