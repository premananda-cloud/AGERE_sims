import asyncio
from mavsdk import System


async def run():
    drone = System()
    # Use udpin:// (MAVSDK 3.x) instead of deprecated udp://
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Drone discovered!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ Global position estimate OK")
            break

    async for pos in drone.telemetry.position():
        print(f"Position: lat={pos.latitude_deg:.6f}, "
              f"lon={pos.longitude_deg:.6f}, "
              f"alt={pos.relative_altitude_m:.2f} m")
        break


if __name__ == "__main__":
    asyncio.run(run())
