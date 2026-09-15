import asyncio
from mavsdk import System


async def print_altitude(drone):
    async for pos in drone.telemetry.position():
        print(f"  altitude: {pos.relative_altitude_m:6.2f} m")


async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected")
            break

    print("Waiting for GPS...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("✅ GPS OK")
            break

    asyncio.ensure_future(print_altitude(drone))

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)

    print("Landing...")
    await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("✅ Landed")
            break


if __name__ == "__main__":
    asyncio.run(run())
