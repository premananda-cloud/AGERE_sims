#!/usr/bin/env python3
"""Test MAVSDK connection to PX4 SITL"""
import asyncio
from mavsdk import System

async def test_connection():
    print("Connecting to PX4 via MAVSDK...")
    drone = System()
    
    # Connect to the PX4 container via host network
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("✅ Connected to PX4!")
            break
    
    # Get some basic info
    print("\nGetting drone info...")
    async for info in drone.telemetry.flight_mode():
        print(f"Flight mode: {info}")
        break
    
    async for health in drone.telemetry.health():
        print(f"Health: {health}")
        break
    
    return drone

async def fly_manual():
    drone = await test_connection()
    
    print("\n🚁 Ready to fly! Taking off...")
    
    # Arm and takeoff
    await drone.action.arm()
    print("✅ Armed")
    
    await drone.action.takeoff()
    print("✅ Takeoff command sent")
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Move forward (in NED frame: north, east, down)
    print("Moving forward...")
    await drone.action.set_attitude(roll=0, pitch=-10, yaw=0, throttle=0.5)
    await asyncio.sleep(3)
    
    # Hover
    print("Hovering...")
    await drone.action.set_attitude(roll=0, pitch=0, yaw=0, throttle=0.5)
    await asyncio.sleep(2)
    
    # Land
    print("Landing...")
    await drone.action.land()
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(fly_manual())
