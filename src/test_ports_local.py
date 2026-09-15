#!/usr/bin/env python3
import asyncio
from mavsdk import System

async def try_port(port):
    drone = System()
    addr = f"udp://127.0.0.1:{port}"
    print(f"Trying {addr}...")
    try:
        await drone.connect(system_address=addr)
        # Wait up to 3 seconds for connection
        for _ in range(30):
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print(f"✅ Connected on {addr}")
                    return drone
            await asyncio.sleep(0.1)
        print(f"⏱️ Timeout on {addr}")
    except Exception as e:
        print(f"❌ Error on {addr}: {e}")
    return None

async def main():
    # From logs: onboard remote port 14540, normal remote 14550, onboard listening 14580, normal listening 18570
    ports = [14540, 14550, 14580, 18570]
    for port in ports:
        drone = await try_port(port)
        if drone:
            print("\nGetting telemetry...")
            try:
                async for flight_mode in drone.telemetry.flight_mode():
                    print(f"Flight mode: {flight_mode}")
                    break
                async for health in drone.telemetry.health():
                    print(f"Health: {health}")
                    break
            except Exception as e:
                print(f"Telemetry error: {e}")
            break
    else:
        print("\n❌ No connection on any port.")

if __name__ == "__main__":
    asyncio.run(main())
