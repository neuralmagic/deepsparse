import asyncio
from fastapi_websocket_pubsub import PubSubClient

async def on_events(data, topic):
    print(data)

async def main():
   
    client = PubSubClient(["tweets"], callback=on_events)
    client.start_client(f"ws://localhost:8000/pubsub")
    await client.wait_until_done()

asyncio.run(main())