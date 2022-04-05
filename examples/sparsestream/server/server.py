from deepsparse.transformers import pipeline
from fastapi import FastAPI
from fastapi.routing import APIRouter
from fastapi_websocket_pubsub import PubSubEndpoint
from tweepy.asynchronous import AsyncStream
import uvicorn
import asyncio

from data import user_name, user_id
from util import get_tokens

model_path = "zoo:nlp/text_classification/bert-base/pytorch/huggingface/sst2/base-none"
text_classification = pipeline(task="text-classification", model_path=model_path)

app = FastAPI()
router = APIRouter()
endpoint = PubSubEndpoint()
endpoint.register_route(router)
app.include_router(router)
token = get_tokens()

class SparseStream(AsyncStream):

    async def on_status(self, status):
        
        # if (not status.retweeted) \
        # and ('RT @' not in status.text) \
        # and (status.in_reply_to_screen_name not in user_name) \
        # and (status.in_reply_to_status_id is None):

            inference = text_classification(status.text)[0]
            inference = "positive" if inference['label'] == 'LABEL_1' else "negative"
            await endpoint.publish(topics=["tweets"], data={"tweet": status.text, "inference": inference})
            print(status.text)

async def tweet_stream():

    stream = SparseStream(
            token["consumer_key"], 
            token["consumer_secret"], 
            token["access_token"], 
            token["access_token_secret"]
        )

    await stream.filter(follow=user_id, stall_warnings=True)

@app.get("/sparsestream")
async def http_trigger():
    asyncio.create_task(tweet_stream())

uvicorn.run(app, host="0.0.0.0", port=8000)