# server.py
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from graph import agent
from langchain_core.messages import BaseMessage

app = FastAPI()


# -----------------------------
# Serialization for LangChain messages
# -----------------------------
def serialize_recursive(obj):
    if isinstance(obj, BaseMessage):
        return {
            "type": obj.type,
            "content": obj.content
        }

    if isinstance(obj, dict):
        return {k: serialize_recursive(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize_recursive(i) for i in obj]

    return obj


# -----------------------------
# Streaming generator
# -----------------------------
async def event_stream(query: str):
    inputs = {"messages": [HumanMessage(content=query)]}

    async for event in agent.astream(inputs, stream_mode="values"):
        safe = serialize_recursive(event)
        yield f"data: {json.dumps(safe)}\n\n"

    yield "data: [DONE]\n\n"


# -----------------------------
# FastAPI endpoint
# -----------------------------
@app.get("/chat")
async def chat(query: str):
    return StreamingResponse(
        event_stream(query),
        media_type="text/event-stream"
    )
