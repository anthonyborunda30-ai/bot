"""FastAPI app that Twilio connects to.

Flow:
  runner.py --REST--> Twilio --places call--> +1-805-439-8008
  Twilio --GET /twiml?scenario=ID--> we return <Connect><Stream>
  Twilio --WS /ws--> media frames both directions, Pipecat handles the call
"""

import json
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner

from bot.pipeline import build_task
from scenarios.scenarios import BY_ID

load_dotenv()

app = FastAPI()
CALLS_DIR = Path("calls")
CALLS_DIR.mkdir(exist_ok=True)


@app.get("/twiml")
async def twiml(scenario: str):
    """Twilio fetches this when the callee answers."""
    public_host = os.environ["PUBLIC_HOST"]  # e.g. abc123.ngrok.app
    return HTMLResponse(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{public_host}/ws">
      <Parameter name="scenario" value="{scenario}"/>
    </Stream>
  </Connect>
</Response>""",
        media_type="application/xml",
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    # Twilio sends "connected" then "start"; the start frame carries our
    # custom parameters and the stream/call SIDs.
    await websocket.receive_text()
    start = json.loads(await websocket.receive_text())["start"]
    stream_sid = start["streamSid"]
    call_sid = start["callSid"]
    scenario_id = start["customParameters"]["scenario"]
    scenario = BY_ID[scenario_id]

    logger.info(f"call {call_sid} started — scenario {scenario_id}")

    # Live transcript, written as we go so a crashed call still leaves evidence.
    live_path = CALLS_DIR / f"{scenario_id}-{call_sid}.live.jsonl"

    async def sink(role: str, text: str):
        with live_path.open("a") as f:
            f.write(json.dumps({"role": role, "text": text}) + "\n")

    task = await build_task(websocket, stream_sid, call_sid, scenario, sink)
    await PipelineRunner(handle_sigint=False).run(task)
    logger.info(f"call {call_sid} finished")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8765)))
