"""Places outbound calls and collects artifacts.

    python runner.py --scenario 01-new-appointment
    python runner.py --all
"""

import argparse
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger
from twilio.rest import Client

from scenarios.scenarios import BY_ID, SCENARIOS

load_dotenv()

TARGET = os.environ["TARGET_NUMBER"]  # +18054398008 — the assessment line only
FROM = os.environ["TWILIO_FROM_NUMBER"]  # your single number, E.164
PUBLIC_HOST = os.environ["PUBLIC_HOST"]

client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
CALLS = Path("calls")
CALLS.mkdir(exist_ok=True)


def place(scenario_id: str):
    scenario = BY_ID[scenario_id]
    logger.info(f"calling {TARGET} — {scenario.name}")

    call = client.calls.create(
        to=TARGET,
        from_=FROM,
        url=f"https://{PUBLIC_HOST}/twiml?scenario={scenario_id}",
        # Dual channel puts each party on its own track, so the post-call
        # transcript gets speaker attribution from physics, not guesswork.
        record=True,
        recording_channels="dual",
        time_limit=scenario.max_seconds,
    )

    logger.info(f"call sid {call.sid}")
    return wait_and_save(call.sid, scenario_id)


def wait_and_save(call_sid: str, scenario_id: str, timeout: int = 420):
    deadline = time.time() + timeout
    while time.time() < deadline:
        call = client.calls(call_sid).fetch()
        if call.status in ("completed", "failed", "busy", "no-answer", "canceled"):
            logger.info(f"status {call.status}, duration {call.duration}s")
            break
        time.sleep(3)
    else:
        logger.error("timed out waiting for call to finish")
        return None

    if call.status != "completed":
        logger.error(f"call did not complete: {call.status}")
        return None

    # Recordings take a few seconds to become available after hangup.
    for _ in range(20):
        recordings = list(client.recordings.list(call_sid=call_sid, limit=1))
        if recordings:
            break
        time.sleep(3)
    else:
        logger.error("no recording appeared")
        return None

    rec = recordings[0]
    out = CALLS / f"{scenario_id}-{call_sid}.mp3"
    url = f"https://api.twilio.com{rec.uri.replace('.json', '.mp3')}"
    audio = requests.get(
        url, auth=(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    )
    out.write_bytes(audio.content)
    logger.info(f"saved {out} ({len(audio.content) // 1024} KB)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gap", type=int, default=20, help="seconds between calls")
    args = ap.parse_args()

    if args.all:
        for s in SCENARIOS:
            place(s.id)
            time.sleep(args.gap)
    elif args.scenario:
        place(args.scenario)
    else:
        ap.error("pass --scenario ID or --all")


if __name__ == "__main__":
    main()
