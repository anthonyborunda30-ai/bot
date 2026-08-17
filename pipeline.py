"""Pipecat pipeline: Twilio media stream <-> OpenAI Realtime.

Design notes (expand these in ARCHITECTURE.md):
- Speech-to-speech rather than STT->LLM->TTS. We are calling a *live* voice
  agent, so every 100ms of added latency makes both sides collide on turn
  boundaries. Realtime lands ~500ms end to end; a cascade realistically sits
  near 1s once endpointing, first-token and first-byte are stacked.
- Server-side VAD for turn detection. We are the caller, so we want to sound
  like a person who waits, not a benchmark runner that barges.
- Transcripts here are captured for live logging only. The graded transcript
  comes from a post-call pass over Twilio's dual-channel recording, where
  speaker attribution is physical rather than inferred.
"""

import os
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# NOTE: Pipecat moves fast and this import path has changed between releases.
# Pin the version in requirements.txt and check the current docs if it fails.
from pipecat.services.openai_realtime_beta import (
    InputAudioTranscription,
    OpenAIRealtimeBetaLLMService,
    SessionProperties,
    TurnDetection,
)

from scenarios.scenarios import Scenario


async def build_task(
    websocket,
    stream_sid: str,
    call_sid: str,
    scenario: Scenario,
    transcript_sink,
) -> PipelineTask:
    """Wire up one call. `transcript_sink` is an async callable (role, text)."""

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=os.environ["TWILIO_ACCOUNT_SID"],
        auth_token=os.environ["TWILIO_AUTH_TOKEN"],
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    session_properties = SessionProperties(
        input_audio_transcription=InputAudioTranscription(model="whisper-1"),
        # Server VAD keeps us polite. Raise silence_ms if we cut the agent off;
        # lower it if we leave dead air. This is the main knob for "natural".
        turn_detection=TurnDetection(
            type="server_vad",
            threshold=0.6,
            prefix_padding_ms=300,
            silence_duration_ms=700,
        ),
        instructions=scenario.system_prompt(),
        voice=os.getenv("REALTIME_VOICE", "sage"),
        temperature=0.8,
    )

    llm = OpenAIRealtimeBetaLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        session_properties=session_properties,
        start_audio_paused=False,
    )

    # Lets the model close the call itself once its objective is met, rather
    # than us guessing with a fixed timer.
    async def end_call(params):
        logger.info(f"[{scenario.id}] model requested end_call")
        await params.result_callback({"status": "ok"})
        await task.queue_frame(EndFrame())  # noqa: F821  (bound below)

    llm.register_function("end_call", end_call)

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": scenario.system_prompt()}],
        tools=[
            {
                "type": "function",
                "name": "end_call",
                "description": "Hang up. Call this only after saying goodbye.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)
    transcript = TranscriptProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
            transcript.user(),
            transcript.assistant(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transcript.event_handler("on_transcript_update")
    async def _on_transcript(processor, frame):
        for msg in frame.messages:
            await transcript_sink(msg.role, msg.content)
            logger.info(f"[{scenario.id}] {msg.role}: {msg.content}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(transport, client):
        # We are the caller, so we speak first — same as a real person would.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(transport, client):
        logger.info(f"[{scenario.id}] client disconnected")
        await task.cancel()

    return task


from pipecat.frames.frames import EndFrame  # noqa: E402  (needed by end_call)
