"""
DESCRIPTION:
    This sample demonstrates handling a client-executed `function` tool during
    a live voice-agent session:

    1) Create a voice agent configured with a `get_weather` function tool.
    2) Stream the default microphone and play replies on the default speaker.
    3) Listen for `response.function_call_arguments.done`, execute the function
       locally, and send the result back with `conversation.item.create` +
       `response.create` so the agent can finish its reply using the tool output.
    4) Keep listening across turns until Ctrl+C.
       Speaking during a reply interrupts it and discards unplayed audio.

USAGE:
    python example-voice-live-function-tool.py

    Before running the sample:

    pip install "azure-ai-projects[voice]>=2.7.0" azure-identity python-dotenv pyaudio

    Set these environment variables with your own values:
    1) FOUNDRY_PROJECT_ENDPOINT - The Azure AI Project endpoint.
    2) FOUNDRY_VOICE_AGENT_MODEL - Optional. The realtime model deployment name.
       Defaults to "gpt-realtime".
    3) FOUNDRY_VOICE_AGENT_NAME - Optional. Name for the sample voice agent
       created by this script. Defaults to
       "sample-voice-agent-function-tool".

    Requires a default microphone and speaker. Use headphones to avoid feedback.
    Audio is mono, 16-bit PCM at 24 kHz. Press Ctrl+C to stop.
"""

import json
import os
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import Final, List

import pyaudio
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    RealtimeConversationItemFunctionCallOutput,
    RealtimeAudioFormatsAudioPcm,
    RealtimeFunctionToolParameters,
    RealtimeServerEventError,
    RealtimeServerEventInputAudioBufferSpeechStarted,
    VoiceAgentDefinition,
    VoiceAgentFunctionTool,
    VoiceAgentAudioConfig,
    VoiceAgentAudioInputConfig,
    VoiceAgentAudioOutputConfig,
    VoiceAgentServerVadTurnDetection,
    RealtimeServerEventResponseAudioDelta,
    RealtimeServerEventResponseAudioTranscriptDone,
    RealtimeServerEventResponseCreated,
    RealtimeServerEventResponseDone,
    RealtimeServerEventResponseFunctionCallArgumentsDone,
    RealtimeServerEventResponseTextDone,
    VoiceModelType,
    VoiceOutputModality,
)


load_dotenv()

SAMPLE_RATE: Final = 24000
CHANNELS: Final = 1
CHUNK_FRAMES: Final = 2400  # 100 ms of 24 kHz mono audio
PLAYBACK_FRAMES: Final = 480  # 20 ms buffers keep local barge-in responsive.
_POLL_INTERVAL: Final = 0.1


@dataclass
class _AudioItem:
    item_id: str
    content_index: int
    chunks: deque[bytes] = field(default_factory=deque)
    offset: int = 0
    played_frames: int = 0


class AudioPlayback:
    """Share interruptible PCM buffers between the receiver and PortAudio callback."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._items: deque[_AudioItem] = deque()
        self._last_item: _AudioItem | None = None

    def add(self, item_id: str, content_index: int, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            item = self._items[-1] if self._items else self._last_item
            if item is None or (item.item_id, item.content_index) != (item_id, content_index):
                item = _AudioItem(item_id, content_index)
                self._items.append(item)
            elif not self._items:
                self._items.append(item)
            item.chunks.append(data)

    def render(self, _input, frame_count: int, _time_info, _status) -> tuple[bytes, int]:
        output = bytearray()
        bytes_per_frame = CHANNELS * 2
        required_bytes = frame_count * bytes_per_frame
        with self._lock:
            while self._items and len(output) < required_bytes:
                item = self._items[0]
                chunk = item.chunks[0]
                count = min(len(chunk) - item.offset, required_bytes - len(output))
                output.extend(chunk[item.offset:item.offset + count])
                item.offset += count
                item.played_frames += count // bytes_per_frame
                self._last_item = item
                if item.offset == len(chunk):
                    item.chunks.popleft()
                    item.offset = 0
                if not item.chunks:
                    self._items.popleft()
        output.extend(b"\x00" * (required_bytes - len(output)))
        return bytes(output), pyaudio.paContinue

    def interrupt(self) -> list[tuple[str, int, int]]:
        with self._lock:
            items = list(self._items)
            if self._last_item is not None and not any(item is self._last_item for item in items):
                items.insert(0, self._last_item)
            self._items.clear()
            self._last_item = None
            return [
                (item.item_id, item.content_index, item.played_frames * 1000 // SAMPLE_RATE)
                for item in items
            ]


def safe_print(text: str) -> None:
    """Print text that may contain characters the current console can't display.

    Model-generated replies and instructions can contain characters (curly quotes, em-dashes,
    etc.) outside some legacy, non-Unicode console encodings (for example when stdout is
    piped/redirected on Windows). Rather than crashing with UnicodeEncodeError, fall back to
    replacing just the unsupported characters; a real interactive UTF-8 console prints unaffected.

    :param text: The text to print.
    :type text: str
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


def get_weather(city: str) -> str:
    """A trivial local "tool" implementation the agent can call.

    :param city: The city to look up.
    :type city: str
    :return: A canned weather report for the city.
    :rtype: str
    """
    return json.dumps({"city": city, "condition": "sunny", "temperature_f": 72})


def stream_microphone(connection, microphone: pyaudio.Stream, stop_event: Event) -> None:
    while not stop_event.is_set():
        audio_chunk = microphone.read(CHUNK_FRAMES, exception_on_overflow=False)
        if not stop_event.is_set():
            connection.input_audio_buffer.append(audio=audio_chunk)


def _run_session_with_tool_support(client: AIProjectClient, agent_name: str) -> None:
    """Stream device audio and resolve tool calls until interrupted or a failure occurs."""
    with ExitStack() as stack:
        audio = pyaudio.PyAudio()
        stack.callback(audio.terminate)
        input_device = audio.get_default_input_device_info()
        output_device = audio.get_default_output_device_info()
        microphone = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=int(input_device["index"]),
            frames_per_buffer=CHUNK_FRAMES,
        )
        stack.callback(microphone.close)
        stack.callback(microphone.stop_stream)
        playback = AudioPlayback()
        speaker = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=int(output_device["index"]),
            frames_per_buffer=PLAYBACK_FRAMES,
            stream_callback=playback.render,
        )
        stack.callback(speaker.close)
        stack.callback(speaker.stop_stream)
        conn = stack.enter_context(client.beta.voice_agents.realtime.connect(agent_name=agent_name))
        stop_event = Event()

        safe_print(f"Listening on {input_device['name']}. Speak now (Ctrl+C to stop).")
        safe_print(f"Playing responses on {output_device['name']}.")
        pending_tool_outputs: dict[str, list[tuple[str, str]]] = {}
        active_response_id: str | None = None
        interrupted_responses: set[str] = set()
        with ThreadPoolExecutor(max_workers=1) as executor:
            workers: List[Future[None]] = []
            try:
                workers.append(executor.submit(stream_microphone, conn, microphone, stop_event))
                while True:
                    for worker in workers:
                        if worker.done():
                            worker.result()
                            raise RuntimeError("Audio stream ended unexpectedly.")
                    if not speaker.is_active():
                        raise RuntimeError("Speaker stream ended unexpectedly.")
                    try:
                        event = conn.recv(timeout=_POLL_INTERVAL)
                    except TimeoutError:
                        continue
                    if isinstance(event, RealtimeServerEventResponseCreated):
                        active_response_id = event.response.id
                    elif isinstance(event, RealtimeServerEventInputAudioBufferSpeechStarted):
                        if active_response_id is not None:
                            interrupted_responses.add(active_response_id)
                            pending_tool_outputs.pop(active_response_id, None)
                        # interrupt_response cancels generation on the server, not device playback.
                        for item_id, content_index, audio_end_ms in playback.interrupt():
                            conn.conversation.item.truncate(
                                item_id=item_id,
                                content_index=content_index,
                                audio_end_ms=audio_end_ms,
                            )
                    elif isinstance(event, RealtimeServerEventResponseAudioDelta):
                        if event.response_id in interrupted_responses:
                            continue
                        # The Foundry SDK has already base64-decoded the audio delta.
                        playback.add(event.item_id, event.content_index, event.delta)
                    elif isinstance(event, RealtimeServerEventResponseAudioTranscriptDone):
                        if event.response_id not in interrupted_responses:
                            safe_print(f"Agent: {event.transcript}")
                    elif isinstance(event, RealtimeServerEventResponseTextDone):
                        if event.response_id not in interrupted_responses:
                            safe_print(f"Agent: {event.text}")
                    elif isinstance(event, RealtimeServerEventResponseFunctionCallArgumentsDone):
                        if event.response_id in interrupted_responses:
                            continue
                        args = json.loads(event.arguments)
                        safe_print(f"Tool call: {event.name}({args})")
                        if event.name == "get_weather":
                            result = get_weather(**args)
                        else:
                            result = json.dumps({"error": f"Unknown tool: {event.name}"})
                        pending_tool_outputs.setdefault(event.response_id, []).append((event.call_id, result))
                    elif isinstance(event, RealtimeServerEventResponseDone):
                        response_id = event.response.id
                        tool_outputs = pending_tool_outputs.pop(response_id, [])
                        if active_response_id == response_id:
                            active_response_id = None
                        if response_id in interrupted_responses:
                            interrupted_responses.remove(response_id)
                            continue
                        if event.response.status == "failed":
                            raise RuntimeError(f"Agent response failed: {event.response.status_details}")
                        if event.response.status != "completed":
                            continue
                        # Wait for response.done before requesting the tool follow-up.
                        if tool_outputs:
                            for call_id, result in tool_outputs:
                                conn.conversation.item.create(
                                    item=RealtimeConversationItemFunctionCallOutput(call_id=call_id, output=result)
                                )
                            conn.response.create()
                    elif isinstance(event, RealtimeServerEventError):
                        raise RuntimeError(f"Session error: {event.error.message}")
            finally:
                stop_event.set()
                playback.interrupt()
                for worker in workers:
                    worker.result()


def main() -> None:
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    model = os.environ.get("FOUNDRY_VOICE_AGENT_MODEL") or "gpt-realtime"
    agent_name = os.environ.get("FOUNDRY_VOICE_AGENT_NAME") or "sample-voice-agent-function-tool"

    get_weather_tool = VoiceAgentFunctionTool(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=RealtimeFunctionToolParameters(
            {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, e.g. Seattle."}},
                "required": ["city"],
            }
        ),
    )

    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=endpoint, credential=credential, allow_preview=True) as project_client,
    ):
        created_version = None
        try:
            created_version = project_client.agents.create_version(
                agent_name=agent_name,
                definition=VoiceAgentDefinition(
                    model_type=VoiceModelType.MANAGED,
                    model=model,
                    instructions=(
                        "You are a helpful voice assistant. Use the get_weather tool when the "
                        "caller asks about the weather, then answer using its result."
                    ),
                    output_modalities=[VoiceOutputModality.AUDIO],
                    audio=VoiceAgentAudioConfig(
                        input=VoiceAgentAudioInputConfig(
                            format=RealtimeAudioFormatsAudioPcm(rate=24000),
                            turn_detection=VoiceAgentServerVadTurnDetection(
                                threshold=0.5,
                                prefix_padding_ms=300,
                                silence_duration_ms=300,
                                create_response=True,
                                interrupt_response=True,
                            ),
                        ),
                        output=VoiceAgentAudioOutputConfig(
                            format=RealtimeAudioFormatsAudioPcm(rate=24000),
                            voice="alloy",
                        ),
                    ),
                    tools=[get_weather_tool],
                ),
            )
            print(f"Created voice agent: {agent_name}")

            _run_session_with_tool_support(project_client, agent_name)
        finally:
            if created_version is not None:
                project_client.agents.delete_version(agent_name=agent_name, agent_version=created_version.version)
                print(f"Deleted voice agent version: {created_version.version}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")