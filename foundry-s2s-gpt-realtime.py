import os
import base64
import asyncio
import json
from contextlib import suppress

import pyaudio
from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_FRAMES = 2400  # 100 ms of 24 kHz mono audio


async def stream_microphone(connection, microphone, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        audio_chunk = await asyncio.to_thread(
            microphone.read,
            CHUNK_FRAMES,
            exception_on_overflow=False,
        )
        if stop_event.is_set():
            break

        await connection.input_audio_buffer.append(
            audio=base64.b64encode(audio_chunk).decode("ascii")
        )


async def receive_response(connection: AsyncRealtimeConnection, speaker) -> None:
    async for event in connection:
        if event.type == "response.output_audio.delta":
            await asyncio.to_thread(speaker.write, base64.b64decode(event.delta))
        elif event.type == "response.output_audio_transcript.delta":
            print(event.delta, end="", flush=True)
        elif event.type == "conversation.item.input_audio_transcription.completed":
            print(f"\n[User said]: {event.transcript}")
        elif event.type == "error":
            raise RuntimeError(f"Realtime API error: {event.error.message}")
        elif event.type == "response.done":
            if event.response.status != "completed":
                continue
            tool_outputs_sent = False
            for item in event.response.output or []:
                if item.type != "function_call":
                    continue
                if item.name != "tool_get_weather":
                    raise ValueError(f"Unknown tool requested: {item.name}")
                if not item.call_id:
                    raise ValueError("tool_get_weather call is missing a call_id.")
                arguments = json.loads(item.arguments)
                if (
                    not isinstance(arguments, dict)
                    or set(arguments) != {"city"}
                    or not isinstance(arguments["city"], str)
                    or not arguments["city"].strip()
                ):
                    raise ValueError("tool_get_weather requires a nonempty city string.")
                await connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_get_weather(arguments["city"]),
                    }
                )
                tool_outputs_sent = True
            if tool_outputs_sent:
                await connection.response.create()


def tool_get_weather(city: str):
    if city.lower() == "detroit":
        return "The weather in Detroit is currently cold, but sunny with a temperature of 33°F."
    return f"The weather in {city} is currently raining with a temperature of 54°F."


async def main() -> None:
    """
    Stream audio from the default microphone to the Realtime API and receive
    an audio response.
    """
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://ai.azure.com/.default"
    )
    token = token_provider()

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
    base_url = endpoint.replace("https://", "wss://").rstrip("/")

    client = AsyncOpenAI(websocket_base_url=base_url, api_key=token)
    audio = pyaudio.PyAudio()
    microphone = None
    speaker = None

    try:
        default_input_device = audio.get_default_input_device_info()
        microphone = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=int(default_input_device["index"]),
            frames_per_buffer=CHUNK_FRAMES,
        )
        default_output_device = audio.get_default_output_device_info()
        speaker = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            output_device_index=int(default_output_device["index"]),
            frames_per_buffer=CHUNK_FRAMES,
        )

        async with client.realtime.connect(model=deployment_name) as connection:
            # Configure the session for audio input and output
            await connection.session.update(
                session={
                    "type": "realtime",
                    "instructions": (
                        "You are a helpful assistant. Respond conversationally. "
                        "Use tool_get_weather for weather questions. "
                        "Weather tool data is illustrative, not live weather."
                    ),
                    "tools": [{
                        "type": "function",
                        "name": "tool_get_weather",
                        "description": "Get example weather for a city (demo data, not live weather).",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    }],
                    "tool_choice": "auto",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "transcription": {"model": "whisper-1"},
                            "format": {
                                "type": "audio/pcm",
                                "rate": SAMPLE_RATE,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                                "create_response": True,
                                "interrupt_response": False,
                            },
                        },
                        "output": {
                            "voice": "alloy",
                            "format": {
                                "type": "audio/pcm",
                                "rate": SAMPLE_RATE,
                            },
                        },
                    },
                }
            )

            print(f"Listening on {default_input_device['name']}. Speak now (Ctrl+C to stop).")
            print(f"Playing responses on {default_output_device['name']}.")
            
            stop_event = asyncio.Event()
            microphone_task = asyncio.create_task(
                stream_microphone(connection, microphone, stop_event)
            )
            response_task = asyncio.create_task(receive_response(connection, speaker))

            try:
                done, _ = await asyncio.wait(
                    {microphone_task, response_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if microphone_task in done:
                    await microphone_task
                    raise RuntimeError("Microphone stream ended unexpectedly.")

                await response_task
            finally:
                stop_event.set()
                await microphone_task
                if not response_task.done():
                    response_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await response_task

    finally:
        if microphone is not None:
            microphone.stop_stream()
            microphone.close()
        if speaker is not None:
            speaker.stop_stream()
            speaker.close()
            
        audio.terminate()
        credential.close()


if __name__ == "__main__":
    asyncio.run(main())