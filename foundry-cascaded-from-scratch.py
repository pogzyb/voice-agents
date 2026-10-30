import asyncio
import json
import os
import re
from urllib.parse import urlparse

from dotenv import load_dotenv
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import DefaultAzureCredential
from openai.types.responses import ResponseInputParam, ToolParam
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

SPEECH_SCOPE = "https://cognitiveservices.azure.com/.default"
TextChunk = tuple[int, str | None]


class TurnState:
    def __init__(self) -> None:
        self.generation = 0
        self.response_task: asyncio.Task[None] | None = None
        self.playback_task: asyncio.Task[None] | None = None

    def interrupt(self) -> None:
        # Called only on the asyncio loop, including from Speech SDK callbacks.
        self.generation += 1
        for task in (self.response_task, self.playback_task):
            if task is not None and not task.done() and not task.cancelling():
                task.cancel()


async def speech_authorization_token(credential: DefaultAzureCredential) -> str:
    token = await credential.get_token(SPEECH_SCOPE)
    return f"aad#{os.environ['COGNITIVE_SERVICES_RESOURCE_ID']}#{token.token}"


async def create_speech_config(
    credential: DefaultAzureCredential,
) -> speechsdk.SpeechConfig:
    parsed = urlparse(os.environ["COGNITIVE_SERVICES_ENDPOINT"])
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("COGNITIVE_SERVICES_ENDPOINT must be an HTTPS URL.")
    speech_config = speechsdk.SpeechConfig(
        endpoint=f"{parsed.scheme}://{parsed.netloc}"
    )
    speech_config.authorization_token = await speech_authorization_token(credential)
    return speech_config


async def text_to_speech(
    text_queue: asyncio.Queue[TextChunk],
    credential: DefaultAzureCredential,
    turns: TurnState,
) -> None:
    speech_config = await create_speech_config(credential)
    speech_config.speech_synthesis_voice_name = "en-US-Ethan:MAI-Voice-2.1-Flash"
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    async def synthesize(text: str) -> None:
        try:
            speech_synthesizer.authorization_token = await speech_authorization_token(credential)
            # Start on the loop before awaiting, so cancellation cannot stop an
            # operation that a delayed worker thread has not started yet.
            synthesis = speech_synthesizer.speak_text_async(text)
            result = await asyncio.to_thread(synthesis.get)
        except asyncio.CancelledError:
            stopped = speech_synthesizer.stop_speaking_async()
            await asyncio.to_thread(stopped.get)
            raise
        if not isinstance(result, speechsdk.SpeechSynthesisResult):
            raise RuntimeError("Speech synthesis returned no result.")
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            raise RuntimeError(
                f"Speech synthesis failed: {result.reason}; "
                f"{details.reason if details else ''}; "
                f"{details.error_details if details else ''}"
            )

    async def speak(text: str) -> bool:
        if pending_generation != turns.generation:
            return False
        if not text.strip():
            return True
        turns.playback_task = asyncio.create_task(synthesize(text))
        try:
            await turns.playback_task
            return pending_generation == turns.generation
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return False
        finally:
            turns.playback_task = None

    pending = ""
    pending_generation = turns.generation
    try:
        while True:
            generation, text = await text_queue.get()
            if pending_generation != turns.generation:
                pending = ""
                pending_generation = turns.generation
            if text is not None and generation != turns.generation:
                continue
            # None ends the pipeline; an empty string flushes the current response.
            if text is None or text == "":
                await speak(pending)
                pending = ""
                if text is None:
                    return
                continue
            pending += text
            # Speak phrases rather than individual tokens to preserve natural prosody.
            while match := re.search(r"[.!?](?:\s|$)|\n", pending):
                if not await speak(pending[:match.end()]):
                    pending = ""
                    break
                pending = pending[match.end():]
    finally:
        stopped = speech_synthesizer.stop_speaking_async()
        await asyncio.to_thread(stopped.get)


async def speech_to_text(
    recognized_queue: asyncio.Queue[str | None],
    credential: DefaultAzureCredential,
    turns: TurnState,
) -> None:
    speech_config = await create_speech_config(credential)
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_RecoModelName, "mai-transcribe-2"
    )
    speech_config.speech_recognition_language = "en-US"
    speech_config.set_property(
        speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
    )

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    loop = asyncio.get_running_loop()
    stopped: asyncio.Future[None] = loop.create_future()
    utterance_started = False

    def begin_utterance() -> None:
        nonlocal utterance_started
        if not utterance_started:
            utterance_started = True
            turns.interrupt()

    def speech_started(_event: speechsdk.RecognitionEventArgs) -> None:
        loop.call_soon_threadsafe(begin_utterance)

    def recognizing(event: speechsdk.SpeechRecognitionEventArgs) -> None:
        if event.result.text.strip():
            loop.call_soon_threadsafe(begin_utterance)

    def accept_utterance(text: str | None) -> None:
        nonlocal utterance_started
        if text is not None:
            begin_utterance()
            recognized_queue.put_nowait(text)
        utterance_started = False

    def recognized(event: speechsdk.SpeechRecognitionEventArgs) -> None:
        if event.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = event.result.text
            if text.strip():
                print(f"\n[User said]: {text}")
                # Speech SDK callbacks run on SDK threads, not the asyncio loop.
                loop.call_soon_threadsafe(accept_utterance, text)
            else:
                loop.call_soon_threadsafe(accept_utterance, None)
        elif event.result.reason == speechsdk.ResultReason.NoMatch:
            print("Speech could not be recognized; please try again.")
            loop.call_soon_threadsafe(accept_utterance, None)

    def finish(error: str | None = None) -> None:
        if not stopped.done():
            if error is not None:
                stopped.set_exception(RuntimeError(error))
            else:
                stopped.set_result(None)

    def canceled(event: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        details = event.cancellation_details
        loop.call_soon_threadsafe(
            finish, f"Speech recognition canceled: {details.reason}; {details.error_details}"
        )

    def session_stopped(_event: speechsdk.SessionEventArgs) -> None:
        loop.call_soon_threadsafe(finish)

    speech_recognizer.speech_start_detected.connect(speech_started)
    speech_recognizer.recognizing.connect(recognizing)
    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.canceled.connect(canceled)
    speech_recognizer.session_stopped.connect(session_stopped)
    try:
        await asyncio.to_thread(
            lambda: speech_recognizer.start_continuous_recognition_async().get()
        )
        print("Listening for speech... (Ctrl+C to stop)")
        while True:
            done, _ = await asyncio.wait({stopped}, timeout=540)
            if done:
                await stopped
                await recognized_queue.put(None)
                return
            speech_recognizer.authorization_token = await speech_authorization_token(credential)
    finally:
        speech_recognizer.speech_start_detected.disconnect_all()
        speech_recognizer.recognizing.disconnect_all()
        speech_recognizer.recognized.disconnect_all()
        speech_recognizer.canceled.disconnect_all()
        speech_recognizer.session_stopped.disconnect_all()
        await asyncio.to_thread(
            lambda: speech_recognizer.stop_continuous_recognition_async().get()
        )


def tool_get_weather(city: str):
    if city.lower() == "detroit":
        return "The weather in Detroit is currently cold, but sunny with a temperature of 33°F."
    return f"The weather in {city} is currently raining with a temperature of 54°F."


async def llm_reasoning(
    recognized_queue: asyncio.Queue[str | None],
    text_queue: asyncio.Queue[TextChunk],
    credential: DefaultAzureCredential,
    turns: TurnState,
) -> None:
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    deployment_name = os.getenv("FOUNDRY_MODEL_DEPLOYMENT", "gpt-6-luna")
    tools: list[ToolParam] = [{
        "type": "function",
        "name": "tool_get_weather",
        "description": "Get example weather for a city (demo data, not live weather).",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }]
    kws = {}
    async with (
        AIProjectClient(endpoint=endpoint, credential=credential) as client,
        client.get_openai_client() as openai_client,
    ):
        async def respond(text: str, generation: int) -> None:
            nonlocal kws
            response_input: ResponseInputParam = [{"role": "user", "content": text}]
            while True:
                stream = await openai_client.responses.create(
                    model=deployment_name,
                    instructions=(
                        "You are a helpful voice assistant. Keep responses concise and "
                        "conversational. Avoid markdown. Weather tool data is illustrative. "
                    ),
                    input=response_input,
                    tools=tools,
                    stream=True,
                    **kws
                )
                response = None
                async with stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            print(event.delta, end="", flush=True)
                            await text_queue.put((generation, event.delta))
                        elif event.type == "response.refusal.delta":
                            print(event.delta, end="", flush=True)
                            await text_queue.put((generation, event.delta))
                        elif event.type == "response.completed":
                            response = event.response
                        elif event.type == "response.failed" or event.type == "response.incomplete":
                            raise RuntimeError(f"LLM response failed: {event.response}")
                        elif event.type == "error":
                            raise RuntimeError(f"LLM streaming error: {event.message}")
                if response is None:
                    raise RuntimeError("LLM stream ended without a completed response.")
                kws["previous_response_id"] = response.id
                response_input = []
                for item in response.output:
                    if item.type != "function_call":
                        continue
                    print(f"response item: {item}")
                    if item.name != "tool_get_weather":
                        raise ValueError(f"Unknown tool requested: {item.name}")
                    arguments = json.loads(item.arguments)
                    if (
                        not isinstance(arguments, dict)
                        or set(arguments) != {"city"}
                        or not isinstance(arguments["city"], str)
                        or not arguments["city"].strip()
                    ):
                        raise ValueError("tool_get_weather requires a nonempty city string.")
                    response_input.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_get_weather(arguments["city"]),
                    })
                if not response_input:
                    break
            kws["previous_response_id"] = response.id
            print()
            await text_queue.put((generation, ""))

        while True:
            text = await recognized_queue.get()
            if text is None:
                await text_queue.put((turns.generation, None))
                return
            turns.response_task = asyncio.create_task(respond(text, turns.generation))
            try:
                await turns.response_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                print("\n[Assistant interrupted]")
            finally:
                turns.response_task = None


async def main() -> None:
    recognized_queue: asyncio.Queue[str | None] = asyncio.Queue()
    text_queue: asyncio.Queue[TextChunk] = asyncio.Queue()
    turns = TurnState()
    async with DefaultAzureCredential() as credential:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(speech_to_text(recognized_queue, credential, turns))
            tasks.create_task(llm_reasoning(recognized_queue, text_queue, credential, turns))
            tasks.create_task(text_to_speech(text_queue, credential, turns))


if __name__ == "__main__":
    asyncio.run(main())