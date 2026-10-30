import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import azure.cognitiveservices.speech as speechsdk


spec = importlib.util.spec_from_file_location(
    "foundry_cascaded", Path(__file__).resolve().parents[1] / "foundry-cascaded.py"
)
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            yield event


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def completed(response_id, output=()):
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(id=response_id, output=list(output)),
    )


class CascadedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "COGNITIVE_SERVICES_ENDPOINT": "https://speech.example.com/path",
            "COGNITIVE_SERVICES_RESOURCE_ID": "/subscriptions/test/resource",
            "FOUNDRY_PROJECT_ENDPOINT": "https://project.example.com",
            "FOUNDRY_MODEL_DEPLOYMENT": "test-deployment",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.credential = SimpleNamespace(
            get_token=AsyncMock(return_value=SimpleNamespace(token="test-token"))
        )

    def project_client(self, streams):
        client = MagicMock()
        client.__aenter__.return_value = client
        openai = MagicMock()
        openai.__aenter__.return_value = openai
        openai.responses.create = AsyncMock(side_effect=streams)
        client.get_openai_client.return_value = openai
        return client, openai

    async def test_recognition_callbacks_cross_threads_and_stop_cleanly(self):
        queue = asyncio.Queue()
        recognizer = MagicMock()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        recognizer.start_continuous_recognition_async.return_value.get.side_effect = (
            lambda: loop.call_soon_threadsafe(started.set)
        )
        with patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer):
            task = asyncio.create_task(agent.speech_to_text(queue, self.credential))
            await asyncio.wait_for(started.wait(), 2)
            callback = recognizer.recognized.connect.call_args.args[0]
            for reason, text in [
                (speechsdk.ResultReason.RecognizingSpeech, "partial"),
                (speechsdk.ResultReason.RecognizedSpeech, "   "),
                (speechsdk.ResultReason.RecognizedSpeech, "Hello"),
                (speechsdk.ResultReason.RecognizedSpeech, "Again"),
            ]:
                await asyncio.to_thread(
                    callback, SimpleNamespace(result=SimpleNamespace(reason=reason, text=text))
                )
            await asyncio.to_thread(
                recognizer.session_stopped.connect.call_args.args[0], SimpleNamespace()
            )
            await asyncio.wait_for(task, 2)
        self.assertEqual([queue.get_nowait() for _ in range(3)], ["Hello", "Again", None])
        self.assertTrue(queue.empty())
        recognizer.stop_continuous_recognition_async.assert_called_once()
        recognizer.recognized.disconnect_all.assert_called_once()

    async def test_recognition_error_is_propagated(self):
        recognizer = MagicMock()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        recognizer.start_continuous_recognition_async.return_value.get.side_effect = (
            lambda: loop.call_soon_threadsafe(started.set)
        )
        with patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer):
            task = asyncio.create_task(agent.speech_to_text(asyncio.Queue(), self.credential))
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.to_thread(
                recognizer.canceled.connect.call_args.args[0],
                SimpleNamespace(cancellation_details=SimpleNamespace(
                    reason="Error", error_details="microphone failure"
                )),
            )
            with self.assertRaisesRegex(RuntimeError, "microphone failure"):
                await asyncio.wait_for(task, 2)
        recognizer.stop_continuous_recognition_async.assert_called_once()

    async def test_cancellation_stops_recognition_and_playback(self):
        recognizer, synthesizer = MagicMock(), MagicMock()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        recognizer.start_continuous_recognition_async.return_value.get.side_effect = (
            lambda: loop.call_soon_threadsafe(started.set)
        )
        with (
            patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer),
            patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
        ):
            recognition = asyncio.create_task(
                agent.speech_to_text(asyncio.Queue(), self.credential)
            )
            playback = asyncio.create_task(
                agent.text_to_speech(asyncio.Queue(), self.credential)
            )
            await asyncio.wait_for(started.wait(), 2)
            recognition.cancel()
            playback.cancel()
            for task in (recognition, playback):
                with self.assertRaises(asyncio.CancelledError):
                    await task
        recognizer.stop_continuous_recognition_async.assert_called_once()
        synthesizer.stop_speaking_async.assert_called_once()

    async def test_llm_streams_deltas_and_preserves_history(self):
        streams = [
            FakeStream([delta("Hello"), delta(" there."), completed("r1")]),
            FakeStream([delta("Second reply"), completed("r2")]),
        ]
        client, openai = self.project_client(streams)
        inputs, outputs = asyncio.Queue(), asyncio.Queue()
        for text in ("First turn", "Second turn", None):
            inputs.put_nowait(text)
        with patch.object(agent, "AIProjectClient", return_value=client):
            await agent.llm_reasoning(inputs, outputs, self.credential)
        self.assertEqual(
            [outputs.get_nowait() for _ in range(6)],
            ["Hello", " there.", "", "Second reply", "", None],
        )
        calls = openai.responses.create.call_args_list
        self.assertEqual(calls[0].kwargs["model"], "test-deployment")
        self.assertTrue(calls[0].kwargs["stream"])
        self.assertIsNone(calls[0].kwargs["previous_response_id"])
        self.assertEqual(calls[1].kwargs["previous_response_id"], "r1")
        self.assertTrue(all(stream.closed for stream in streams))

    async def test_llm_executes_weather_tool_and_continues_streaming(self):
        tool = SimpleNamespace(
            type="function_call", name="tool_get_weather",
            arguments='{"city":"Detroit"}', call_id="weather1",
        )
        client, openai = self.project_client([
            FakeStream([completed("r1", [tool])]),
            FakeStream([delta("It is cold."), completed("r2")]),
        ])
        inputs, outputs = asyncio.Queue(), asyncio.Queue()
        inputs.put_nowait("Weather in Detroit?")
        inputs.put_nowait(None)
        with patch.object(agent, "AIProjectClient", return_value=client):
            await agent.llm_reasoning(inputs, outputs, self.credential)
        followup = openai.responses.create.call_args_list[1].kwargs
        self.assertEqual(followup["previous_response_id"], "r1")
        self.assertEqual(followup["input"], [{
            "type": "function_call_output",
            "call_id": "weather1",
            "output": agent.tool_get_weather("Detroit"),
        }])
        self.assertEqual(outputs.get_nowait(), "It is cold.")

    async def test_llm_errors_and_truncated_streams_are_not_silent(self):
        for events, message in [
            ([SimpleNamespace(type="error", message="service unavailable")], "service unavailable"),
            ([SimpleNamespace(type="response.failed", response="failed")], "LLM response failed"),
            ([SimpleNamespace(type="response.incomplete", response="limit")], "LLM response failed"),
            ([delta("partial")], "without a completed response"),
        ]:
            with self.subTest(message=message):
                stream = FakeStream(events)
                client, _ = self.project_client([stream])
                inputs = asyncio.Queue()
                inputs.put_nowait("Hello")
                with patch.object(agent, "AIProjectClient", return_value=client):
                    with self.assertRaisesRegex(RuntimeError, message):
                        await agent.llm_reasoning(inputs, asyncio.Queue(), self.credential)
                self.assertTrue(stream.closed)

    async def test_tts_speaks_before_response_ends_and_flushes_tail(self):
        queue = asyncio.Queue()
        synthesizer = MagicMock()
        result = MagicMock(spec=speechsdk.SpeechSynthesisResult)
        result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        synthesizer.speak_text_async.return_value.get.return_value = result
        spoken = asyncio.Event()
        loop = asyncio.get_running_loop()
        def synthesize(text):
            loop.call_soon_threadsafe(spoken.set)
            return MagicMock(get=MagicMock(return_value=result))

        synthesizer.speak_text_async.side_effect = synthesize
        with patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer):
            task = asyncio.create_task(agent.text_to_speech(queue, self.credential))
            queue.put_nowait("Hel")
            queue.put_nowait("lo. ")
            await asyncio.wait_for(spoken.wait(), 2)
            self.assertFalse(task.done())
            for text in ("Tail without punctuation", "", "Next response", None):
                queue.put_nowait(text)
            await asyncio.wait_for(task, 2)
        self.assertEqual(
            [call.args[0] for call in synthesizer.speak_text_async.call_args_list],
            ["Hello. ", "Tail without punctuation", "Next response"],
        )
        synthesizer.stop_speaking_async.assert_called_once()

    async def test_refusal_is_spoken(self):
        client, _ = self.project_client([FakeStream([
            SimpleNamespace(type="response.refusal.delta", delta="I cannot help with that."),
            completed("r1"),
        ])])
        inputs, outputs = asyncio.Queue(), asyncio.Queue()
        inputs.put_nowait("Request")
        inputs.put_nowait(None)
        with patch.object(agent, "AIProjectClient", return_value=client):
            await agent.llm_reasoning(inputs, outputs, self.credential)
        self.assertEqual(outputs.get_nowait(), "I cannot help with that.")

    async def test_main_drains_multiple_turns_through_all_stages(self):
        recognizer, synthesizer = MagicMock(), MagicMock()
        result = MagicMock(spec=speechsdk.SpeechSynthesisResult)
        result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
        synthesizer.speak_text_async.return_value.get.return_value = result

        def recognize():
            callback = recognizer.recognized.connect.call_args.args[0]
            for text in ("Hello", "How are you?"):
                callback(SimpleNamespace(result=SimpleNamespace(
                    reason=speechsdk.ResultReason.RecognizedSpeech, text=text
                )))
            recognizer.session_stopped.connect.call_args.args[0](SimpleNamespace())

        recognizer.start_continuous_recognition_async.return_value.get.side_effect = recognize
        client, _ = self.project_client([
            FakeStream([delta("Hello"), completed("r1")]),
            FakeStream([delta("Doing well"), completed("r2")]),
        ])
        credential = MagicMock()
        credential.get_token = self.credential.get_token
        credential.__aenter__.return_value = credential
        with (
            patch.object(agent, "DefaultAzureCredential", return_value=credential),
            patch.object(agent, "AIProjectClient", return_value=client),
            patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer),
            patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
        ):
            await asyncio.wait_for(agent.main(), 2)
        self.assertEqual(
            [call.args[0] for call in synthesizer.speak_text_async.call_args_list],
            ["Hello", "Doing well"],
        )
        recognizer.stop_continuous_recognition_async.assert_called_once()
        synthesizer.stop_speaking_async.assert_called_once()
        credential.__aexit__.assert_awaited_once()

    async def test_tts_failure_is_propagated_and_stops_playback(self):
        synthesizer = MagicMock()
        result = MagicMock(spec=speechsdk.SpeechSynthesisResult)
        result.reason = speechsdk.ResultReason.Canceled
        result.cancellation_details.reason = speechsdk.CancellationReason.Error
        result.cancellation_details.error_details = "speaker failure"
        synthesizer.speak_text_async.return_value.get.return_value = result
        queue = asyncio.Queue()
        queue.put_nowait("Hello.")
        with patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer):
            with self.assertRaisesRegex(RuntimeError, "speaker failure"):
                await agent.text_to_speech(queue, self.credential)
        synthesizer.stop_speaking_async.assert_called_once()

    async def test_main_cancels_other_stages_on_failure(self):
        stopped = []
        both_started = asyncio.Event()
        started = 0

        async def waiting(*args):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

        async def failing(*args):
            await both_started.wait()
            raise RuntimeError("stage failed")

        credential = MagicMock()
        with (
            patch.object(agent, "DefaultAzureCredential", return_value=credential),
            patch.object(agent, "speech_to_text", side_effect=waiting),
            patch.object(agent, "llm_reasoning", side_effect=failing),
            patch.object(agent, "text_to_speech", side_effect=waiting),
        ):
            with self.assertRaises(ExceptionGroup) as error:
                await asyncio.wait_for(agent.main(), 2)
        self.assertIn("stage failed", str(error.exception.exceptions[0]))
        self.assertEqual(len(stopped), 2)
        credential.__aexit__.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
