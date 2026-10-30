import asyncio
import importlib.util
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import azure.cognitiveservices.speech as speechsdk


spec = importlib.util.spec_from_file_location(
    "foundry_cascaded_turn_detection",
    Path(__file__).resolve().parents[1] / "foundry-cascaded-turn-detection.py",
)
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def completed(response_id, output=()):
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(id=response_id, output=list(output)),
    )


class FakeStream:
    def __init__(self, events, block=False):
        self.events = events
        self.block = block
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()


class TurnDetectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            "COGNITIVE_SERVICES_ENDPOINT": "https://speech.example.com",
            "COGNITIVE_SERVICES_RESOURCE_ID": "/subscriptions/test/resource",
            "FOUNDRY_PROJECT_ENDPOINT": "https://project.example.com",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.credential = SimpleNamespace(
            get_token=AsyncMock(return_value=SimpleNamespace(token="test-token"))
        )
        self.turns = agent.TurnState()

    def project_client(self, streams):
        client = MagicMock()
        client.__aenter__.return_value = client
        openai = MagicMock()
        openai.__aenter__.return_value = openai
        openai.responses.create = AsyncMock(side_effect=streams)
        client.get_openai_client.return_value = openai
        return client, openai

    def synthesis_result(self, reason=speechsdk.ResultReason.SynthesizingAudioCompleted):
        result = MagicMock(spec=speechsdk.SpeechSynthesisResult)
        result.reason = reason
        return result

    async def test_endpointing_and_early_speech_detection_from_sdk_threads(self):
        recognizer = MagicMock()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        recognizer.start_continuous_recognition_async.return_value.get.side_effect = (
            lambda: loop.call_soon_threadsafe(started.set)
        )
        inputs = asyncio.Queue()
        with (
            patch.object(
                agent, "create_speech_config", AsyncMock(return_value=MagicMock())
            ) as config,
            patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer),
        ):
            task = asyncio.create_task(
                agent.speech_to_text(inputs, self.credential, self.turns)
            )
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(started.wait(), 2)
            config.return_value.set_property.assert_any_call(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
            )
            response = asyncio.create_task(asyncio.Event().wait())
            playback = asyncio.create_task(asyncio.Event().wait())
            self.turns.response_task = response
            self.turns.playback_task = playback
            for signal, event in [
                (recognizer.speech_start_detected, SimpleNamespace()),
                (recognizer.recognizing, SimpleNamespace(
                    result=SimpleNamespace(text="Wait")
                )),
            ]:
                await asyncio.to_thread(signal.connect.call_args.args[0], event)
            for active in (response, playback):
                with self.assertRaises(asyncio.CancelledError):
                    await active
            self.assertTrue(inputs.empty(), "Interruption must precede the final transcript")
            self.assertEqual(self.turns.generation, 1)
            await asyncio.to_thread(
                recognizer.recognized.connect.call_args.args[0],
                SimpleNamespace(result=SimpleNamespace(
                    reason=speechsdk.ResultReason.RecognizedSpeech, text="Wait"
                )),
            )
            self.assertEqual(await asyncio.wait_for(inputs.get(), 2), "Wait")
            self.assertEqual(self.turns.generation, 1)
            await asyncio.to_thread(
                recognizer.recognizing.connect.call_args.args[0],
                SimpleNamespace(result=SimpleNamespace(text="Next")),
            )
            await asyncio.sleep(0)
            self.assertEqual(self.turns.generation, 2)
            await asyncio.to_thread(
                recognizer.session_stopped.connect.call_args.args[0], SimpleNamespace()
            )
            await asyncio.wait_for(task, 2)
        self.assertIsNone(inputs.get_nowait())
        recognizer.speech_start_detected.disconnect_all.assert_called_once()
        recognizer.recognizing.disconnect_all.assert_called_once()

    async def cancel_task(self, task):
        if not task.done():
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_interrupt_stops_native_audio_and_discards_queued_and_phrase_tails(self):
        synthesizer = MagicMock()
        started = asyncio.Event()
        released = threading.Event()
        loop = asyncio.get_running_loop()
        success = self.synthesis_result()

        def synthesize(text):
            if text == "Old sentence. ":
                loop.call_soon(started.set)

                def wait_for_stop():
                    if not released.wait(2):
                        raise TimeoutError("Playback was not stopped on interruption")
                    return self.synthesis_result(speechsdk.ResultReason.Canceled)

                return SimpleNamespace(get=wait_for_stop)
            return SimpleNamespace(get=lambda: success)

        synthesizer.speak_text_async.side_effect = synthesize
        synthesizer.stop_speaking_async.return_value.get.side_effect = released.set
        outputs = asyncio.Queue()
        outputs.put_nowait((0, "Old sentence. Old buffered tail."))
        outputs.put_nowait((0, "Old queued sentence."))
        with patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer):
            task = asyncio.create_task(
                agent.text_to_speech(outputs, self.credential, self.turns)
            )
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(started.wait(), 2)
            self.turns.interrupt()
            outputs.put_nowait((0, "Late stale delta."))
            outputs.put_nowait((1, "New response"))
            outputs.put_nowait((1, ""))
            outputs.put_nowait((1, None))
            await asyncio.wait_for(task, 2)
        self.assertTrue(released.is_set())
        self.assertEqual(
            [call.args[0] for call in synthesizer.speak_text_async.call_args_list],
            ["Old sentence. ", "New response"],
        )

    async def test_interrupt_discards_unspoken_partial_sentence_on_shutdown(self):
        synthesizer = MagicMock()
        outputs = asyncio.Queue()
        outputs.put_nowait((0, "No punctuation yet"))
        with patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer):
            task = asyncio.create_task(
                agent.text_to_speech(outputs, self.credential, self.turns)
            )
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.sleep(0)
            self.turns.interrupt()
            outputs.put_nowait((0, None))
            await asyncio.wait_for(task, 2)
        synthesizer.speak_text_async.assert_not_called()

    async def test_interrupt_closes_llm_stream_and_resumes_with_completed_history(self):
        streams = [
            FakeStream([delta("First."), completed("r1")]),
            FakeStream([delta("Interrupted text.")], block=True),
            FakeStream([delta("New reply."), completed("r3")]),
        ]
        client, openai = self.project_client(streams)
        inputs, outputs = asyncio.Queue(), asyncio.Queue()
        inputs.put_nowait("First")
        inputs.put_nowait("Second")
        with patch.object(agent, "AIProjectClient", return_value=client):
            task = asyncio.create_task(
                agent.llm_reasoning(inputs, outputs, self.credential, self.turns)
            )
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(streams[1].waiting.wait(), 2)
            self.turns.interrupt()
            inputs.put_nowait("New question")
            inputs.put_nowait(None)
            await asyncio.wait_for(task, 2)
        self.assertTrue(all(stream.closed for stream in streams))
        calls = openai.responses.create.call_args_list
        self.assertEqual(calls[1].kwargs["previous_response_id"], "r1")
        self.assertEqual(calls[2].kwargs["previous_response_id"], "r1")
        self.assertEqual(calls[2].kwargs["input"], [
            {"role": "user", "content": "New question"}
        ])
        self.assertEqual(
            [outputs.get_nowait() for _ in range(outputs.qsize())],
            [(0, "First."), (0, ""), (0, "Interrupted text."),
             (1, "New reply."), (1, ""), (1, None)],
        )

    async def test_interrupted_tool_followup_does_not_leave_unanswered_tool_in_history(self):
        tool = SimpleNamespace(
            type="function_call", name="tool_get_weather",
            arguments='{"city":"Detroit"}', call_id="weather1",
        )
        streams = [
            FakeStream([completed("tool_response", [tool])]),
            FakeStream([], block=True),
            FakeStream([delta("New reply."), completed("new_response")]),
        ]
        client, openai = self.project_client(streams)
        inputs = asyncio.Queue()
        inputs.put_nowait("Weather?")
        with patch.object(agent, "AIProjectClient", return_value=client):
            task = asyncio.create_task(
                agent.llm_reasoning(inputs, asyncio.Queue(), self.credential, self.turns)
            )
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(streams[1].waiting.wait(), 2)
            self.turns.interrupt()
            inputs.put_nowait("Never mind")
            inputs.put_nowait(None)
            await asyncio.wait_for(task, 2)
        calls = openai.responses.create.call_args_list
        self.assertEqual(calls[1].kwargs["previous_response_id"], "tool_response")
        self.assertEqual(calls[1].kwargs["input"][0]["call_id"], "weather1")
        self.assertIsNone(calls[2].kwargs["previous_response_id"])

    async def test_service_errors_still_propagate(self):
        client, _ = self.project_client([
            FakeStream([SimpleNamespace(type="error", message="LLM unavailable")])
        ])
        inputs = asyncio.Queue()
        inputs.put_nowait("Hello")
        with patch.object(agent, "AIProjectClient", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "LLM unavailable"):
                await agent.llm_reasoning(
                    inputs, asyncio.Queue(), self.credential, self.turns
                )
        synthesizer = MagicMock()
        result = self.synthesis_result(speechsdk.ResultReason.Canceled)
        result.cancellation_details.error_details = "speaker failure"
        synthesizer.speak_text_async.return_value.get.return_value = result
        outputs = asyncio.Queue()
        outputs.put_nowait((0, "Hello."))
        with patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer):
            with self.assertRaisesRegex(RuntimeError, "speaker failure"):
                await agent.text_to_speech(outputs, self.credential, self.turns)

    async def test_pipeline_cancellation_is_not_swallowed_as_barge_in(self):
        stream = FakeStream([], block=True)
        client, _ = self.project_client([stream])
        inputs = asyncio.Queue()
        inputs.put_nowait("Hello")
        with patch.object(agent, "AIProjectClient", return_value=client):
            task = asyncio.create_task(
                agent.llm_reasoning(inputs, asyncio.Queue(), self.credential, self.turns)
            )
            await asyncio.wait_for(stream.waiting.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertTrue(stream.closed)

    async def test_normal_tool_reply_and_following_turn_keep_context(self):
        tool = SimpleNamespace(
            type="function_call", name="tool_get_weather",
            arguments='{"city":"Detroit"}', call_id="weather1",
        )
        streams = [
            FakeStream([completed("tool", [tool])]),
            FakeStream([delta("It is cold."), completed("answer")]),
            FakeStream([delta("You're welcome."), completed("thanks")]),
        ]
        client, openai = self.project_client(streams)
        inputs, outputs = asyncio.Queue(), asyncio.Queue()
        for text in ("Weather?", "Thanks", None):
            inputs.put_nowait(text)
        with patch.object(agent, "AIProjectClient", return_value=client):
            await agent.llm_reasoning(inputs, outputs, self.credential, self.turns)
        self.assertEqual(
            [call.kwargs["previous_response_id"]
             for call in openai.responses.create.call_args_list],
            [None, "tool", "answer"],
        )
        self.assertEqual(
            [outputs.get_nowait() for _ in range(outputs.qsize())],
            [(0, "It is cold."), (0, ""), (0, "You're welcome."), (0, ""), (0, None)],
        )

    async def test_main_barge_in_stops_audio_closes_stream_and_plays_new_reply(self):
        recognizer, synthesizer = MagicMock(), MagicMock()
        loop = asyncio.get_running_loop()
        listening, speaking = asyncio.Event(), asyncio.Event()
        released = threading.Event()
        recognizer.start_continuous_recognition_async.return_value.get.side_effect = (
            lambda: loop.call_soon_threadsafe(listening.set)
        )
        success = self.synthesis_result()

        def synthesize(text):
            if text == "Old reply. ":
                loop.call_soon(speaking.set)

                def wait_for_stop():
                    if not released.wait(2):
                        raise TimeoutError("Native playback did not stop")
                    return self.synthesis_result(speechsdk.ResultReason.Canceled)

                return SimpleNamespace(get=wait_for_stop)
            return SimpleNamespace(get=lambda: success)

        synthesizer.speak_text_async.side_effect = synthesize
        synthesizer.stop_speaking_async.return_value.get.side_effect = released.set
        streams = [
            FakeStream([delta("Old reply. Old queued tail.")], block=True),
            FakeStream([delta("New reply."), completed("new")]),
        ]
        client, _ = self.project_client(streams)
        credential = MagicMock()
        credential.get_token = self.credential.get_token
        credential.__aenter__.return_value = credential

        async def recognize(text):
            await asyncio.to_thread(
                recognizer.recognized.connect.call_args.args[0],
                SimpleNamespace(result=SimpleNamespace(
                    reason=speechsdk.ResultReason.RecognizedSpeech, text=text
                )),
            )

        with (
            patch.object(agent, "DefaultAzureCredential", return_value=credential),
            patch.object(agent, "AIProjectClient", return_value=client),
            patch.object(agent.speechsdk, "SpeechRecognizer", return_value=recognizer),
            patch.object(agent.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
        ):
            task = asyncio.create_task(agent.main())
            self.addAsyncCleanup(self.cancel_task, task)
            await asyncio.wait_for(listening.wait(), 2)
            await recognize("First question")
            await asyncio.wait_for(speaking.wait(), 2)
            await asyncio.to_thread(
                recognizer.speech_start_detected.connect.call_args.args[0],
                SimpleNamespace(),
            )
            await recognize("Different question")
            await asyncio.to_thread(
                recognizer.session_stopped.connect.call_args.args[0], SimpleNamespace()
            )
            await asyncio.wait_for(task, 2)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(released.is_set())
        self.assertEqual(
            [call.args[0] for call in synthesizer.speak_text_async.call_args_list],
            ["Old reply. ", "New reply."],
        )


if __name__ == "__main__":
    unittest.main()
