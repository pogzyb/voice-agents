import asyncio
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

from openai.types.realtime import ResponseDoneEvent


spec = importlib.util.spec_from_file_location(
    "foundry_speech2speech", Path(__file__).resolve().parents[1] / "foundry-speech2speech.py"
)
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def weather_call(city="Detroit", call_id: str | None = "weather1", **overrides):
    return {
        "type": "function_call",
        "name": "tool_get_weather",
        "arguments": json.dumps({"city": city}),
        "call_id": call_id,
        **overrides,
    }


def completed(*output, status="completed"):
    return ResponseDoneEvent.model_validate({
        "type": "response.done",
        "event_id": "event1",
        "response": {"id": "response1", "status": status, "output": list(output)},
    })


class Speech2SpeechTests(unittest.IsolatedAsyncioTestCase):
    def connection(self, events):
        connection = MagicMock()
        connection.__aiter__.return_value = events
        connection.conversation.item.create = AsyncMock()
        connection.response.create = AsyncMock()
        connection.session.update = AsyncMock()
        return connection

    async def test_weather_tool_returns_result_then_continues_audio(self):
        audio = b"\x00\x01"
        connection = self.connection([
            SimpleNamespace(
                type="response.function_call_arguments.delta", delta='{"city":'
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                name="tool_get_weather", arguments='{"city":"Detroit"}',
                call_id="weather1",
            ),
            completed(weather_call()),
            SimpleNamespace(
                type="response.output_audio.delta",
                delta=base64.b64encode(audio).decode("ascii"),
            ),
            SimpleNamespace(type="response.output_audio_transcript.delta", delta="Sunny."),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Weather in Detroit?",
            ),
            completed(),
        ])
        speaker = MagicMock()
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            await agent.receive_response(connection, speaker)
        self.assertIn("Sunny.", output.getvalue())
        self.assertIn("[User said]: Weather in Detroit?", output.getvalue())
        speaker.write.assert_called_once_with(audio)
        connection.conversation.item.create.assert_awaited_once_with(item={
            "type": "function_call_output",
            "call_id": "weather1",
            "output": agent.tool_get_weather("Detroit"),
        })
        connection.response.create.assert_awaited_once_with()

    async def test_all_tool_outputs_are_sent_before_one_followup(self):
        connection = self.connection([
            completed(weather_call(), weather_call("Seattle", "weather2"))
        ])
        await agent.receive_response(connection, MagicMock())
        self.assertEqual(connection.mock_calls, [
            call.__aiter__(),
            call.conversation.item.create(item={
                "type": "function_call_output", "call_id": "weather1",
                "output": agent.tool_get_weather("Detroit"),
            }),
            call.conversation.item.create(item={
                "type": "function_call_output", "call_id": "weather2",
                "output": agent.tool_get_weather("Seattle"),
            }),
            call.response.create(),
        ])

    async def test_no_followup_without_completed_tool_calls(self):
        for event in (
            completed(),
            completed(weather_call(), status="cancelled"),
            completed(weather_call(), status="incomplete"),
            completed(weather_call(), status="failed"),
        ):
            with self.subTest(status=event.response.status):
                connection = self.connection([event])
                await agent.receive_response(connection, MagicMock())
                connection.conversation.item.create.assert_not_awaited()
                connection.response.create.assert_not_awaited()

    async def test_invalid_tool_calls_fail_explicitly(self):
        invalid_calls = [
            (weather_call(name="unknown"), ValueError, "Unknown tool"),
            (weather_call(call_id=None), ValueError, "missing a call_id"),
            (weather_call(arguments="{"), json.JSONDecodeError, "Expecting"),
        ]
        for arguments in (
            {}, {"city": ""}, {"city": "  "}, {"city": 42},
            {"city": None}, {"city": "Detroit", "extra": True}, [], None,
        ):
            invalid_calls.append((
                weather_call(arguments=json.dumps(arguments)),
                ValueError, "nonempty city string",
            ))
        for tool_call, error, message in invalid_calls:
            with self.subTest(tool_call=tool_call):
                connection = self.connection([completed(tool_call)])
                with self.assertRaisesRegex(error, message):
                    await agent.receive_response(connection, MagicMock())
                connection.conversation.item.create.assert_not_awaited()
                connection.response.create.assert_not_awaited()

    async def test_api_and_tool_output_errors_propagate(self):
        connection = self.connection([
            SimpleNamespace(type="error", error=SimpleNamespace(message="API failure"))
        ])
        with self.assertRaisesRegex(RuntimeError, "API failure"):
            await agent.receive_response(connection, MagicMock())
        connection = self.connection([completed(weather_call())])
        connection.conversation.item.create.side_effect = RuntimeError("send failed")
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            await agent.receive_response(connection, MagicMock())
        connection.response.create.assert_not_awaited()

    async def test_main_registers_only_weather_tool_and_keeps_audio_settings(self):
        connection = self.connection([])
        client = MagicMock()
        client.realtime.connect.return_value.__aenter__.return_value = connection
        credential = MagicMock()
        audio = MagicMock()
        microphone, speaker = MagicMock(), MagicMock()
        audio.open.side_effect = [microphone, speaker]
        audio.get_default_input_device_info.return_value = {"index": 0, "name": "mic"}
        audio.get_default_output_device_info.return_value = {"index": 1, "name": "speaker"}

        async def stream_microphone(_connection, _microphone, stop_event):
            await stop_event.wait()

        with (
            patch.dict(os.environ, {
                "AZURE_OPENAI_ENDPOINT": "https://example.com",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "test-realtime",
            }),
            patch.object(agent, "DefaultAzureCredential", return_value=credential),
            patch.object(agent, "get_bearer_token_provider", return_value=lambda: "token"),
            patch.object(agent, "AsyncOpenAI", return_value=client),
            patch.object(agent.pyaudio, "PyAudio", return_value=audio),
            patch.object(agent, "stream_microphone", side_effect=stream_microphone),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            await asyncio.wait_for(agent.main(), 2)
        session = connection.session.update.call_args.kwargs["session"]
        self.assertEqual(session["tool_choice"], "auto")
        self.assertEqual(session["tools"], [{
            "type": "function",
            "name": "tool_get_weather",
            "description": "Get example weather for a city (demo data, not live weather).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        }])
        self.assertEqual(session["output_modalities"], ["audio"])
        self.assertTrue(session["audio"]["input"]["turn_detection"]["create_response"])
        self.assertFalse(session["audio"]["input"]["turn_detection"]["interrupt_response"])
        microphone.close.assert_called_once()
        speaker.close.assert_called_once()
        audio.terminate.assert_called_once()
        credential.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
