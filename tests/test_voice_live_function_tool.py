import importlib.util
import io
import json
import os
from pathlib import Path
from threading import Event
from typing import Literal
import unittest
from unittest.mock import MagicMock, patch

from azure.ai.projects import models


spec = importlib.util.spec_from_file_location(
    "voice_live_function_tool",
    Path(__file__).resolve().parents[1] / "example-voice-live-function-tool.py",
)
assert spec is not None and spec.loader is not None
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def response_done(
    status: Literal["completed", "cancelled", "failed", "incomplete", "in_progress"] = "completed",
    response_id: str = "response1",
):
    return models.RealtimeServerEventResponseDone(
        event_id="event1",
        response=models.VoiceAgentRealtimeResponse(id=response_id, status=status, output=[]),
    )


class VoiceLiveFunctionToolTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.connection = self.client.beta.voice_agents.realtime.connect.return_value.__enter__.return_value
        self.audio = MagicMock()
        self.microphone = MagicMock()
        self.speaker = MagicMock()
        self.audio.open.side_effect = [self.microphone, self.speaker]
        self.audio.get_default_input_device_info.return_value = {"index": 0, "name": "mic"}
        self.audio.get_default_output_device_info.return_value = {"index": 1, "name": "speaker"}
        self.read_delay = Event()
        self.microphone.read.side_effect = self.read_audio

    def read_audio(self, *_args, **_kwargs):
        self.read_delay.wait(0.005)
        return b"\x00\x01"

    def run_session(self):
        with (
            patch.object(agent.pyaudio, "PyAudio", return_value=self.audio),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            agent._run_session_with_tool_support(self.client, "test-agent")

    def assert_closed(self):
        for stream in (self.microphone, self.speaker):
            stream.stop_stream.assert_called_once_with()
            stream.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.client.beta.voice_agents.realtime.connect.return_value.__exit__.assert_called_once()

    def test_microphone_sends_raw_pcm(self):
        stop = Event()
        connection = MagicMock()
        connection.input_audio_buffer.append.side_effect = lambda **_kwargs: stop.set()
        agent.stream_microphone(connection, self.microphone, stop)
        self.microphone.read.assert_called_once_with(
            agent.CHUNK_FRAMES, exception_on_overflow=False,
        )
        connection.input_audio_buffer.append.assert_called_once_with(audio=b"\x00\x01")

    def test_microphone_does_not_send_after_stop(self):
        stop = Event()

        def stop_during_read(*_args, **_kwargs):
            stop.set()
            return b"\x00\x01"

        self.microphone.read.side_effect = stop_during_read
        agent.stream_microphone(self.connection, self.microphone, stop)
        self.connection.input_audio_buffer.append.assert_not_called()

    def test_playback_spans_deltas_and_pads_with_silence(self):
        playback = agent.AudioPlayback()
        playback.add("item1", 0, b"\x01\x00" * 100)
        playback.add("item1", 0, b"\x02\x00" * 200)
        data, status = playback.render(None, 480, {}, 0)
        self.assertEqual(data, b"\x01\x00" * 100 + b"\x02\x00" * 200 + b"\x00\x00" * 180)
        self.assertEqual(status, agent.pyaudio.paContinue)
        # Deltas arriving after an underrun still belong to the same audio item.
        playback.add("item1", 0, b"\x03\x00" * 180)
        playback.render(None, 480, {}, 0)
        self.assertEqual(playback.interrupt(), [("item1", 0, 20)])

    def test_idle_and_completed_turns_keep_session_open_with_tool_and_audio(self):
        events = iter([
            TimeoutError(),
            response_done(),
            models.RealtimeServerEventResponseFunctionCallArgumentsDone(
                event_id="event2", response_id="response1", item_id="item1",
                output_index=0, name="get_weather", call_id="call1",
                arguments='{"city":"Seattle"}',
            ),
            response_done(),
            models.RealtimeServerEventResponseAudioDelta({
                "type": "response.output_audio.delta",
                "event_id": "event3", "response_id": "response2", "item_id": "item2",
                "output_index": 0, "content_index": 0, "delta": "AAE=",
            }),
            models.RealtimeServerEventResponseAudioTranscriptDone(
                event_id="event4", response_id="response2", item_id="item2",
                output_index=0, content_index=0, transcript="Sunny in Seattle.",
            ),
            response_done(),
        ])

        received_done = 0

        def receive(**_kwargs):
            nonlocal received_done
            try:
                event = next(events)
            except StopIteration:
                render = self.audio.open.call_args_list[1].kwargs["stream_callback"]
                data, _ = render(None, 480, {}, 0)
                self.assertEqual(data, b"\x00\x01" + b"\x00" * 958)
                raise KeyboardInterrupt
            if isinstance(event, BaseException):
                raise event
            if isinstance(event, models.RealtimeServerEventResponseDone):
                received_done += 1
                if received_done <= 2:
                    self.connection.response.create.assert_not_called()
            return event

        self.connection.recv.side_effect = receive
        with self.assertRaises(KeyboardInterrupt):
            self.run_session()
        self.connection.response.create.assert_called_once_with()
        output = self.connection.conversation.item.create.call_args.kwargs["item"]
        self.assertEqual(output.call_id, "call1")
        self.assertEqual(json.loads(output.output)["city"], "Seattle")
        self.speaker.write.assert_not_called()
        self.assert_closed()

    def test_service_errors_propagate_and_release_devices(self):
        self.connection.recv.return_value = models.RealtimeServerEventError(
            event_id="error1",
            error=models.RealtimeServerEventErrorError(type="server_error", message="Service failure"),
        )
        with self.assertRaisesRegex(RuntimeError, "Service failure"):
            self.run_session()
        self.assert_closed()

    def test_cancelled_tool_response_does_not_request_a_followup(self):
        self.connection.recv.side_effect = [
            models.RealtimeServerEventResponseFunctionCallArgumentsDone(
                event_id="event1", response_id="response1", item_id="item1",
                output_index=0, name="get_weather", call_id="call1",
                arguments='{"city":"Seattle"}',
            ),
            response_done("cancelled"),
            response_done(),
            KeyboardInterrupt(),
        ]
        with self.assertRaises(KeyboardInterrupt):
            self.run_session()
        self.connection.conversation.item.create.assert_not_called()
        self.connection.response.create.assert_not_called()
        self.assert_closed()

    def test_connection_failure_propagates_and_releases_devices(self):
        self.connection.recv.side_effect = ConnectionResetError("Connection lost")
        with self.assertRaisesRegex(ConnectionResetError, "Connection lost"):
            self.run_session()
        self.assert_closed()

    def test_failed_response_propagates(self):
        self.connection.recv.return_value = response_done("failed")
        with self.assertRaisesRegex(RuntimeError, "Agent response failed"):
            self.run_session()
        self.assert_closed()

    def test_microphone_failure_propagates_and_stops_playback(self):
        self.microphone.read.side_effect = OSError("Microphone failure")
        self.connection.recv.side_effect = TimeoutError
        with self.assertRaisesRegex(OSError, "Microphone failure"):
            self.run_session()
        self.assert_closed()

    def test_speaker_failure_propagates_and_stops_capture(self):
        self.speaker.is_active.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Speaker stream ended unexpectedly"):
            self.run_session()
        self.assert_closed()

    def test_partial_device_setup_is_cleaned_up(self):
        self.audio.open.side_effect = [self.microphone, OSError("No speaker")]
        with self.assertRaisesRegex(OSError, "No speaker"):
            self.run_session()
        self.microphone.stop_stream.assert_called_once_with()
        self.microphone.close.assert_called_once_with()
        self.audio.terminate.assert_called_once_with()
        self.client.beta.voice_agents.realtime.connect.assert_not_called()

    def test_main_configures_pcm_audio_and_automatic_turns(self):
        client = MagicMock()
        with (
            patch.dict(os.environ, {"FOUNDRY_PROJECT_ENDPOINT": "https://example.com"}),
            patch.object(agent, "DefaultAzureCredential"),
            patch.object(agent, "AIProjectClient") as client_type,
            patch.object(agent, "_run_session_with_tool_support") as run_session,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            client_type.return_value.__enter__.return_value = client
            agent.main()
        definition = client.agents.create_version.call_args.kwargs["definition"]
        self.assertEqual(definition.output_modalities, ["audio"])
        self.assertEqual(definition.audio.input.format.rate, 24000)
        self.assertEqual(definition.audio.output.format.rate, 24000)
        self.assertEqual(definition.audio.output.voice, "alloy")
        self.assertTrue(definition.audio.input.turn_detection.create_response)
        self.assertTrue(definition.audio.input.turn_detection.interrupt_response)
        self.assertEqual(definition.audio.input.turn_detection.silence_duration_ms, 300)
        self.assertEqual(definition.tools[0].name, "get_weather")
        run_session.assert_called_once_with(client, client.agents.create_version.call_args.kwargs["agent_name"])

    def test_interruption_discards_current_and_queued_audio_but_accepts_new_reply(self):
        playback = agent.AudioPlayback()
        playback.add("old", 0, b"\x01\x00" * 1000)
        audio, status = playback.render(None, 480, {}, 0)
        self.assertEqual(audio, b"\x01\x00" * 480)
        self.assertEqual(status, agent.pyaudio.paContinue)
        self.assertEqual(playback.interrupt(), [("old", 0, 20)])
        audio, _ = playback.render(None, 480, {}, 0)
        self.assertEqual(audio, b"\x00\x00" * 480)
        playback.add("new", 0, b"\x02\x00" * 480)
        audio, _ = playback.render(None, 480, {}, 0)
        self.assertEqual(audio, b"\x02\x00" * 480)

    def test_interruption_truncates_all_queued_items_and_is_repeatable(self):
        playback = agent.AudioPlayback()
        self.assertEqual(playback.interrupt(), [])
        playback.add("first", 0, b"\x01\x00" * 960)
        playback.add("second", 0, b"\x02\x00" * 480)
        playback.render(None, 480, {}, 0)
        self.assertEqual(playback.interrupt(), [("first", 0, 20), ("second", 0, 0)])
        self.assertEqual(playback.interrupt(), [])
        playback.add("third", 0, b"\x03\x00" * 960)
        playback.render(None, 480, {}, 0)
        self.assertEqual(playback.interrupt(), [("third", 0, 20)])

    def test_interruption_after_generation_finishes_still_flushes_device_queue(self):
        events = iter([
            models.RealtimeServerEventResponseAudioDelta(
                event_id="audio", response_id="old", item_id="old-item",
                output_index=0, content_index=0, delta=b"\x01\x00" * 2000,
            ),
            response_done(response_id="old"),
            models.RealtimeServerEventInputAudioBufferSpeechStarted(
                event_id="speech", audio_start_ms=0, item_id="user-item",
            ),
        ])

        def receive(**_kwargs):
            render = self.audio.open.call_args_list[1].kwargs["stream_callback"]
            try:
                event = next(events)
            except StopIteration:
                self.assertEqual(render(None, 480, {}, 0)[0], b"\x00\x00" * 480)
                raise KeyboardInterrupt
            if isinstance(event, models.RealtimeServerEventInputAudioBufferSpeechStarted):
                render(None, 480, {}, 0)
            return event

        self.connection.recv.side_effect = receive
        with self.assertRaises(KeyboardInterrupt):
            self.run_session()
        self.connection.conversation.item.truncate.assert_called_once_with(
            item_id="old-item", content_index=0, audio_end_ms=20,
        )
        self.connection.response.cancel.assert_not_called()
        self.assert_closed()

    def test_interrupted_tool_cannot_start_a_stale_followup(self):
        self.connection.recv.side_effect = [
            models.RealtimeServerEventResponseCreated(
                event_id="created", response=models.VoiceAgentRealtimeResponse(id="old"),
            ),
            models.RealtimeServerEventResponseFunctionCallArgumentsDone(
                event_id="old-tool", response_id="old", item_id="tool-item",
                output_index=0, name="get_weather", call_id="old-call",
                arguments='{"city":"Seattle"}',
            ),
            models.RealtimeServerEventInputAudioBufferSpeechStarted(
                event_id="speech", audio_start_ms=0, item_id="user-item",
            ),
            models.RealtimeServerEventResponseCreated(
                event_id="new-created", response=models.VoiceAgentRealtimeResponse(id="new"),
            ),
            models.RealtimeServerEventResponseFunctionCallArgumentsDone(
                event_id="new-tool", response_id="new", item_id="new-tool-item",
                output_index=0, name="get_weather", call_id="new-call",
                arguments='{"city":"Detroit"}',
            ),
            response_done(response_id="old"),
            response_done(response_id="new"),
            KeyboardInterrupt(),
        ]
        with self.assertRaises(KeyboardInterrupt):
            self.run_session()
        self.connection.response.create.assert_called_once_with()
        self.connection.conversation.item.create.assert_called_once()
        item = self.connection.conversation.item.create.call_args.kwargs["item"]
        self.assertEqual(item.call_id, "new-call")
        self.assertEqual(json.loads(item.output)["city"], "Detroit")
        self.assert_closed()

    def test_speech_started_flushes_playback_and_ignores_late_response_events(self):
        render = None
        events = iter([
            models.RealtimeServerEventResponseCreated(
                event_id="created", response=models.VoiceAgentRealtimeResponse(id="old"),
            ),
            models.RealtimeServerEventResponseAudioDelta(
                event_id="audio", response_id="old", item_id="old-item",
                output_index=0, content_index=0, delta=b"\x01\x00" * 2000,
            ),
            models.RealtimeServerEventInputAudioBufferSpeechStarted(
                event_id="speech", audio_start_ms=0, item_id="user-item",
            ),
            models.RealtimeServerEventResponseAudioDelta(
                event_id="late-audio", response_id="old", item_id="old-item",
                output_index=0, content_index=0, delta=b"\x03\x00" * 480,
            ),
            models.RealtimeServerEventResponseFunctionCallArgumentsDone(
                event_id="late-tool", response_id="old", item_id="old-tool",
                output_index=0, name="get_weather", call_id="stale",
                arguments='{"city":"Seattle"}',
            ),
            response_done(response_id="old"),
            models.RealtimeServerEventResponseCreated(
                event_id="created-new", response=models.VoiceAgentRealtimeResponse(id="new"),
            ),
            models.RealtimeServerEventResponseAudioDelta(
                event_id="new-audio", response_id="new", item_id="new-item",
                output_index=0, content_index=0, delta=b"\x02\x00" * 480,
            ),
        ])

        def receive(**_kwargs):
            nonlocal render
            render = self.audio.open.call_args_list[1].kwargs["stream_callback"]
            try:
                event = next(events)
            except StopIteration:
                self.assertEqual(render(None, 480, {}, 0)[0], b"\x02\x00" * 480)
                raise KeyboardInterrupt
            if isinstance(event, models.RealtimeServerEventInputAudioBufferSpeechStarted):
                self.assertEqual(render(None, 480, {}, 0)[0], b"\x01\x00" * 480)
            if isinstance(event, models.RealtimeServerEventResponseCreated) and event.response.id == "new":
                self.assertEqual(render(None, 480, {}, 0)[0], b"\x00\x00" * 480)
            return event

        self.connection.recv.side_effect = receive
        with self.assertRaises(KeyboardInterrupt):
            self.run_session()
        self.connection.conversation.item.truncate.assert_called_once_with(
            item_id="old-item", content_index=0, audio_end_ms=20,
        )
        self.connection.response.create.assert_not_called()
        self.connection.conversation.item.create.assert_not_called()
        self.assert_closed()


if __name__ == "__main__":
    unittest.main()
