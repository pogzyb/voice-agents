## voice-agents

| File | Description |
| -------- | -------- |
| `foundry-cascaded.py`   | Standalone example of `Speech-to-Text -> LLM -> Text-to-Speech` style of voice agents.  |
| `foundry-cascaded-turn-detection.py` | Cascaded speech agent with turn detection and barge-in. |
| `foundry-speech2speech.py`   | Standalone example of `S2S` style of voice agents using `gpt-realtime-2.1`.  |   
| `foundry-voicelive-agent.py`   | Standalone example of Microsoft's `VoiceLive` API for voice agents.  |   
| `twilio-call-me.sh`   | Standalone example of Microsoft's `VoiceLive` API for voice agents.  |   
| `twilio-call-agent.md`   | Instructions for calling your Voice Agent with Twilio.  |   

### Speech-to-speech agent

Run `python foundry-speech2speech.py` with `AZURE_OPENAI_ENDPOINT` and
`AZURE_OPENAI_DEPLOYMENT_NAME` set (or supplied in `.env`) and an Azure identity
authorized to access the Realtime deployment. Requires the dependencies in
`requirements.txt` and a default microphone and speaker.

The session exposes one function, `tool_get_weather`, which accepts a city and
returns demo data, not live weather. Completed function calls are executed
locally, their results are added to the conversation, and a follow-up response
is requested so the assistant can speak the answer. Invalid tool names or
arguments terminate the script with an error. Use headphones and press Ctrl+C
to stop.

Run the speech-to-speech tests without Azure access or audio devices using
`python -m unittest discover -s tests -p "test_foundry_speech2speech.py" -v`.

### Foundry voice agent with a function tool

Run `python example-voice-live-function-tool.py` with `FOUNDRY_PROJECT_ENDPOINT`
set (or supplied in `.env`) and an Azure identity authorized to create and use
voice agents. Install `"azure-ai-projects[voice]>=2.7.0"`, `azure-identity`,
`python-dotenv`, and `pyaudio`. The optional `FOUNDRY_VOICE_AGENT_MODEL` and
`FOUNDRY_VOICE_AGENT_NAME` settings default to `gpt-realtime` and
`sample-voice-agent-function-tool`.

The script streams the default microphone and plays responses on the default
speaker using 24 kHz mono, 16-bit PCM. Server-side voice activity detection
automatically starts responses after each utterance. It keeps the same session
open across turns and idle periods until Ctrl+C, which stops the audio workers
and closes the devices and connection. Device and service errors terminate the
script with an error.

Barge-in is enabled: when server-side VAD detects you speaking during a reply,
the service cancels generation and the client clears buffered playback, ignores
remaining events from that response, and truncates its audio in conversation
history to the playback position. The agent then answers your latest utterance
after a 300 ms silence window. Playback uses 20 ms PortAudio callbacks, so a
generated reply cannot keep playing from an old application queue after you
interrupt it. A small amount of audio already buffered by the device can still
be heard; truncation position is estimated from frames handed to the device.

Ask about the weather to exercise the local `get_weather` function, which returns
canned demo data, not live weather. Use headphones to avoid audio feedback and
the agent interrupting itself; this example does not configure echo cancellation. The
created agent version is retained, matching the example's existing behavior.

Run its tests without Azure access or audio devices using
`python -m unittest discover -s tests -p "test_voice_live_function_tool.py" -v`.

### Cascaded speech agent

Requires Python 3.11+ and the dependencies in `requirements.txt`. Authenticate
with Azure (for example, `az login`) using an identity authorized to access the
Speech resource and the Foundry model deployment. Set these environment
variables (or add them to `.env`):

- `COGNITIVE_SERVICES_ENDPOINT`: HTTPS endpoint of the Speech resource.
- `COGNITIVE_SERVICES_RESOURCE_ID`: full Azure resource ID of that resource.
- `FOUNDRY_PROJECT_ENDPOINT`: Foundry project endpoint.
- `FOUNDRY_MODEL_DEPLOYMENT`: model deployment supporting the Responses API and
  function calling (defaults to `gpt-6-luna` when unset).

Run `python foundry-cascaded.py` and speak into the default microphone. Finalized
utterances are queued to the LLM; response text deltas are queued to the default
speaker and synthesized in sentence-sized chunks, with any remaining text
spoken at the end of each response. Conversation context is retained through
Responses API response IDs. The weather tool returns demo data, not live weather.

The example uses `mai-transcribe-2` and `en-US-Ethan:MAI-Voice-2.1-Flash`; the
Speech resource must support these models. Use headphones to avoid the microphone
recognizing the agent's own speech. Replies play in order; barge-in/interruption
is not implemented. Press Ctrl+C to stop all stages. Service failures terminate
the pipeline with an error rather than leaving other stages waiting indefinitely.

For interruption support, run `python foundry-cascaded-turn-detection.py` with
the same environment variables. It sets the Speech SDK segmentation silence
timeout to 500 ms and uses speech-start events (with partial recognition as a
fallback) to cancel the active LLM stream and stop speech synthesis immediately,
without waiting for the final transcript. Queued text and buffered phrases from
the interrupted reply are discarded. The next finalized utterance starts a new
reply; only completed LLM responses are retained as conversation context, so a
completed response may include text that was not played before interruption.
Use headphones: this example does not implement acoustic echo cancellation.
Voice Live turn-detection settings do not apply to this separate STT/LLM/TTS
pipeline.

Run the pipeline tests without Azure access or audio devices using
`python -m unittest discover -s tests -p "test_foundry_cascaded*.py" -v`.


&flight=voice_agent_bundle
 