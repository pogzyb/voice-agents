## voice-agents

| File | Description |
| -------- | -------- |
| `foundry-cascaded-from-scratch.py`   | Standalone example of `Speech-to-Text -> LLM -> Text-to-Speech` style of voice agents.  |
| `foundry-s2s-gpt-realtime.py`   | Standalone example of `S2S` style of voice agents using `gpt-realtime-2.1`.  |   
| `foundry-voice-live-agent.py`   | Standalone example of Microsoft's `VoiceLive` API for voice agents (Foundry project API).  |   
| `foundry-voice-live-service.py`   | Standalone example of Microsoft's `VoiceLive` API for voice service.  |   
| `mcp/function_app.py` | Azure Functions MCP tools, including employee search. |

The `search_employees` MCP tool accepts a `name`, a `title`, or both. At least
one must be non-empty. It returns the three closest employees as a JSON array,
using case-insensitive fuzzy matching and weighting supplied fields equally.
