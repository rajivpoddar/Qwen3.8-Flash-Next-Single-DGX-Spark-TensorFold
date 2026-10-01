"""Real CUDA cache-resume proof for a synthetic thinking-mode tool conversation.

No external tool or customer data is used. Compare the cached second turn with
an uncached serial reference; both must produce exactly the same token IDs.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8888"


def call(messages, draft=True):
    body = {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": True}, "draft": draft,
            "messages": messages,
            "tools": [{"type": "function", "function": {
                "name": "get_time", "description": "Get time in a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                               "required": ["city"]}}}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


messages = [{"role": "system", "content": "Use get_time when asked for time. " +
             "Synthetic stable context line.\n" * 1000},
            {"role": "user", "content": "Use get_time to get the time in Delhi."}]
first = call(messages)
assistant = first["choices"][0]["message"]
tool = assistant.get("tool_calls", [None])[0]
assert tool and tool["function"]["name"] == "get_time", first
# Claude's next tool turn need not replay the model's private reasoning text.
assistant.pop("reasoning_content", None)
messages += [assistant, {"role": "tool", "tool_call_id": tool["id"],
                        "content": "Delhi time is 12:00. Reply with this time, no more tool calls."}]
cached = call(messages)
repeats = [call(messages) for _ in range(2)]
serial = call(messages, draft=False)
assert serial["tensorfold"]["cached"] == 0, serial["tensorfold"]
for result in [cached, *repeats]:
    assert result["tensorfold"]["cached"] >= 4096, result["tensorfold"]
    assert result["tensorfold"]["token_sha"] == serial["tensorfold"]["token_sha"], "cached/MTP output changed"
assert cached["choices"][0]["message"].get("content"), cached
print(json.dumps({"thinking_tool_prefix_cache": "PASS", "first": first["tensorfold"],
                  "cached_second_turn": cached["tensorfold"], "uncached_reference": serial["tensorfold"],
                  "repeated_cached_turns": [r["tensorfold"] for r in repeats],
                  "prompt_tokens": cached["usage"]["prompt_tokens"]}), flush=True)
