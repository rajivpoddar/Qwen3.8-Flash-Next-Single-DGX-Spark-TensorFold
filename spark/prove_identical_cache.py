"""Bounded deployed-API proof that identical long requests remain cached."""
import json
import urllib.request

body = {"model": "qwen3.8-flash-next", "temperature": 0, "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [{"role": "system", "content": "Synthetic cache diagnostic.\n" +
                      "alpha beta gamma delta epsilon zeta eta theta\n" * 800},
                     {"role": "user", "content": "Reply with READY."}]}
results = []
for _ in range(4):
    request = urllib.request.Request("http://127.0.0.1:8888/v1/chat/completions",
                                    json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    results.append({"prompt_tokens": result["usage"]["prompt_tokens"], **result["tensorfold"]})
assert len({result["token_sha"] for result in results}) == 1, "identical request output changed"
assert all(result["cached"] >= 6144 for result in results[1:]), "reusable prefix was lost"
print(json.dumps({"identical_requests": results,
                  "output_parity": True,
                  "persistent_cache": True}), flush=True)
