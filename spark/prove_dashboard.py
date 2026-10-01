"""Read the dashboard's live WebSocket state after a TensorFold cutover.

Run from a machine with `websockets` installed; this script does not mutate
dashboard configuration or start/stop the model.
"""

import asyncio
import json
import os

import websockets


async def main() -> None:
    url = os.environ.get("DASHBOARD_WS", "ws://192.168.68.113:3000/ws")
    async with websockets.connect(url, open_timeout=5) as socket:
        for _ in range(20):
            snapshot = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
            engines = [e for e in snapshot.get("engines", []) if e.get("endpoint", "").endswith(":30000")]
            for engine in engines:
                model = (engine.get("model") or {}).get("name")
                status = engine.get("status") or {}
                state = status.get("type") if isinstance(status, dict) else status
                metrics = engine.get("metrics") or {}
                if (state == "Running" and model == "qwen3.8-flash-next"
                        and not metrics.get("warming_up", True)
                        and metrics.get("tokens_per_sec") is not None
                        and metrics.get("total_generation_tokens") is not None
                        and metrics.get("e2e_latency_ms") is not None
                        and metrics.get("e2e_buckets")):
                    print("Dashboard: running TensorFold alias on :30000; live rate and E2E histogram populated")
                    return
                print(json.dumps({"endpoint": engine.get("endpoint"), "status": state,
                                  "model": model, "metrics": engine.get("metrics")}), flush=True)
        raise SystemExit("dashboard did not report the target model and metrics on :30000")


if __name__ == "__main__":
    asyncio.run(main())
