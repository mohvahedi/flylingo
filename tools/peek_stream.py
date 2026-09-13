"""What does the live telemetry stream actually say right now?

The HUD reads this over a WebSocket, so this is the ground truth for what the panels show.
"""
import asyncio
import json
import sys

import websockets


async def main():
    url = "ws://127.0.0.1:8770/stream"
    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            frames = []
            for _ in range(5):
                raw = await asyncio.wait_for(ws.recv(), timeout=12)
                frames.append(json.loads(raw))
            print(f"got {len(frames)} frames\n")
            f = frames[0]
            print("keys:", ", ".join(sorted(f.keys())))
            print()
            for k in sorted(f.keys()):
                v = f[k]
                if isinstance(v, list):
                    print(f"  {k:20s}: list[{len(v)}] {json.dumps(v)[:160]}")
                elif isinstance(v, dict):
                    print(f"  {k:20s}: {json.dumps(v)[:200]}")
                else:
                    print(f"  {k:20s}: {v}")
            print()
            # does anything CHANGE between frames?
            print("what changes frame to frame:")
            for k in sorted(frames[0].keys()):
                vals = [json.dumps(fr.get(k))[:40] for fr in frames]
                if len(set(vals)) > 1:
                    print(f"  CHANGES  {k:20s}: {vals[0]} -> {vals[-1]}")
                else:
                    print(f"  static   {k:20s}: {vals[0]}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)


asyncio.run(main())
