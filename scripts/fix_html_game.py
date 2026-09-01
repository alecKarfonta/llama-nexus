#!/usr/bin/env python3
"""Regenerate a single broken game, or continue a truncated one to completion."""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8600/v1/chat/completions"
API_KEY = "sk-mlapi-qwen38-7bf65f5cc1d558d58611030770dc67f2a71b8a404ae07f5f"
GAMES_DIR = Path(__file__).resolve().parent.parent / "games"


def chat(messages: list, max_tokens: int = 16000, temperature: float = 0.7) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=900))
    resp["_dt"] = round(time.time() - t0, 1)
    return resp


def strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def regenerate(slug: str, title: str, concept: str) -> str:
    """Fresh generation with a tighter scope to fit the token budget."""
    prompt = (
        f'You are an expert HTML5 game developer. Create a complete, playable game: "{title}".\n\n'
        f"Concept: {concept}\n\n"
        "Requirements:\n"
        "1. ONE self-contained HTML file. All CSS and JS inline. No external dependencies.\n"
        "2. Canvas rendering, 60fps requestAnimationFrame loop, dark neon theme.\n"
        "3. IMPORTANT: keep the code COMPACT. Aim for under 500 lines total. No verbose comments. "
        "Compact but readable code. It must be COMPLETE — reaching </html> is mandatory. "
        "If you feel the budget tightening, cut visual polish, never cut closing tags or required features.\n"
        "4. Start screen, game-over screen with restart, score, pause on P/Esc.\n"
        "5. Must actually work — no TODOs, no stubs.\n\n"
        "Respond with the raw HTML ONLY. Start <!DOCTYPE html>, end </html>. No markdown fences."
    )
    resp = chat([
        {"role": "system", "content": "You write complete, working, compact single-file HTML5 games."},
        {"role": "user", "content": prompt},
    ])
    html = strip_fence(resp["choices"][0]["message"]["content"])
    print(f"regenerated {slug}: {len(html)} chars, {resp['_dt']}s, finish={resp['choices'][0]['finish_reason']}")
    return html


def continue_truncated(slug: str, max_rounds: int = 3) -> str:
    """Ask the model to complete a truncated HTML file, feeding the tail back."""
    path = GAMES_DIR / slug / "index.html"
    html = path.read_text()
    for round_no in range(1, max_rounds + 1):
        tail = html[-3000:]
        prompt = (
            "The following HTML file was CUT OFF mid-generation. Your job: output ONLY the remaining "
            "code that continues exactly from where it stops. Do not repeat any part of the file. "
            "Do not add commentary or markdown fences. Complete all remaining functionality and close "
            "the <script> and </body></html> tags.\n\n"
            f"...file so far (last 3000 chars)...\n{tail}\n\n...continue from here:"
        )
        resp = chat([{"role": "user", "content": prompt}], temperature=0.3)
        cont = strip_fence(resp["choices"][0]["message"]["content"])
        if not cont:
            print(f"round {round_no}: empty continuation, stopping")
            break
        html = html + cont
        done = html.rstrip().endswith("</html>")
        print(f"round {round_no}: +{len(cont)} chars (total {len(html)}), finish={resp['choices'][0]['finish_reason']}, closed={done}")
        if done:
            break
    return html


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "regen"
    slug = sys.argv[2]
    titles = {
        "starfield-dodge": ("Starfield Dodge",
                            "Endless runner: pilot a ship through an accelerating starfield obstacle course, "
                            "mouse or arrow keys to steer, near-miss bonus scoring, shield pickups, "
                            "explosion particles on death, high score in localStorage. Keep it compact."),
    }
    if slug not in titles:
        print(f"unknown slug {slug}")
        return 2
    title, concept = titles[slug]

    if mode == "regen":
        html = regenerate(slug, title, concept)
    else:
        html = continue_truncated(slug)

    ok = html.rstrip().endswith("</html>")
    out = GAMES_DIR / slug / "index.html"
    out.write_text(html)
    print(f"wrote {out}: {len(html)} chars, ends-with-html={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
