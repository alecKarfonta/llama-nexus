#!/usr/bin/env python3
"""Generate single-file HTML games from Qwen3.8-Flash-Next, saved for manual testing.

Each game is written to games/<slug>/index.html plus a launcher index.html linking all.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8600/v1/chat/completions"
API_KEY = "sk-mlapi-qwen38-7bf65f5cc1d558d58611030770dc67f2a71b8a404ae07f5f"
OUT_DIR = Path(__file__).resolve().parent.parent / "games"

GAMES = [
    ("gravity-well-lander", "Gravity Well Lander",
     "A lunar-lander style game where the landing pad orbits a planet, so gravity constantly "
     "pulls your ship sideways. Arrow keys/WASD thrust, safe landing under velocity threshold, "
     "fuel gauge, particle exhaust, 5 levels with increasing planet mass."),
    ("neon-block-breaker", "Neon Block Breaker",
     "A Breakout variant with neon glow canvas rendering, mouse paddle, power-ups that fall "
     "from broken bricks (multiball, wide paddle, slow ball), 3 levels, score + lives, "
     "screen shake on brick hits."),
    ("orbit-snake", "Orbit Snake",
     "Snake played on a circular track: your snake moves along concentric orbital rings, "
     "arrow keys jump between rings, eat glowing orbs, avoid your own tail, score, "
     "progressive speed, trailing glow effects."),
    ("color-cascade", "Color Cascade",
     "A color-matching puzzle: blocks fall in pairs, click to cycle a block's color, "
     "matching 3+ adjacent same-color blocks clears them with a cascade animation, "
     "combo multiplier, next-piece preview, game over when column overflows."),
    ("starfield-dodge", "Starfield Dodge",
     "Endless runner: pilot a ship through an accelerating starfield obstacle course, "
     "mouse or arrow keys to steer, near-miss bonus scoring, shield pickups, "
     "explosion particles on death, high score in localStorage."),
    ("word-forge-typing", "Word Forge Typing Defense",
     "A typing defense game: enemy words drift toward your tower, type them exactly to "
     "destroy them, backspace works, difficulty ramps with word length and speed, "
     "lives, WPM counter, wave system."),
]


def build_prompt(title: str, concept: str) -> str:
    return (
        f'You are an expert HTML5 game developer. Create a complete, playable game: "{title}".\n\n'
        f"Concept: {concept}\n\n"
        "Requirements:\n"
        "1. Output ONE complete self-contained HTML file. All CSS and JavaScript inline, no external "
        "dependencies, no CDN, no images (use canvas drawing only).\n"
        "2. Modern polished visuals: dark theme, glow effects, smooth 60fps requestAnimationFrame loop.\n"
        "3. Responsive canvas that fits the window.\n"
        "4. Keyboard and/or mouse controls as appropriate. Show controls on a start screen.\n"
        "5. Start screen, game-over screen with restart, score display, pause with P or Esc.\n"
        "6. Sound optional (WebAudio ok) but must be muted-friendly.\n"
        "7. Must actually work — no TODOs, no stubs, no external files.\n\n"
        "Respond with the raw HTML file contents ONLY. Start with <!DOCTYPE html> and end with </html>. "
        "No markdown fences, no commentary before or after."
    )


def extract_html(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def generate(slug: str, title: str, concept: str) -> dict:
    prompt = build_prompt(title, concept)
    req = urllib.request.Request(
        URL,
        data=json.dumps({
            "messages": [
                {"role": "system", "content": "You write complete, working single-file HTML5 games."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 16000,
            "temperature": 0.7,
            "top_p": 0.95,
            # [TIME] | INFO | SCRIPTS.GEN_HTML_GAMES | Server default preserves thinking mode,
            # which burns the whole budget on hidden reasoning; disable per-request.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    html = extract_html(content)
    ok = html.lower().startswith("<!doctype html>")
    if not ok:
        # model sometimes prefixes prose before <!DOCTYPE; salvage the document
        m = re.search(r"(<!DOCTYPE html>.*</html>)", html, re.DOTALL | re.IGNORECASE)
        if m:
            html = m.group(1)
            ok = True
    game_dir = OUT_DIR / slug
    game_dir.mkdir(parents=True, exist_ok=True)
    (game_dir / "index.html").write_text(html)
    return {
        "slug": slug, "title": title,
        "valid_html": ok, "chars": len(html),
        "gen_seconds": round(dt, 1),
        "completion_tokens": resp["usage"]["completion_tokens"],
        "tok_s": round(resp["usage"]["completion_tokens"] / dt, 1),
        "reasoning_tokens": resp["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (slug, title, concept) in enumerate(GAMES, 1):
        print(f"[{i}/{len(GAMES)}] generating {title} ...", flush=True)
        try:
            r = generate(slug, title, concept)
            print(f"    ok={r['valid_html']} chars={r['chars']} "
                  f"{r['gen_seconds']}s @ {r['tok_s']} tok/s (reasoning: {r['reasoning_tokens']})", flush=True)
            results.append(r)
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            results.append({"slug": slug, "title": title, "error": str(exc)})

    cards = "\n".join(
        f'  <div class="card"><a href="{r["slug"]}/index.html">{r.get("title", r["slug"])}</a>'
        f'<span class="meta">{r.get("chars", "?")} chars | {r.get("gen_seconds", "?")}s | '
        f'{r.get("tok_s", "?")} tok/s{" | INVALID HTML" if not r.get("valid_html", False) else ""}</span></div>'
        if "error" not in r else
        f'  <div class="card failed">{r["title"]}: FAILED — {r["error"]}</div>'
        for r in results
    )
    index = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LLM HTML Games — Test Gallery</title>
<style>
body {{ background:#0b0e1a; color:#e8eaf6; font-family:system-ui,sans-serif; max-width:820px; margin:2rem auto; padding:0 1rem; }}
h1 {{ color:#7c9cff; }} .card {{ background:#151a2e; border:1px solid #2a3354; border-radius:10px; padding:1rem 1.25rem; margin:.6rem 0; }}
.card a {{ color:#9fd0ff; font-size:1.15rem; font-weight:600; text-decoration:none; }} .card a:hover {{ text-decoration:underline; }}
.meta {{ display:block; color:#8892b0; font-size:.85rem; margin-top:.25rem; }}
.failed {{ border-color:#a33; color:#f88; }}
</style></head><body>
<h1>Generated HTML Games</h1>
<p>Qwen3.8-Flash-Next · {time.strftime('%Y-%m-%d %H:%M')} · single-file games, click to play</p>
{cards}
</body></html>"""
    (OUT_DIR / "index.html").write_text(index)
    (OUT_DIR / "generation_report.json").write_text(json.dumps(results, indent=2))
    ok_count = sum(1 for r in results if r.get("valid_html"))
    print(f"\ndone: {ok_count}/{len(GAMES)} valid -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
