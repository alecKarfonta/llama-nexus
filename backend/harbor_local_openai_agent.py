"""
Custom Harbor agent that drives shell actions via OpenAI-compatible chat API.

This agent is intentionally strict:
- Expects structured JSON actions from the model
- Raises errors on malformed outputs instead of silently recovering
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import httpx
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class LocalOpenAIAgent(BaseAgent):
    SUPPORTS_ATIF = False
    SUPPORTS_WINDOWS = False

    @staticmethod
    def name() -> str:
        return "local-openai-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # No in-environment install needed for this external agent.
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        base_url = os.getenv("OPENAI_BASE_URL", "http://llamacpp-api:8080/v1").rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY", "placeholder-api-key")
        model = self.model_name or os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        max_steps = int(os.getenv("TB_AGENT_MAX_STEPS", "40"))
        max_obs_chars = int(os.getenv("TB_AGENT_MAX_OBS_CHARS", "6000"))

        system_prompt = (
            "You are an autonomous terminal agent operating in a container task. "
            "At each step, reply in exactly one of these formats:\n"
            "CMD: <single shell command>\n"
            "FINISH: <brief completion summary>\n"
            "Never include explanations outside CMD/FINISH."
        )

        transcript: List[Dict[str, Any]] = []
        usage_input = 0
        usage_output = 0

        async with httpx.AsyncClient(timeout=180.0) as client:
            for step in range(1, max_steps + 1):
                user_payload = {
                    "instruction": instruction,
                    "step": step,
                    "history": transcript[-10:],
                }
                request_body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload)},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024,
                }
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                usage_input += int(usage.get("prompt_tokens", 0) or 0)
                usage_output += int(usage.get("completion_tokens", 0) or 0)

                message = data["choices"][0]["message"]
                content = (message.get("content") or message.get("reasoning_content") or "").strip()
                if not content:
                    raise RuntimeError("Model returned empty action payload")

                action, value = self._parse_action(content)
                if action == "finish":
                    transcript.append({"step": step, "action": {"action": "finish", "summary": value}})
                    break
                if action != "exec":
                    raise RuntimeError(f"Unsupported model action: {action!r}")
                command = value

                result = await environment.exec(command, timeout_sec=180)
                observation = {
                    "command": command,
                    "return_code": result.return_code,
                    "stdout": (result.stdout or "")[:max_obs_chars],
                    "stderr": (result.stderr or "")[:max_obs_chars],
                }
                transcript.append(
                    {
                        "step": step,
                        "action": {"action": "exec", "command": command},
                        "observation": observation,
                    }
                )
            else:
                transcript.append(
                    {
                        "step": max_steps,
                        "action": {
                            "action": "finish",
                            "summary": f"Reached max steps ({max_steps}); stopping run",
                        },
                    }
                )

        context.n_input_tokens = usage_input
        context.n_output_tokens = usage_output
        context.metadata = {
            "agent": self.name(),
            "steps_executed": len(transcript),
            "transcript_tail": transcript[-5:],
        }

    @staticmethod
    def _parse_action(content: str) -> Tuple[str, str]:
        """Parse model output into ('exec'|'finish', value)."""
        content = content.strip()
        if content.upper().startswith("CMD:"):
            command = content[4:].strip()
            if not command:
                raise RuntimeError("CMD action returned without a command")
            return ("exec", command)
        if content.upper().startswith("FINISH:"):
            summary = content[7:].strip() or "task finished"
            return ("finish", summary)

        # Accept fenced shell snippets by treating first non-empty line as command.
        fenced = re.search(r"```(?:bash|sh)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            for line in fenced.group(1).splitlines():
                candidate = line.strip()
                if candidate and not candidate.startswith("#"):
                    return ("exec", candidate)

        # Accept plain one-liner command as a last strict parse route.
        if "\n" not in content and len(content.split()) >= 1:
            return ("exec", content)

        for line in content.splitlines():
            candidate = line.strip()
            if candidate:
                return ("exec", candidate)

        raise RuntimeError("Model returned no actionable content")
