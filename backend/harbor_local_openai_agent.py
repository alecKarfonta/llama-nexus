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
import shlex
from datetime import datetime
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
        model_timeout_sec = float(os.getenv("TB_AGENT_MODEL_TIMEOUT_SEC", "75"))
        command_timeout_sec = int(os.getenv("TB_AGENT_COMMAND_TIMEOUT_SEC", "120"))
        max_model_tokens = int(os.getenv("TB_AGENT_MAX_MODEL_TOKENS", "512"))
        max_same_command_repeats = int(os.getenv("TB_AGENT_MAX_SAME_COMMAND_REPEATS", "2"))

        system_prompt = (
            "You are an autonomous terminal agent operating in a container task. "
            "Your objective is to complete the task quickly and safely with short, high-value commands. "
            "Rules: avoid long-running or interactive commands (tail -f, watch, top, less, vim, nano). "
            "Use one command per step and only inspect files/dirs relevant to the task. "
            "If the task is done or cannot proceed due to a concrete blocker, return FINISH immediately. "
            "Do not write a thinking process, analysis, markdown, or prose. "
            "At each step, reply with exactly one line in one of these formats:\n"
            "CMD: <single shell command>\n"
            "FINISH: <brief completion summary>\n"
            "Never include explanations outside CMD/FINISH."
        )

        transcript: List[Dict[str, Any]] = []
        usage_input = 0
        usage_output = 0
        last_commands: List[str] = []
        model_call_latencies_ms: List[int] = []
        format_errors = 0

        async with httpx.AsyncClient(timeout=model_timeout_sec) as client:
            for step in range(1, max_steps + 1):
                user_payload = {
                    "instruction": instruction,
                    "step": step,
                    "history": transcript[-10:],
                    "response_format": "Reply only as CMD: <single shell command> or FINISH: <summary>.",
                }
                request_body = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(user_payload)},
                    ],
                    "temperature": 0.1,
                    "max_tokens": max_model_tokens,
                }
                if self._should_disable_thinking(model):
                    request_body["chat_template_kwargs"] = {"enable_thinking": False}

                model_call_started = datetime.utcnow()
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                if response.status_code >= 400 and request_body["max_tokens"] > 256:
                    request_body["max_tokens"] = 256
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=request_body,
                    )
                model_call_ms = int((datetime.utcnow() - model_call_started).total_seconds() * 1000)
                model_call_latencies_ms.append(model_call_ms)
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                usage_input += int(usage.get("prompt_tokens", 0) or 0)
                usage_output += int(usage.get("completion_tokens", 0) or 0)

                message = data["choices"][0]["message"]
                content = (message.get("content") or message.get("reasoning_content") or "").strip()
                if not content:
                    raise RuntimeError("Model returned empty action payload")

                try:
                    action, value = self._parse_action(content)
                    format_errors = 0
                except RuntimeError as exc:
                    format_errors += 1
                    transcript.append(
                        {
                            "step": step,
                            "format_error": str(exc),
                            "model_output": content[:max_obs_chars],
                            "correction": "Reply with exactly one line starting CMD: or FINISH:.",
                        }
                    )
                    if format_errors >= 3:
                        raise RuntimeError(f"Model failed to produce a valid action: {exc}") from exc
                    continue

                if action == "finish":
                    transcript.append({"step": step, "action": {"action": "finish", "summary": value}})
                    break
                if action != "exec":
                    raise RuntimeError(f"Unsupported model action: {action!r}")
                command = self._sanitize_command(value)

                last_commands.append(command)
                if len(last_commands) > max_same_command_repeats + 1:
                    last_commands = last_commands[-(max_same_command_repeats + 1) :]
                if (
                    len(last_commands) == max_same_command_repeats + 1
                    and len(set(last_commands)) == 1
                ):
                    transcript.append(
                        {
                            "step": step,
                            "repeat_skipped": command,
                            "correction": "Choose a different next command or reply FINISH.",
                        }
                    )
                    continue

                result = await environment.exec(command, timeout_sec=command_timeout_sec)
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
            "model_call_timeout_sec": model_timeout_sec,
            "command_timeout_sec": command_timeout_sec,
            "model_call_latencies_ms_tail": model_call_latencies_ms[-10:],
        }

    @staticmethod
    def _parse_action(content: str) -> Tuple[str, str]:
        """Parse model output into ('exec'|'finish', value)."""
        content = content.strip()
        lines = content.splitlines()
        action_index: int | None = None
        action = ""
        value = ""
        for idx, line in enumerate(lines):
            match = re.match(r"^\s*(CMD|FINISH):\s*(.*?)\s*$", line, flags=re.IGNORECASE)
            if match:
                action_index = idx
                action = match.group(1).upper()
                value = match.group(2).strip()
        if action_index is None:
            raise RuntimeError("Model output did not contain a CMD: or FINISH: line")

        if action == "CMD":
            value = LocalOpenAIAgent._collect_command(lines, action_index, value)
            if not value:
                raise RuntimeError("CMD action returned without a command")
            return ("exec", value)
        return ("finish", value.strip().strip("`") or "task finished")

    @staticmethod
    def _collect_command(lines: List[str], action_index: int, first_line: str) -> str:
        command = first_line.strip().strip("`")
        heredoc = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_-]*)['\"]?", command)
        collected = [first_line.rstrip()]
        if heredoc:
            terminator = heredoc.group(1)
            for line in lines[action_index + 1 :]:
                collected.append(line.rstrip())
                if line.strip() == terminator:
                    break
            return "\n".join(collected).strip().strip("`")

        if not LocalOpenAIAgent._has_unclosed_shell_quote(command):
            return command

        for line in lines[action_index + 1 :]:
            collected.append(line.rstrip())
            candidate = "\n".join(collected)
            if not LocalOpenAIAgent._has_unclosed_shell_quote(candidate):
                break
        return "\n".join(collected).strip().strip("`")

    @staticmethod
    def _has_unclosed_shell_quote(command: str) -> bool:
        try:
            shlex.split(command)
        except ValueError as exc:
            return "No closing quotation" in str(exc)
        return False

    @staticmethod
    def _should_disable_thinking(model: str) -> bool:
        setting = os.getenv("TB_AGENT_DISABLE_THINKING", "auto").strip().lower()
        if setting in {"1", "true", "yes", "on"}:
            return True
        if setting in {"0", "false", "no", "off"}:
            return False
        return "qwen" in model.lower()

    @staticmethod
    def _sanitize_command(command: str) -> str:
        command = command.strip()
        if not command:
            raise RuntimeError("Parsed empty command")

        blocked_patterns = [
            r"\bwatch\b",
            r"\btop\b",
            r"\bhtop\b",
            r"\bless\b",
            r"\bmore\b",
            r"\bvim\b",
            r"\bnano\b",
            r"\btail\s+-f\b",
            r"\bsleep\s+([3-9]\d{1,}|\d{3,})\b",
        ]
        for pattern in blocked_patterns:
            if re.search(pattern, command, flags=re.IGNORECASE):
                raise RuntimeError(f"Blocked potentially non-terminating command: {command!r}")
        return command
