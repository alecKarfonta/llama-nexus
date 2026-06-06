"""Parse llama-server log lines for MTP / speculative decoding statistics."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

# draft acceptance rate = 0.76482 ( 3483 accepted / 4554 generated)
_ACCEPTANCE_RATE_RE = re.compile(
    r"draft\s+acceptance\s+rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\s*\)",
    re.IGNORECASE,
)
_AGGREGATE_ACCEPT_RE = re.compile(
    r"aggregate_accept_rate\s*[=:]\s*([0-9.]+)",
    re.IGNORECASE,
)
_DRAFT_ACC_RE = re.compile(
    r"\bacc\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)
_DRAFT_SUMMARY_RE = re.compile(
    r"draft\s*=\s*(\d+)\s+acc\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)


def parse_mtp_stats_from_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Return structured MTP stats if the log line contains them."""
    if not line or "draft" not in line.lower():
        return None

    match = _ACCEPTANCE_RATE_RE.search(line)
    if match:
        rate = float(match.group(1))
        accepted = int(match.group(2))
        generated = int(match.group(3))
        return {
            "acceptance_rate": rate,
            "tokens_accepted": accepted,
            "tokens_drafted": generated,
            "source": "draft_acceptance_rate",
            "raw": line.strip(),
        }

    match = _AGGREGATE_ACCEPT_RE.search(line)
    if match:
        return {
            "acceptance_rate": float(match.group(1)),
            "source": "aggregate_accept_rate",
            "raw": line.strip(),
        }

    match = _DRAFT_SUMMARY_RE.search(line)
    if match:
        return {
            "tokens_drafted": int(match.group(1)),
            "acceptance_rate": float(match.group(2)),
            "source": "draft_summary",
            "raw": line.strip(),
        }

    if "acc=" in line.lower():
        match = _DRAFT_ACC_RE.search(line)
        if match:
            return {
                "acceptance_rate": float(match.group(1)),
                "source": "draft_acc",
                "raw": line.strip(),
            }

    return None
