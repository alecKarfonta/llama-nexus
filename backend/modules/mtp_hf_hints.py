"""Heuristics for Hugging Face repos/files that may ship MTP-converted GGUFs.

File metadata is authoritative; repo/filename hints are for search UI only.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

MTP_GGUF_CAVEAT = (
    "MTP requires GGUF files converted with prediction heads. Standard GGUFs of "
    "MTP-trained model families usually do not include those tensors — verify with "
    "mtp_capable after download or check for an explicit MTP conversion repo."
)

# Known publishers / repo id patterns for MTP-converted GGUF releases (hint only).
MTP_REPO_ID_PATTERNS = (
    re.compile(r"(?i)-mtp-gguf"),
    re.compile(r"(?i)-mtp-gguf/"),
    re.compile(r"(?i)/.*-MTP-GGUF"),
    re.compile(r"(?i)Qwen3\.6-.*-MTP-GGUF"),
    re.compile(r"(?i)unsloth/.*MTP"),
)

MTP_FILENAME_PATTERNS = (
    re.compile(r"(?i)-mtp[-_]"),
    re.compile(r"(?i)_mtp[-_]"),
    re.compile(r"(?i)mtp\.gguf"),
)


def repo_likely_mtp_converted(repo_id: str) -> bool:
    if not repo_id:
        return False
    return any(pattern.search(repo_id) for pattern in MTP_REPO_ID_PATTERNS)


def filename_suggests_mtp(filename: str) -> bool:
    if not filename or not filename.lower().endswith(".gguf"):
        return False
    return any(pattern.search(filename) for pattern in MTP_FILENAME_PATTERNS)


def enrich_repo_files_response(repo_id: str, files: List[str]) -> Dict[str, Any]:
    """Augment repo file listing with non-authoritative MTP hints."""
    file_hints: Dict[str, Dict[str, Any]] = {}
    for name in files:
        if filename_suggests_mtp(name):
            file_hints[name] = {
                "filename_suggests_mtp": True,
                "note": "Filename suggests an MTP conversion; confirm after download via mtp_capable.",
            }
    return {
        "files": files,
        "mtp": {
            "repo_likely_converted": repo_likely_mtp_converted(repo_id),
            "caveat": MTP_GGUF_CAVEAT,
        },
        "file_hints": file_hints,
    }
