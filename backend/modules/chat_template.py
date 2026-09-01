"""Chat template selection modes for llama-server."""

# Use GGUF-embedded chat template with --jinja only (no --chat-template-file).
TEMPLATE_EMBEDDED_JINJA = "__embedded_jinja__"


def is_embedded_jinja(selected: str | None) -> bool:
    return selected == TEMPLATE_EMBEDDED_JINJA


def is_custom_template_file(selected: str | None) -> bool:
    return bool(selected) and not is_embedded_jinja(selected)
