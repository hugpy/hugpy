"""Transport-neutral result objects produced by the core command layer.

A core command function talks to hugpy central and returns a :class:`CommandResult`
describing *what* to deliver, never *how*. Each transport (the Discord cogs, the
chatshare adapter) renders a ``CommandResult`` in its own idiom — Discord into
embeds / ``discord.File`` / followups, chatshare into ``msg.send`` + ``file.put``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutFile:
    """A file to attach to a reply (produced in-memory by a core function)."""
    filename: str
    data: bytes
    mime: str = "application/octet-stream"


@dataclass
class CommandResult:
    """Everything a transport needs to render one command's reply.

    Exactly one shape is usually populated, but the renderer handles any mix:
    - ``text``     — the main body; ``long`` asks for the split/attach treatment.
    - ``files``    — attachments (images, JSON blobs, long-text fallbacks).
    - ``title`` / ``sections`` / ``footer`` — an "embed"-style structured reply
      (Discord renders a real embed; chatshare renders bold headings + text).
    - ``error``    — True when ``text`` is a failure notice (⚠️).
    - ``ephemeral``— the reply is for the caller's eyes only.
    """
    text: str = ""
    files: list[OutFile] = field(default_factory=list)
    title: str | None = None
    sections: list[tuple[str, str]] = field(default_factory=list)
    footer: str | None = None
    color: str = "blurple"            # "blurple" | "red"
    long: bool = False                # render text with the split/attach policy
    filename: str = "result.txt"      # attachment name when `long` overflows
    ephemeral: bool = False
    error: bool = False
    # Old-central compatibility: when central predates POST /prompt, stream this
    # prompt through /chat/stream instead (the transport owns the streaming).
    fallback_prompt: str | None = None
    fallback_file: str | None = None

    @classmethod
    def fail(cls, message: str, *, ephemeral: bool = False) -> "CommandResult":
        return cls(text=message, error=True, ephemeral=ephemeral)
