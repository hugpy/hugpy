"""Configuration for the hugpy discord bot (``hugpy_discord``).

Settings come from the environment. A ``.env`` is loaded from ``$HUGPY_BOT_ENV``
when set, otherwise python-dotenv searches up from the current working directory,
so ``hugpy bot`` picks up a local ``.env``. The discord token can also fall back
to a legacy darnell ``.env`` when present, so existing deployments keep working.
"""
import os
import re
from pathlib import Path

try:  # python-dotenv ships with the ``bot`` extra; without it only os.environ counts.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised when the extra is absent
    def load_dotenv(*_args, **_kwargs):
        return False

# Load a .env: explicit path wins, else search upward from CWD.
_EXPLICIT_ENV = os.getenv("HUGPY_BOT_ENV")
load_dotenv(_EXPLICIT_ENV) if _EXPLICIT_ENV else load_dotenv()

# Writable data dir for per-user prefs etc. — NOT inside the installed package.
DATA_DIR = Path(os.getenv("HUGPY_BOT_DATA") or (Path.home() / ".hugpy" / "bot"))
# Optional legacy darnell .env, kept only as a token fallback for old setups.
LEGACY_ENV_PATH = Path(os.getenv("HUGPY_BOT_LEGACY_ENV") or (Path.home() / "darnell" / ".env"))

# Default central is the local console/API (hugpy serve / dev.hugpy.ai → :7002).
# Resolved via the shared resolver so HUGPY_BASE_URL is canonical and the legacy
# aliases (HUGPY_CENTRAL/HUGPY_URL/WORKER_CENTRAL_URL) still redirect the bot too.
from hugpy_platform.central import central_base_url
HUGPY_BASE_URL = central_base_url()
DEFAULT_MODEL_KEY = os.getenv("DEFAULT_MODEL_KEY") or None
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None
# The operator's Discord user id. Escalation choice-clicks are fail-closed to
# this user: if unset (None) no one can answer via the buttons/select — the
# question stays answerable by text. Mirror of the GUILD_ID idiom above.
OPERATOR_DISCORD_ID = int(os.getenv("OPERATOR_DISCORD_ID")) if os.getenv("OPERATOR_DISCORD_ID") else None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# Privileged Server Members intent. Lets the bot enumerate guild members so the
# console can offer a user dropdown when binding. MUST also be toggled ON in the
# Discord Developer Portal (Bot → Privileged Gateway Intents → Server Members),
# or the bot fails to connect — hence default-off and env-gated.
MEMBERS_INTENT = _env_flag("HUGPY_BOT_MEMBERS_INTENT", False)

# Discord caps a single message at 2000 chars; leave headroom for markdown.
MESSAGE_CHAR_LIMIT = 1900
# Largest attachment we will pull from Discord and forward to hugpy.
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
# Conversation turns kept per channel for chat context.
HISTORY_MAX_TURNS = 20


def _read_legacy_token(name: str) -> str | None:
    if not LEGACY_ENV_PATH.is_file():
        return None
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$")
    for line in LEGACY_ENV_PATH.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def get_discord_token() -> str:
    token = os.getenv("DISCORD_TOKEN") or _read_legacy_token("darnell_token")
    if not token:
        raise RuntimeError(
            "No discord token: set DISCORD_TOKEN in the environment (or a .env in "
            "the working dir, or $HUGPY_BOT_ENV), or keep darnell_token in "
            f"{LEGACY_ENV_PATH}"
        )
    return token
