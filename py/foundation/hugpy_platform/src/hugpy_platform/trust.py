"""Publisher trust tiers for Hugging Face repos.

Lifted out of flask_app/app/routes/search_routes.py so that code with no Flask
context — the CLI, the review timer, any worker — can ask the same question and
get the same answer. search_routes imports from here and keeps its own names
bound, so nothing that referenced them moves.

HF exposes no "trust rating", so trust here is a hand-kept allowlist of who
published the repo. TIER-1 = the canonical FIRST-PARTY orgs that originate the
weights (a Llama from meta-llama, a FLUX from black-forest-labs — the real
thing, not a reupload). TIER-2 = reputable community REPACKAGERS/quantizers
whose GGUF/mirror repos are broadly relied on. Everyone else is untrusted (0):
not "bad", just unvetted. Match is on the repo OWNER (org), case-insensitive.
Add names here as the fleet's trusted sources grow — this is the one place.
Downloads/likes are deliberately NOT trust (they're gameable popularity); trust
outranks them so a canonical repo beats a more-liked fork with the same name.
"""

TRUST_TIER1 = frozenset(s.lower() for s in (
    # LLM / multimodal first-party
    "meta-llama", "Qwen", "google", "google-bert", "mistralai", "deepseek-ai",
    "microsoft", "openai", "openai-community", "nvidia", "HuggingFaceTB",
    "HuggingFaceM4", "allenai", "tiiuae", "01-ai", "CohereForAI", "CohereLabs",
    "ibm-granite", "databricks", "MiniMaxAI", "moonshotai", "zai-org", "THUDM",
    "inclusionAI", "ByteDance-Seed", "rhymes-ai", "internlm", "baichuan-inc",
    "facebook", "EleutherAI", "bigcode", "bigscience", "xai-org", "servicenow",
    # image / video / audio first-party
    "stabilityai", "black-forest-labs", "Wan-AI", "tencent", "genmo",
    "Lightricks", "PixArt-alpha", "playgroundai", "Efficient-Large-Model",
    "ByteDance", "Kwai-Kolors", "suno", "coqui", "laion", "openbmb",
))

TRUST_TIER2 = frozenset(s.lower() for s in (
    # reputable community quantizers / mirrors (GGUF & friends)
    "bartowski", "TheBloke", "unsloth", "city96", "mradermacher", "ggml-org",
    "lmstudio-community", "NousResearch", "cognitivecomputations", "bullerwins",
    "Mungert", "second-state", "QuantFactory", "MaziyarPanahi", "DevQuasar",
    "bfloat16", "featherless-ai-quants", "legraphista", "nightmedia",
    "calcuis", "Comfy-Org",
))


def trust_tier(hub_id: str, author=None) -> int:
    """2 = canonical first-party publisher, 1 = reputable repackager, 0 = unvetted.
    Owner = explicit ``author`` if the Hub gave one, else the org before the '/'."""
    org = (author or (hub_id or "").split("/", 1)[0] or "").lower()
    if org in TRUST_TIER1:
        return 2
    if org in TRUST_TIER2:
        return 1
    return 0


def trust_label(tier: int):
    """UI-facing label for a trust tier (None = unvetted, no badge)."""
    return {2: "first-party", 1: "community"}.get(tier)
