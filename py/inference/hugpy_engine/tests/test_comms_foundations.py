"""F2 principals + F4 settings + F3.1 model_meta unit checks."""
import sys, tempfile, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = 0
def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print(f"  ok - {name}")


def test_comms_foundations():
    """Converted from the script-style suite: every `check` is an assert."""

    tmp = tempfile.mkdtemp(prefix="comms-found-")

    # ---------------- F2 principals ----------------
    from hugpy_control.principals import PrincipalStore, Principal, allowed

    ps = PrincipalStore(os.path.join(tmp, "principals.json"))
    p = ps.create(kind="user", name="alice", groups=["media"])
    check("create principal", p.id.startswith("pr_") and p.active)
    check("get round-trips", ps.get(p.id).name == "alice")

    tok = ps.issue_token(p.id)
    check("token minted", tok and tok.startswith("hpp_"))
    check("token resolves", ps.resolve_token(tok).id == p.id)
    check("bad token is None", ps.resolve_token("hpp_deadbeef") is None)

    # groups/policy
    op = ps.create(kind="operator", name="root", groups=["operators"])
    check("operator passes any action", allowed(op, "pip.install"))
    check("media group passes media", allowed(p, "media"))
    check("media group fails workers.manage", not allowed(p, "workers.manage"))
    check("wildcard chat allows any principal", allowed(p, "chat"))
    check("unknown action fails closed", not allowed(p, "wat.unknown"))
    check("None principal fails", not allowed(None, "chat"))

    # discord link (DISC-05)
    linked = ps.link_discord(tok, "111222333")
    check("discord link binds snowflake", linked.discord_user_id == "111222333")
    check("for_discord_user finds it", ps.for_discord_user("111222333").id == p.id)
    check("link with bad token is None", ps.link_discord("hpp_nope", "999") is None)

    # expiry + revocation
    short = ps.create(kind="ephemeral", name="tmp", expires_in=0.05)
    time.sleep(0.1)
    check("expired principal inactive", not ps.get(short.id).active)
    ps.revoke(p.id)
    check("revoked principal inactive", not ps.get(p.id).active)
    check("revoked principal's token dead", ps.resolve_token(tok) is None)

    # implicit principals (legacy credential attribution)
    imp = ps.resolve_api_key("abc123", "my key")
    check("implicit api-key principal", imp.id == "apikey:abc123"
          and "api" in imp.groups)

    # ---------------- F4 settings ----------------
    from hugpy_control.settings import SettingsStore

    ss = SettingsStore(os.path.join(tmp, "settings.json"))
    events = []
    ss.on_change = lambda ns, key, value: events.append((ns, key))

    ss.set("discord.channels", "123", {"respond": "all"})
    check("set/get", ss.get("discord.channels", "123") == {"respond": "all"})
    ss.merge("discord.channels", "123", {"personality": "pirate"})
    check("merge keeps + adds",
          ss.get("discord.channels", "123") == {"respond": "all",
                                                "personality": "pirate"})
    ss.merge("discord.channels", "123", {"respond": None})
    check("merge None deletes field",
          ss.get("discord.channels", "123") == {"personality": "pirate"})
    check("all(ns)", "123" in ss.all("discord.channels"))
    check("namespaces", "discord.channels" in ss.namespaces())
    check("change events fired", len(events) == 3)
    check("delete", ss.delete("discord.channels", "123")
          and ss.get("discord.channels", "123") is None)

    # cross-instance (simulating two processes): TTL cache respected
    ss2 = SettingsStore(os.path.join(tmp, "settings.json"))
    ss.set("personalities", "pirate", {"system": "Ye be a pirate.",
                                       "model_key": None, "params": {}})
    check("second store sees write (fresh read)",
          "pirate" in ss2.all("personalities"))

    # ---------------- F3.1 model_meta ----------------
    from hugpy_engine.config.models.model_meta import parse_quant, parse_params_b, recommended_settings

    check("quant q4_k_m", parse_quant("qwen2.5-3b-instruct-q4_k_m.gguf") == "q4_k_m")
    check("quant iq3_xs", parse_quant("model.IQ3_XS.gguf") == "iq3_xs")
    check("quant f16", parse_quant("weights-f16.gguf") == "f16")
    check("quant none for safetensors", parse_quant("model.safetensors") is None)
    check("params 3B", parse_params_b("Qwen2.5-3B-Instruct") == 3.0)
    check("params 0.5b", parse_params_b("Qwen2.5-0.5B-Instruct") == 0.5)
    check("params none", parse_params_b("all-minilm-l6-v2") is None)
    check("year not params", parse_params_b("model-2024") is None)

    GIB = 1024 ** 3
    r = recommended_settings(size_bytes=2 * GIB, ctx_max=32768,
                             framework="llama_cpp", vram_bytes=8 * GIB)
    check("fits -> all layers", r["n_gpu_layers"] == -1 and r["fits_vram"])
    r2 = recommended_settings(size_bytes=40 * GIB, ctx_max=8192,
                              framework="llama_cpp", vram_bytes=8 * GIB)
    check("too big -> partial + honest reason",
          r2["fits_vram"] is False and "gpu_fraction" in r2)
    r3 = recommended_settings(size_bytes=None, ctx_max=4096)
    check("unknown size -> honest none", r3["fits_vram"] is None
          and "unknown" in r3["reason"])

    print(f"\nALL {ok} CHECKS PASSED")
