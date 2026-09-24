from hugpy_engine.wire.install_schemas import InstallOption, InstallOptions
from hugpy_platform.constants import GGUF_QUANT
from hugpy_platform.utils import is_mmproj_file

def _human(n):
    if not n: return ""
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def _gguf_options(files, free_bytes):
    groups: dict[str, list] = {}
    for f in files:
        if not f.path.lower().endswith(".gguf"):
            continue
        # A vision projector is never a quant to install: it rides along as a
        # sidecar of whatever quant is chosen (download_one fetches it). Its
        # name often carries a quant tag (``mmproj/<model>-vision-Q6_K.gguf``),
        # and offering it as "GGUF · Q6_K" is how central came to hold ONLY
        # projectors for zerodigest/Qwen3.8-27B-Uncensored-YMQ-MTP-GGUF
        # (2026-09-23).
        if is_mmproj_file(f.path):
            continue
        m = GGUF_QUANT.search(f.path)
        groups.setdefault(m.group(0).upper() if m else f.path, []).append(f)

    opts = []
    for label, group in sorted(groups.items()):
        total = sum(g.size for g in group if g.size) or None
        common = dict(
            id=f"gguf:{label}",
            framework="gguf",
            total_bytes=total,
            fits_disk=(None if total is None or free_bytes is None
                       else total < free_bytes),
        )
        if len(group) == 1:
            opts.append(InstallOption(
                label=f"GGUF · {label} · {_human(total)}",
                filename=group[0].path, **common))
        else:
            opts.append(InstallOption(           # sharded → glob, not filename
                label=f"GGUF · {label} · {len(group)} shards · {_human(total)}",
                include=[f"*{label}*.gguf"], **common))
    return opts

def _transformers_option(files, free_bytes):
    has_st  = any(f.path.endswith(".safetensors") for f in files)
    has_bin = any(f.path.endswith(".bin") for f in files)
    if not (has_st or has_bin):
        return None
    if has_st and has_bin:                        # skip duplicate .bin weights
        include = ["*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt"]
        total = sum(f.size for f in files if f.size and (
            f.path.endswith((".safetensors", ".json", ".model", ".txt"))
            or "tokenizer" in f.path)) or None
        label = f"Transformers · safetensors only · {_human(total)}"
    else:
        include, total = None, (sum(f.size for f in files if f.size) or None)
        label = f"Transformers · full snapshot · {_human(total)}"
    return InstallOption(
        id="transformers", framework="transformers", label=label,
        include=include, total_bytes=total,
        fits_disk=(None if total is None or free_bytes is None
                   else total < free_bytes),
    )

def resolve_options(hub_id, task, files, free_bytes) -> InstallOptions:
    opts = _gguf_options(files, free_bytes)
    tf = _transformers_option(files, free_bytes)
    if tf:
        opts.append(tf)
    # prefer a mid GGUF quant, else transformers
    rec = next((o.id for o in opts if o.id == "gguf:Q4_K_M"),
               opts[0].id if opts else None)
    return InstallOptions(hub_id=hub_id, task=task, options=opts, recommended=rec)
