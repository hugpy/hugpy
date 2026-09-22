# Allocation modes — where your model's memory lives

Every model gets **one of five modes** that says how its weights (and KV
context) are placed across the GPU (VRAM) and system RAM. Pick the intent; the
fleet does the math.

> **The promise:** a *slow* mode you chose is yours; a model with *no working
> mode at all* is our bug. The default never OOMs — a blank model serves.

## The five modes

| mode | what it does | when it refuses ("bust") |
|---|---|---|
| **gpu-only** | Every layer on the GPU, no spill. Fastest when it fits. | Won't fit the GPU (even after evicting idle residents) → refuse. Never a half-load. |
| **ram-only** | Everything in system RAM; the GPU is never touched (binds the CPU even when a GPU is present). | Won't fit RAM → refuse. |
| **max-gpu** | *The default.* As much GPU as is available and needed, the rest spills to RAM. Zero knobs — "use my GPU, spill the rest". | Only when GPU + RAM together can't hold it. |
| **max-ram** | Mirror of max-gpu: fill RAM first, only the overflow rides the GPU. For keeping cards free while big models stay warm. | Only when RAM + GPU together can't hold it. |
| **explicit** | You set the targets: a VRAM and/or RAM budget, a **leniency %**, and a **priority device** (`gpu` default, or `ram`), plus the tolerance bands/priority you already know. | Can't hit the target *even at the loosened floor* → refuse. |

`max-gpu` and `max-ram` are internally explicit-with-generous-leniency — but
they stay flat, named modes on purpose: simple intent gets a simple pick; the
moment you want to tune numbers, you reach for `explicit`.

## Leniency — how much off-device you tolerate

**N% leniency = up to N% of the MODEL may land off its ideal device before we
refuse.**

Example: `100% GPU target + 30% leniency`
- Try 100% on GPU first.
- Under pressure, degrade step by step — 90/10, 80/20 … — down to the **floor:
  70% GPU / 30% RAM**.
- If even 70/30 won't fit, the load refuses **honestly**, naming the mode and
  the floor (e.g. *"explicit: even the loosened floor won't fit — target 100%
  of the model on GPU with 30% leniency gives a floor of 70% …"*).

No leniency set = exact target or refuse (no undeclared tolerance is ever
assumed). Degradation only happens under real contention — a model that fits
at its target always gets its target.

## Engine coverage

Two transformers families load differently, so they honor the modes by
different mechanisms:

- **transformers TEXT** (`AutoModelFor*`: the causal-LM generator, the flan /
  seq2seq / summarization back-ends, the vision-language coder) place weights
  with **accelerate** — `device_map="auto"` + a `max_memory` budget map. Every
  mode maps onto that budget map.
- **diffusers** (text-to-image / image-to-image pipelines) do **not** take
  `device_map`/`max_memory`. They spill via diffusers' own **CPU-offload API**
  instead (see the note below the table).

| mode | GGUF (llama.cpp) | transformers TEXT (accelerate) | diffusers (image) |
|---|---|---|---|
| gpu-only | ✅ | ✅ whole model on the card (no CPU budget) | ✅ `.to(cuda)` |
| ram-only | ✅ | ✅ CPU placement (gpu budget 0) | ✅ `enable_sequential_cpu_offload()` |
| max-gpu | ✅ | ✅ accelerate fit-and-spill | ✅ `.to(cuda)` (auto/default) |
| max-ram | ✅ | ✅ RAM-priority `max_memory` [^purerram] | ✅ `enable_model_cpu_offload()` |
| explicit | ✅ | ❌ GGUF-only (banded leniency floor has no transformers analogue) | ❌ GGUF-only (same rationale) |

[^purerram]: **Pure-RAM caveat.** Without a resolved `model_need_bytes` the
transformers loaders take the SAFE pure-RAM path (the whole model on the CPU) —
they do not yet compute the RAM-first / GPU-overflow split. A follow-up may add
overflow-to-GPU sizing; until then max-ram on a non-GGUF text model means "RAM
first" without spilling the remainder onto the card when the size is unknown.

**Loader-level vs central gate.** Slice C wired the transformers gap loaders to
the spill seam, so at the WORKER a transformers text model honors max-ram
(RAM-priority `max_memory`) and the gpu-only / ram-only / max-gpu placement
intents. **Central now accepts max-ram for non-GGUF models** (operator-approved
2026-07-24): the engine gate at `/assign` was opened for the max-ram *mode*
because all three gap loaders honor it (text via `transformers_max_memory`'s
RAM-priority branch, imagegen via `enable_model_cpu_offload`). **`explicit`
stays GGUF-only at the central gate** — its banded leniency floor is a
llama.cpp concept with no transformers analogue, so admitting it would break the
mode's promise. The remaining GGUF-only knobs (`gpu_mem_gib` / `cpu_mem_gib` /
`threads` / `tensor_split` / the t21 bands / MoE `n_cpu_moe`) stay refused for
non-GGUF exactly as before.

**diffusers offload mechanism.** A diffusers pipeline can't shard by a
`max_memory` map, so the honest spill is diffusers' offload API, chosen from the
SAME placement seam (`spill.n_gpu_layers_intent` / `alloc_mode_env`):
- CPU-leaning intent (ram-only / `n_gpu_layers` `off`/`0`) →
  `enable_sequential_cpu_offload()` — submodules stream one at a time,
  smallest VRAM footprint (slowest).
- max-ram → `enable_model_cpu_offload()` — whole submodules ride RAM, pulled to
  the card only while active (big model, one consumer GPU, no OOM).
- gpu-only / auto / no intent → today's `.to(cuda)` — byte-identical when the
  seam is silent.
- A pipeline class without the requested offload method is a **genuine
  capability gap**: it is logged once and falls back to `.to(cuda)` rather than
  silently ignoring the mode.

`explicit`'s numeric leniency floor is a llama.cpp-only concept with no
transformers/diffusers analogue, so `explicit` is refused for non-GGUF at the
central gate and is not a reachable live path for transformers. (Were the gate
opened, a transformers/diffusers model under explicit would get only the
budgeted fit-and-spill, not the banded degrade-to-floor — which is exactly why
the mode's promise can't be kept there and the gate stays closed.)

## MoE models — the expert split (measured, on by default)

A Mixture-of-Experts GGUF is mostly **cold expert tensors**: on
Qwen3-Coder-Next (80B-A3B class, 512 experts / 10 used per token) the expert
FFN tensors are ~43.6 GiB of the ~45.1 GiB file — the always-hot
attention/shared/KV share is only ~1.5 GiB. Splitting **by kind of bytes**
instead of by whole layers changes everything:

| plan (ae / RTX 3090, coder-next) | tok/s | VRAM |
|---|---|---|
| naive layer split (autofit, 17/48 layers) | ~15.2 | 16.6 GiB |
| **MoE split** (`n_gpu_layers=-1` + `--n-cpu-moe 999`) | **~24.1** | **3.2 GiB** |

+59% throughput AND 5× less VRAM — a strict improvement, so it is the
**default** for the hybrid case:

- **Auto policy** (no knobs): a detected-MoE GGUF whose whole file does **not**
  fit the card (exactly when autofit would have produced a partial layer
  split) serves with all layers on the GPU and the expert tensors on CPU.
  *Whole model fits → fully on GPU, unchanged. Dense model → byte-identical to
  before (no expert tensors, nothing to split).* Detection is the GGUF header
  the loader already parses: `expert_count > 0` is a MoE; the expert tensors
  are the `blk.<i>.ffn_*_exps.*` names (`_exps` is the per-tensor expert bit).
- **The knob**: `n_cpu_moe` — a first-class serve override / spill key (like
  `n_gpu_layers`): the number of MoE layers whose expert tensors stay on CPU
  (`999` = all, `0` = experts on GPU). An explicit value always wins over the
  auto policy, and explicit `n_gpu_layers` designations (gpu-only/ram-only or
  a layer count) always win over the auto split.
- **Fit/feasibility price the split**: probes, the ⭐ boot load, admission and
  central's feasibility judge a MoE by its **non-expert GPU share** (+KV) when
  the split governs — a 41.6 GB MoE on an empty 24 GB card *fits* (needs
  ~3 GiB of VRAM); its expert bytes are checked against host RAM instead.
- **Engine**: needs a native `llama-server` with `--n-cpu-moe` (probed once;
  an older binary or the `llama_cpp.server` python fallback degrades to the
  layer-split behavior with one log line — never a crash).

**Coexistence note**: ae's worker unit currently carries
`LLAMA_ARG_N_CPU_MOE=999` (the transition-era env hack that proved the win).
Explicit `--n-cpu-moe` argv beats the env in llama-server, so this feature is
correct with or without it — the env line can be removed once this ships.

## Legacy names (the honest rename)

Old controls were misleadingly named; existing settings keep working — old
names are accepted on input, resolved, and never written back:

| old name / setting | is now | why |
|---|---|---|
| **autofit** ({} / no knobs) | **max-gpu** | it always was "as much GPU as fits, spill the rest" |
| **Max GPU** (`n_gpu_layers: -1`) | **gpu-only** | it was all-or-OOM, not "maximize" — now it refuses instead of crashing |
| **CPU only** (`n_gpu_layers: "off"`) | **ram-only** | RAM is what it actually binds |
| **budget** / **bands** (chaos) | **explicit** | a budget/band IS an explicit allocation |

Nothing on the wire changed for these three: `gpu-only` still rides
`n_gpu_layers: -1`, `ram-only` still rides `"off"`, `max-gpu` is still `{}`.
Your `serve_overrides.json` is never rewritten — the mode is **derived at read
time** (`-1` → gpu-only, `0/"off"` → ram-only, budgets/bands → explicit, unset
→ max-gpu).

## Old workers

`max-ram` and `explicit` ride new spill keys (`alloc_mode`, `leniency_pct`,
`priority_device`), and the MoE `n_cpu_moe` knob rides the same gate.
Central only emits them to workers running **0.1.203+**;
an older worker gets max-gpu (autofit) for that request and the downgrade is
logged — your persisted choice is kept and applies the moment the worker
updates. A selected mode is never a silent dead knob.

## Protection is unchanged

Modes reorder *placement*, never protection: 🔒 static residents and models
actively answering are never evicted to make room, whatever mode the incoming
model carries.
