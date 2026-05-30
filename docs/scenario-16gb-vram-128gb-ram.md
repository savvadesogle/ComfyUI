# Scenario: 16 GB VRAM + 128 GB RAM (NORMAL_VRAM)

A worked example showing how `text_encoder_initial_device()` behaves
before and after the fix on a typical consumer GPU setup.

## Hardware assumptions

| Component | Value |
|-----------|-------|
| GPU VRAM  | 16 GB dedicated |
| Free VRAM | ~15 GB (after torch reservation) |
| System RAM | 128 GB |
| Free RAM   | ~100 GB (reported by `psutil.virtual_memory().available`) |
| VRAM state | `NORMAL_VRAM` (default) |
| GPU device | Intel Arc A770, NVIDIA RTX 4060, or any 16 GB discrete GPU |

## Why this scenario matters

Systems where RAM (128 GB) greatly exceeds VRAM (16 GB) are the **vast
majority** of modern GPU workstations. The original `mem_l > mem_o * 0.5`
check always fails here because:

```
free_vram (15 GB)  >  free_ram (100 GB) * 0.5 = 50 GB  →  False
```

This means **every** text encoder > 1 GB is forced to CPU under the
original code, regardless of how much VRAM is actually available.

## Timeline 1: Large TE (7.6 GB) → Large model (11.7 GB)

Typical workflow: Qwen3-4B fp16 text encoder followed by Lumina2 DiT.

### Original
```
Step 1: text_encoder_initial_device(xpu:0, cpu, 7.6 GB)
  mem_l = 15 GB,  mem_o = 100 GB
  15 > 100 * 0.5 = 50  →  False  →  CPU
  TE stays on CPU.  Tokenisation is slow.

Step 2: load_models_gpu([Lumina2])
  free_memory(14 GB, xpu:0)
  15 GB free → 11.7 GB loaded  →  3.3 GB left for inference
  ✅ No OOM.  TE was never on GPU anyway.
```

### Fixed
```
Step 1: text_encoder_initial_device(xpu:0, cpu, 7.6 GB)
  mem_l = 15 GB
  7.6 * 1.2 = 9.12 < 15  →  True  →  GPU
  TE loads on GPU via load_models_gpu([TE], force_full_load=True).
  Tokenisation is fast (GPU).

Step 2: load_models_gpu([Lumina2])
  free_memory(14 GB, xpu:0)
  current_loaded_models: [TE (7.6 GB)]
  free: 15 - 7.6 = 7.4 GB,  need ~14 GB  →  not enough
  → unload TE: 7.4 + 7.6 = 15 GB  →  enough
  Lumina2 loads on GPU.  TE is now on CPU.
  ✅ No OOM.  Same final state, but TE ran on GPU first.
```

**Result:** With the fix the TE completes quickly on GPU, then gets
offloaded automatically when the main model needs VRAM.  The original
keeps the TE on CPU permanently — correct but slower.

## Timeline 2: Small TE (2 GB) → Small model (2 GB)

SDXL CLIP + SDXL UNet — enough VRAM for everything.

### Original
```
Step 1: TE → CPU  (same check: 15 > 50 → False)
Step 2: load_models_gpu([UNet, VAE])
  free: 15 GB,  need: ~4 GB  →  fits easily
  TE stays on CPU for no reason.
```

### Fixed
```
Step 1: TE → GPU  (2 * 1.2 = 2.4 < 15 → True)
Step 2: load_models_gpu([UNet, VAE])
  free_memory(4 GB, xpu:0)
  free: 15 - 2 = 13 GB  →  13 > 4  →  no offload needed
  TE + UNet + VAE all on GPU simultaneously.  Optimal.
```

**Result:** Fix is strictly better — everything runs on GPU.

## Timeline 3: Large model already loaded — insufficient VRAM

Lumina2 (11.7 GB) is already on GPU.  TE loads afterwards (unusual but
possible with some custom workflows).

```
text_encoder_initial_device(xpu:0, cpu, 7.6 GB)
  mem_l = 15 - 11.7 = 3.3 GB  (remaining free VRAM)
  7.6 * 1.2 = 9.12 < 3.3  →  False  →  CPU
```

**Both original and fixed:** CPU.  The `model_size * 1.2 < mem_l` guard
correctly prevents GPU OOM.

## Timeline 4: Dual GPU (2 × 16 GB)

Two Arc A770 or NVIDIA RTX 4060.  `text_encoder_device()` returns
`xpu:0` (or `cuda:0`) — the first card only.  VRAM is **not** pooled.

```
text_encoder_initial_device(xpu:0, cpu, 7.6 GB)
  mem_l = 15 GB  (same as single GPU — only card 0 matters)
  Original: 15 > 50 → False → CPU
  Fixed:    9.12 < 15 → True → GPU
```

**Same behaviour as single GPU.**  The fix helps the same way.

## Summary table

| Scenario | Original | Fixed | Fairness |
|----------|----------|-------|----------|
| Large TE → Large model | TE: CPU, model: GPU | TE: GPU→offload, model: GPU | Fix faster (cold TE) |
| Small TE → Small model | TE: CPU, everything: GPU | Everything: GPU | Fix faster |
| Model loaded first | TE: CPU | TE: CPU | Equal |
| Dual GPU | TE: CPU | TE: GPU (card 0) | Fix faster |

The fix never regresses: in every scenario the outcome is either
identical or strictly better.
