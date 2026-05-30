"""Simulate text_encoder_initial_device() before and after the fix.

Models the decision logic for various VRAM, RAM, model size, and
workload scenarios.  Shows whether the original or fixed code places
the text encoder on GPU or CPU.
"""

# ── scenario definitions ──────────────────────────────────────────────

VRAMS_GB = [4, 6, 8, 12, 16, 20, 24, 32, 48]
RAMS_GB  = [8, 16, 32, 64, 128, 256]

# (label, model_size_gb)  —  sizes in gibibytes
MODELS = [
    ("SD1.5 CLIP fp32",       1.3),
    ("SDXL CLIP",             2.0),
    ("Qwen3-4B fp16",         7.6),   # user's scenario
    ("Qwen3-4B fp32",        15.2),
    ("Llama-8B fp16",        15.0),
    ("Flux text encoders",    8.5),
]

# (label, vram_already_used_gb) — how much VRAM is occupied *before*
# the text encoder is considered.
SCENARIOS = [
    ("clean (nothing loaded)",   0.5),
    ("+ small diffusion (7 GB)", 7.5),
    ("+ Lumina2 (11.7 GB)",     12.2),   # ≈ user's real scenario
    ("+ large model (15 GB)",   15.5),
]

HEADROOM = 1.2            # 20 % safety margin
OVERHEAD_GB = 0.5         # OS / PyTorch allocator reservation
CPU_AVAIL_RATIO = 0.80    # fraction of total RAM reported as "available"


def original(free_vram_gb: float, free_ram_gb: float, model_gb: float) -> str:
    """Decision of the **original** (buggy) code."""
    if free_vram_gb > free_ram_gb * 0.5 and model_gb * HEADROOM < free_vram_gb:
        return "GPU"
    return "CPU"


def fixed(free_vram_gb: float, free_ram_gb: float, model_gb: float) -> str:
    """Decision of the **fixed** code."""
    if model_gb * HEADROOM < free_vram_gb:
        return "GPU"
    return "CPU"


def simulate():
    rows = []

    for model_label, model_gb in MODELS:
        for scenario_label, vram_used_gb in SCENARIOS:
            for vram_total_gb in VRAMS_GB:
                for ram_total_gb in RAMS_GB:
                    free_vram = vram_total_gb - vram_used_gb
                    free_ram  = ram_total_gb * CPU_AVAIL_RATIO
                    if free_vram <= 0:
                        continue          # would OOM before TE is even considered

                    old = original(free_vram, free_ram, model_gb)
                    new = fixed(free_vram, free_ram, model_gb)
                    changed = old != new

                    rows.append((
                        model_label, model_gb,
                        scenario_label,
                        vram_total_gb, ram_total_gb,
                        round(free_vram, 1), round(free_ram, 0),
                        old, new, changed,
                    ))

    # ── summary section: only rows where behaviour changed ──────────
    changed_rows = [r for r in rows if r[-1]]
    changed_rows.sort(key=lambda r: (r[0], r[2], r[3], r[4]))

    print("=" * 140)
    print("text_encoder_initial_device() - ORIGINAL vs FIXED - simulation")
    print("=" * 120)
    print()
    print(f"  Precondition:  aimdo_enabled=False, load_device != offload_device, model_size > 1 GB")
    print(f"  Safety margin:  model_size * {HEADROOM}  <  free_vram")
    print(f"  RAM available:  {CPU_AVAIL_RATIO*100:.0f} % of total")
    print(f"  Overhead:       {OVERHEAD_GB} GB reserved on GPU")
    print()

    # full table
    print("-" * 95)
    print(f"{'Model':<20} {'Scenario':<30} {'VRAM':>5} {'RAM':>5} {'FreeV':>6} {'FreeR':>6} {'Orig':>5} {'Fix':>5}  {'D'}")
    print("-" * 95)
    for r in rows:
        mark = " <" if r[-1] else ""
        print(f"{r[0]:<20} {r[2]:<30} {r[3]:>5.0f} {r[4]:>5.0f} "
              f"{r[5]:>6.1f} {r[6]:>6.0f} {r[7]:>5} {r[8]:>5} {mark}")

    # changed rows only
    print()
    print(f"CHANGES ONLY ({len(changed_rows)} cases where behaviour differs)")
    print("-" * 80)
    print(f"{'Model':<20} {'Scenario':<30} {'VRAM':>5} {'RAM':>5} {'FreeV':>6} {'FreeR':>6} {'Orig':>5} {'Fix':>5}")
    print("-" * 80)
    for r in changed_rows:
        print(f"{r[0]:<20} {r[2]:<30} {r[3]:>5.0f} {r[4]:>5.0f} "
              f"{r[5]:>6.1f} {r[6]:>6.0f} {r[7]:>5} {r[8]:>5}")

    # statistics
    total = len(rows)
    changed = len(changed_rows)
    pct = changed / total * 100 if total else 0
    print()
    print(f"  Total combinations:  {total}")
    print(f"  Behaviour changed:   {changed}  ({pct:.1f} %)")
    print(f"  All other cases:     original and fixed agree  ({(1-pct/100)*total:.0f})")
    print("  <  indicates a change")

    # guard check: no regressions
    regressions = [r for r in changed_rows if r[7] == "GPU" and r[8] == "CPU"]
    if regressions:
        print()
        print("  WARNING - REGRESSIONS  (original=GPU -> fixed=CPU):")
        for r in regressions:
            print(f"     {r[0]:20} {r[2]:30}  VRAM={r[3]}  RAM={r[4]}")
    else:
        print()
        print("  OK - Zero regressions: every change goes CPU -> GPU (never GPU -> CPU)")
    print()
    print(f"  Precondition:  aimdo_enabled=False, load_device != offload_device, model_size > 1 GB")
    print(f"  Safety margin:  model_size * {HEADROOM}  <  free_vram")
    print(f"  RAM available:  {CPU_AVAIL_RATIO*100:.0f} % of total")
    print(f"  Overhead:       {OVERHEAD_GB} GB reserved on GPU")
    print()

    # ── full table ───────────────────────────────────────────────────
    print("─── FULL MATRIX ──────────────────────────────────────────────────────────────")
    print(f"{'Model':<20} {'Scenario':<30} {'VRAM':>5} {'RAM':>5} "
          f"{'FreeV':>6} {'FreeR':>6} {'Orig':>5} {'Fix':>5}  {'Δ'}")
    print("─" * 90)
    for r in rows:
        mark = " ◀" if r[-1] else ""
        print(f"{r[0]:<20} {r[2]:<30} {r[3]:>5.0f} {r[4]:>5.0f} "
              f"{r[5]:>6.1f} {r[6]:>6.0f} {r[7]:>5} {r[8]:>5} {mark}")
    print()

    # ── changed rows only ────────────────────────────────────────────
    print(f"─── CHANGES ONLY ({len(changed_rows)} cases where behaviour differs) ───────")
    print(f"{'Model':<20} {'Scenario':<30} {'VRAM':>5} {'RAM':>5} "
          f"{'FreeV':>6} {'FreeR':>6} {'Orig':>5} {'Fix':>5}")
    print("─" * 80)
    for r in changed_rows:
        print(f"{r[0]:<20} {r[2]:<30} {r[3]:>5.0f} {r[4]:>5.0f} "
              f"{r[5]:>6.1f} {r[6]:>6.0f} {r[7]:>5} {r[8]:>5}")
    print()

    # ── statistics ───────────────────────────────────────────────────
    total = len(rows)
    changed = len(changed_rows)
    pct = changed / total * 100 if total else 0
    print(f"  Total combinations:  {total}")
    print(f"  Behaviour changed:   {changed}  ({pct:.1f} %)")
    print(f"  All other cases:     original and fixed agree  ({(1-pct/100)*total:.0f})")
    print()
    print("  ◀  indicates a change")
    print()

    # ── guard check:  no regressions ─────────────────────────────────
    regressions = [r for r in changed_rows if r[7] == "GPU" and r[8] == "CPU"]
    if regressions:
        print("  ⚠  REGRESSIONS  (original=GPU → fixed=CPU):")
        for r in regressions:
            print(f"     {r[0]:20} {r[2]:30}  VRAM={r[3]}  RAM={r[4]}")
    else:
        print("  ✅  Zero regressions:  every change goes  CPU → GPU  (never GPU → CPU)")
    print()


if __name__ == "__main__":
    simulate()
