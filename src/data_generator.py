"""
Data Generator for Project 002: Classification Engine
Generates two synthetic datasets:
  1. Ancient Manuscript Authenticity Classification (General/Universal)
  2. Silicon Timing Test Pass/Fail (Post-Silicon Validation)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import expit  # sigmoid

SEED = 42
N = 3000
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def generate_manuscript_dataset(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """
    Generates a synthetic dataset for classifying ancient manuscripts as
    authentic or forged based on physical and chemical analysis features.
    Binary target: 1 = Authentic, 0 = Forged
    """
    rng = np.random.default_rng(seed)

    # Features
    ink_iron_ratio      = rng.uniform(0.05, 0.95, n)   # Iron-to-carbon ratio in ink
    parchment_density   = rng.uniform(0.3, 1.2, n)     # g/cm^3
    carbon_14_ratio     = rng.uniform(0.85, 1.05, n)   # C14/C12 normalized
    scribal_pressure    = rng.uniform(10.0, 90.0, n)   # microns indentation variance
    pigment_layers      = rng.integers(1, 8, n)         # count
    uv_fluorescence     = rng.uniform(0.0, 1.0, n)     # normalized intensity
    linguistic_score    = rng.uniform(0.0, 10.0, n)    # anachronism score (0=none)
    vellum_thickness    = rng.uniform(0.1, 0.6, n)     # mm

    # Log-odds: authentic manuscripts have specific chemical signatures
    log_odds = (
        -2.0
        + 3.5 * ink_iron_ratio
        + 2.0 * parchment_density
        + 8.0 * (carbon_14_ratio - 0.95)   # close to 0.95 = authentic age
        - 0.04 * scribal_pressure           # low variance = trained scribe
        + 0.3 * pigment_layers
        - 2.5 * uv_fluorescence             # high UV = modern materials
        - 0.4 * linguistic_score            # anachronisms = forgery
        + 1.5 * vellum_thickness
        + rng.normal(0, 0.5, n)
    )
    prob = expit(log_odds)
    label = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "ink_iron_ratio":       np.round(ink_iron_ratio, 3),
        "parchment_density":    np.round(parchment_density, 3),
        "carbon_14_ratio":      np.round(carbon_14_ratio, 4),
        "scribal_pressure_var": np.round(scribal_pressure, 1),
        "pigment_layer_count":  pigment_layers,
        "uv_fluorescence_idx":  np.round(uv_fluorescence, 3),
        "linguistic_anachronism_score": np.round(linguistic_score, 2),
        "vellum_thickness_mm":  np.round(vellum_thickness, 3),
        "is_authentic":         label,
    })
    return df


def generate_silicon_timing_dataset(n: int = N, seed: int = SEED + 1) -> pd.DataFrame:
    """
    Generates a synthetic dataset for predicting silicon timing test pass/fail.
    Binary target: 1 = PASS, 0 = FAIL
    Class imbalance: ~85% PASS, ~15% FAIL (realistic for production silicon)
    """
    rng = np.random.default_rng(seed)

    vdd_core        = rng.uniform(0.72, 1.08, n)    # Volts
    junction_temp   = rng.uniform(25.0, 125.0, n)   # Celsius
    process_corner  = rng.choice([0, 1, 2], n)      # 0=slow, 1=typical, 2=fast
    leakage_current = rng.uniform(5.0, 80.0, n)     # mA
    ring_osc_freq   = rng.uniform(800.0, 1400.0, n) # MHz
    ir_drop         = rng.uniform(5.0, 65.0, n)     # mV
    metal_resistance= rng.uniform(0.5, 3.5, n)      # Ohm/sq (normalized)

    corner_effect = np.array([-1.5, 0.0, 1.5])[process_corner]

    # Log-odds: high VDD, low temp, fast corner, low IR drop = more likely to pass
    log_odds = (
        2.5                                # strong prior toward PASS
        + 4.0 * (vdd_core - 0.9)          # higher VDD helps
        - 0.04 * (junction_temp - 75.0)   # higher temp hurts
        + corner_effect
        + 0.02 * (leakage_current - 40.0) # fast leakage = fast process
        + 0.003 * (ring_osc_freq - 1100.0)
        - 0.03 * (ir_drop - 35.0)         # high IR drop hurts
        - 0.5 * metal_resistance
        + rng.normal(0, 0.4, n)
    )
    prob = expit(log_odds)
    label = (rng.uniform(0, 1, n) < prob).astype(int)

    df = pd.DataFrame({
        "vdd_core":          np.round(vdd_core, 3),
        "junction_temp":     np.round(junction_temp, 1),
        "process_corner":    process_corner,
        "leakage_current":   np.round(leakage_current, 2),
        "ring_osc_freq":     np.round(ring_osc_freq, 1),
        "ir_drop_mv":        np.round(ir_drop, 2),
        "metal_resistance":  np.round(metal_resistance, 3),
        "timing_pass":       label,
    })
    return df


if __name__ == "__main__":
    ms_df = generate_manuscript_dataset()
    ms_path = DATA_DIR / "manuscript_authenticity_data.csv"
    ms_df.to_csv(ms_path, index=False)
    vc = ms_df['is_authentic'].value_counts()
    print(f"Manuscript dataset saved: {ms_path} ({len(ms_df)} rows)")
    print(f"  Class balance: Authentic={vc[1]} ({vc[1]/len(ms_df)*100:.1f}%), Forged={vc[0]} ({vc[0]/len(ms_df)*100:.1f}%)")

    si_df = generate_silicon_timing_dataset()
    si_path = DATA_DIR / "silicon_timing_test_data.csv"
    si_df.to_csv(si_path, index=False)
    vc2 = si_df['timing_pass'].value_counts()
    print(f"\nSilicon timing dataset saved: {si_path} ({len(si_df)} rows)")
    print(f"  Class balance: PASS={vc2[1]} ({vc2[1]/len(si_df)*100:.1f}%), FAIL={vc2[0]} ({vc2[0]/len(si_df)*100:.1f}%)")
