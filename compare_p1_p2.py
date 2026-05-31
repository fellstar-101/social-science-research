"""
Compare covariate balance results: p=1 (original) vs p=2 (intended).
Replicates the exact data-cleaning pipeline from the original code,
then runs rdrobust with p=2 on all splits and compares.
"""

import pandas as pd
import numpy as np
from rdrobust import rdrobust
from statsmodels.stats.multitest import multipletests
import warnings
import json

warnings.filterwarnings('ignore')

# ── Configuration (must match original) ──────────────────────────
DATA_PATH = '/mnt/user-data/uploads/ncslballotmeasures_dataset.csv'
P1_RESULTS_PATH = '/mnt/user-data/uploads/appendix_covariate_balance_full.csv'
SMALL_CELL_THRESHOLD = 10

ALL_TOPIC_COLS = [
    'abort', 'agricult', 'animalrights_huntfish', 'arts_cult',
    'banking_finance', 'bond_meas', 'budgets', 'bus_commerce',
    'civil_conlaw', 'criminal', 'drug', 'econdev', 'ed_higher',
    'ed_prek12', 'elections', 'elections_initiative', 'energy_electric',
    'environ', 'ethics_lobby_camfin', 'fedgov', 'gambling_lottery',
    'health', 'humanservices', 'insurance', 'judiciary',
    'juvenile_justice', 'labor_employ', 'landuse_proprights',
    'legislatures', 'localgov', 'military_vet', 'natresourc',
    'redistricting', 'stategov', 'tribalrelat', 'tax_rev',
    'telecom_it', 'termlimits', 'transportation'
]
FISCAL_COLS = ['tax_rev', 'bond_meas', 'budgets']


# ── Data cleaning (exact replica of original pipeline) ───────────
def clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    df['type_clean'] = df['type'].str.strip()
    df['year_num'] = pd.to_numeric(df['year'].str.strip(), errors='coerce')
    df['state_clean'] = df['state'].str.strip()
    df['electiontype_clean'] = df['electiontype'].str.strip()

    df = df[df['pctyesvotes'].notna() & (df['pctyesvotes'] > 0)]
    df = df[df['unknownstatus'] == 0]

    sup_mask = (df['passed'] == 0) & (df['pctyesvotes'] > 50)
    inv_mask = (df['passed'] == 1) & (df['pctyesvotes'] < 50)
    df = df[~sup_mask & ~inv_mask].copy()

    df['margin'] = df['pctyesvotes'] - 50.0
    df['general_election'] = (df['electiontype_clean'] == 'General').astype(int)
    df['state_id'] = pd.Categorical(df['state_clean']).codes
    df['is_fiscal'] = df[FISCAL_COLS].max(axis=1)

    return df


# ── Single covariate balance test ────────────────────────────────
def run_balance_test(data, covariate, p_order):
    """Run rdrobust with specified polynomial order."""
    y = data[covariate].values.astype(float)
    x = data['margin'].values.astype(float)
    cl = data['state_id'].values

    try:
        result = rdrobust(y, x, c=0, kernel='tri', cluster=cl,
                          bwselect='mserd', p=p_order)
        bw = result.bws.iloc[0, 0]

        within_bw = data[data['margin'].abs() <= bw]
        n_pos_bw = int(within_bw[covariate].sum())
        n_pos_left = int(within_bw[within_bw['margin'] < 0][covariate].sum())
        n_pos_right = int(within_bw[within_bw['margin'] >= 0][covariate].sum())

        return {
            'covariate': covariate,
            'coef': result.coef.iloc[0, 0],
            'se_robust': result.se.iloc[2, 0],
            'pv_robust': result.pv.iloc[2, 0],
            'bw': bw,
            'n_eff_left': int(result.N_h[0]),
            'n_eff_right': int(result.N_h[1]),
            'n_pos_bw': n_pos_bw,
            'n_pos_left': n_pos_left,
            'n_pos_right': n_pos_right,
            'success': True,
            'reliable': (n_pos_bw >= SMALL_CELL_THRESHOLD
                         and n_pos_left >= 5
                         and n_pos_right >= 5),
        }
    except Exception as e:
        return {'covariate': covariate, 'success': False, 'error': str(e),
                'reliable': False}


def run_year_test(data, p_order):
    """Run rdrobust on year (continuous, no reliability screen)."""
    try:
        result = rdrobust(
            data['year_num'].values.astype(float),
            data['margin'].values.astype(float),
            c=0, kernel='tri',
            cluster=data['state_id'].values,
            bwselect='mserd', p=p_order)
        return {
            'covariate': 'year_num',
            'coef': result.coef.iloc[0, 0],
            'se_robust': result.se.iloc[2, 0],
            'pv_robust': result.pv.iloc[2, 0],
            'bw': result.bws.iloc[0, 0],
            'success': True,
            'reliable': True,
        }
    except Exception as e:
        return {'covariate': 'year_num', 'success': False,
                'error': str(e), 'reliable': False}


def run_full_battery(data, covariates, p_order):
    """Run balance tests on all covariates + year + general_election."""
    results = []
    all_covs = covariates + ['general_election']
    for cov in all_covs:
        r = run_balance_test(data, cov, p_order)
        results.append(r)
    # year
    r_year = run_year_test(data, p_order)
    results.append(r_year)
    return results


def compute_holm(results):
    """Compute Holm-corrected p-values for reliable results."""
    reliable = [r for r in results if r.get('reliable') and r.get('success')]
    if not reliable:
        return {}
    pvals = [r['pv_robust'] for r in reliable]
    _, pv_holm, _, _ = multipletests(pvals, method='holm')
    return {r['covariate']: ph for r, ph in zip(reliable, pv_holm)}


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("Loading and cleaning data...")
    df = clean_data(DATA_PATH)

    init = df[df['type_clean'] == 'Initiative'].copy()
    legref = df[df['type_clean'] == 'Legislative Referendum'].copy()
    legref_fiscal = legref[legref['is_fiscal'] == 1].copy()
    legref_nonfiscal = legref[legref['is_fiscal'] == 0].copy()
    init_fiscal = init[init['is_fiscal'] == 1].copy()
    init_nonfiscal = init[init['is_fiscal'] == 0].copy()

    splits = {
        'Initiative': init,
        'Legislative Referendum': legref,
        'Fiscal Leg.Ref': legref_fiscal,
        'Non-fiscal Leg.Ref': legref_nonfiscal,
        'Fiscal Init. (placebo)': init_fiscal,
        'Non-fiscal Init. (placebo)': init_nonfiscal,
    }

    print(f"Sample sizes: " + ", ".join(
        f"{k}: {len(v)}" for k, v in splits.items()))

    # ── Run p=2 on all splits ────────────────────────────────────
    p2_all = {}
    for split_name, data in splits.items():
        print(f"\nRunning p=2 on {split_name} (N={len(data)})...")
        results = run_full_battery(data, ALL_TOPIC_COLS, p_order=2)
        holm_map = compute_holm(results)
        for r in results:
            r['p_holm'] = holm_map.get(r['covariate'], None)
        p2_all[split_name] = results
        # Quick summary
        reliable = [r for r in results if r.get('reliable') and r.get('success')]
        n_sig = sum(1 for r in reliable if r['pv_robust'] < 0.05)
        print(f"  Reliable: {len(reliable)}, Sig at 5% (raw): {n_sig}")

    # ── Load p=1 results ─────────────────────────────────────────
    print("\n\nLoading p=1 results from CSV...")
    p1_df = pd.read_csv(P1_RESULTS_PATH)

    # ── Build comparison ─────────────────────────────────────────
    # Map split names between the two result sets
    split_name_map = {
        'Initiative': 'Initiative',
        'Legislative Referendum': 'Legislative Referendum',
        'Fiscal Leg.Ref': 'Fiscal Leg.Ref',
        'Non-fiscal Leg.Ref': 'Non-fiscal Leg.Ref',
        'Fiscal Init. (placebo)': 'Fiscal Init. (placebo)',
        'Non-fiscal Init. (placebo)': 'Non-fiscal Init. (placebo)',
    }

    comparison_rows = []
    for split_name, p2_results in p2_all.items():
        p1_split = p1_df[p1_df['split'] == split_name_map[split_name]]

        for r2 in p2_results:
            cov = r2['covariate']
            p1_row = p1_split[p1_split['covariate'] == cov]

            row = {
                'split': split_name,
                'covariate': cov,
                'p2_success': r2.get('success', False),
                'p2_reliable': r2.get('reliable', False),
            }

            if r2.get('success'):
                row['p2_coef'] = r2['coef']
                row['p2_se'] = r2['se_robust']
                row['p2_pval'] = r2['pv_robust']
                row['p2_bw'] = r2['bw']
                row['p2_holm'] = r2.get('p_holm')
            else:
                row['p2_coef'] = None
                row['p2_se'] = None
                row['p2_pval'] = None
                row['p2_bw'] = None
                row['p2_holm'] = None

            if len(p1_row) > 0:
                p1r = p1_row.iloc[0]
                row['p1_coef'] = p1r.get('beta1')
                row['p1_pval'] = p1r.get('pv_robust')
                row['p1_bw'] = p1r.get('bandwidth')
                row['p1_reliable'] = p1r.get('reliable')
                row['p1_holm'] = p1r.get('p_holm')
            else:
                row['p1_coef'] = None
                row['p1_pval'] = None
                row['p1_bw'] = None
                row['p1_reliable'] = None
                row['p1_holm'] = None

            comparison_rows.append(row)

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv('/home/claude/p1_vs_p2_comparison.csv', index=False)

    # ── Print key comparison tables ──────────────────────────────

    # 1. Focus on covariates that were significant (raw p < 0.05)
    #    in EITHER p=1 or p=2
    print("\n" + "=" * 100)
    print("COVARIATES SIGNIFICANT AT 5% (RAW) IN EITHER p=1 OR p=2")
    print("=" * 100)

    for split_name in splits:
        sub = comp_df[comp_df['split'] == split_name].copy()
        # Convert to numeric for comparison
        sub['p1_pval_n'] = pd.to_numeric(sub['p1_pval'], errors='coerce')
        sub['p2_pval_n'] = pd.to_numeric(sub['p2_pval'], errors='coerce')
        sub['p1_coef_n'] = pd.to_numeric(sub['p1_coef'], errors='coerce')
        sub['p1_bw_n'] = pd.to_numeric(sub['p1_bw'], errors='coerce')
        sub['p1_holm_n'] = pd.to_numeric(sub['p1_holm'], errors='coerce')
        sub['p2_holm_n'] = pd.to_numeric(sub['p2_holm'], errors='coerce')

        # Filter: significant in either, and at least one is reliable+success
        sig = sub[
            ((sub['p1_pval_n'] < 0.05) & (sub['p1_reliable'] == True)) |
            ((sub['p2_pval_n'] < 0.05) & (sub['p2_reliable'] == True))
        ].copy()

        print(f"\n{'─'*90}")
        print(f"  {split_name}")
        print(f"{'─'*90}")
        if len(sig) == 0:
            print("  No significant covariates in either specification.")
            continue

        print(f"  {'Covariate':<22} │ {'β₁(p=1)':>8} {'p(p=1)':>8} {'BW':>5} {'Holm':>7}"
              f" │ {'β₁(p=2)':>8} {'p(p=2)':>8} {'BW':>5} {'Holm':>7} │ Δ")
        print(f"  {'─'*22}─┼─{'─'*33}─┼─{'─'*33}─┼──────")

        for _, r in sig.iterrows():
            p1_str = f"{r['p1_coef_n']:>8.4f} {r['p1_pval_n']:>8.4f} {r['p1_bw_n']:>5.1f}" if pd.notna(r['p1_pval_n']) else f"{'---':>8} {'---':>8} {'---':>5}"
            p2_str = f"{r['p2_coef']:>8.4f} {r['p2_pval']:>8.4f} {r['p2_bw']:>5.1f}" if pd.notna(r['p2_pval']) else f"{'---':>8} {'---':>8} {'---':>5}"

            p1_h = f"{r['p1_holm_n']:>7.4f}" if pd.notna(r['p1_holm_n']) else f"{'---':>7}"
            p2_h = f"{r['p2_holm_n']:>7.4f}" if pd.notna(r['p2_holm_n']) else f"{'---':>7}"

            # Direction change?
            if pd.notna(r['p1_pval_n']) and pd.notna(r['p2_pval']):
                if (r['p1_pval_n'] < 0.05) != (r['p2_pval'] < 0.05):
                    delta = "FLIP"
                elif abs(r['p1_coef_n'] - r['p2_coef']) > 0.05:
                    delta = "SHIFT"
                else:
                    delta = "~same"
            else:
                delta = "n/a"

            print(f"  {r['covariate']:<22} │ {p1_str} {p1_h} │ {p2_str} {p2_h} │ {delta}")

    # 2. Summary table: count of significant covariates per split
    print("\n\n" + "=" * 100)
    print("SUMMARY: SIGNIFICANT COVARIATE COUNTS PER SPLIT")
    print("=" * 100)
    print(f"  {'Split':<30} │ {'p=1 sig/rel':>12} │ {'p=2 sig/rel':>12} │ {'p=1 Holm':>8} │ {'p=2 Holm':>8}")
    print(f"  {'─'*30}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*8}─┼─{'─'*8}")

    for split_name in splits:
        sub = comp_df[comp_df['split'] == split_name].copy()
        sub['p1_pval_n'] = pd.to_numeric(sub['p1_pval'], errors='coerce')
        sub['p1_holm_n'] = pd.to_numeric(sub['p1_holm'], errors='coerce')
        sub['p2_holm_n'] = pd.to_numeric(sub['p2_holm'], errors='coerce')

        p1_rel = sub[sub['p1_reliable'] == True]
        p1_sig = (p1_rel['p1_pval_n'] < 0.05).sum()
        p1_n = len(p1_rel)
        p1_holm_sig = (p1_rel['p1_holm_n'] < 0.05).sum()

        p2_rel = sub[sub['p2_reliable'] == True]
        p2_sig = (p2_rel['p2_pval'] < 0.05).sum()
        p2_n = len(p2_rel)
        p2_holm_sig = (p2_rel['p2_holm_n'] < 0.05).sum()

        print(f"  {split_name:<30} │ {p1_sig:>5}/{p1_n:<5}   │ {p2_sig:>5}/{p2_n:<5}   │ {p1_holm_sig:>8} │ {p2_holm_sig:>8}")

    # 3. Print full p=2 results for the key splits
    print("\n\n" + "=" * 100)
    print("FULL p=2 RESULTS FOR KEY SPLITS (reliable covariates only)")
    print("=" * 100)

    for split_name in ['Initiative', 'Legislative Referendum',
                       'Fiscal Leg.Ref', 'Non-fiscal Leg.Ref']:
        results = p2_all[split_name]
        reliable = [r for r in results if r.get('reliable') and r.get('success')]
        reliable.sort(key=lambda r: r['pv_robust'])

        print(f"\n{'─'*80}")
        print(f"  {split_name} — p=2 (reliable only)")
        print(f"{'─'*80}")
        print(f"  {'Covariate':<25} {'β₁':>8} {'SE':>8} {'p_rob':>8} {'BW':>6} {'Holm':>8}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*8}")

        for r in reliable:
            sig = '*' if r['pv_robust'] < 0.05 else ''
            holm_v = r.get('p_holm')
            holm_str = f"{holm_v:.4f}" if holm_v is not None else "---"
            holm_sig = '**' if (holm_v is not None and holm_v < 0.05) else ''
            print(f"  {r['covariate']:<25} {r['coef']:>8.4f} {r['se_robust']:>8.4f} "
                  f"{r['pv_robust']:>8.4f}{sig:<2} {r['bw']:>5.1f} {holm_str:>8}{holm_sig}")

    print("\n\nDone. Full comparison saved to /home/claude/p1_vs_p2_comparison.csv")


if __name__ == '__main__':
    main()
