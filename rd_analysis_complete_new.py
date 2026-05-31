"""
RD Validation for Statewide Ballot Measures — Complete Pipeline
================================================================
Tests whether Conlin & Thompson (2023) agenda-setting concerns generalise
from local tax referenda to statewide ballot measures, comparing
citizen-initiated vs legislature-referred measures.

This is a single-file replication script that starts from the raw NCSL
dataset and produces every result discussed in the analysis.

Usage
-----
    python rd_analysis_complete.py

Dependencies
------------
    pip install pandas numpy rdrobust scipy statsmodels scikit-learn matplotlib

Input
-----
    ncslballotmeasures_dataset.csv   (update DATA_PATH below if needed)

Outputs
-------
    Console  — all test results, tables, and diagnostics
    CSVs     — margin files for R density tests (rddensity)
    PNGs     — histograms, balance plots, placebo plots, LOO plots
    .R       — helper R script for density tests

After running this script, run the generated R script with the
rddensity package for the density continuity tests.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdrobust import rdrobust
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'ncslballotmeasures_dataset.csv'       # path to raw dataset
OUTPUT_DIR = './output'                              # output directory
GENERATE_PLOTS = True                                # set False to skip plots
N_PERMUTATIONS = 5000                                # for omnibus F-test
SMALL_CELL_THRESHOLD = 10                            # min positive cases in BW
np.random.seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topic columns (all binary indicators in the dataset)
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

# Fiscal definition (narrow, following Conlin & Thompson 2023)
FISCAL_COLS = ['tax_rev', 'bond_meas', 'budgets']


# ============================================================
# SECTION 1: DATA CLEANING
# ============================================================
def clean_data(filepath):
    """
    Clean the raw NCSL ballot measures dataset.

    Pipeline (matching rd_analysis.py Part 1):
      1. Strip whitespace from type, year, state, electiontype
      2. Remove rows with missing or zero pctyesvotes
      3. Remove rows with unknownstatus != 0
      4. Remove supermajority cases using the ORIGINAL 'passed' column
         (measures where pctyesvotes > 50 but passed == 0, or vice-versa)
      5. Define running variable: margin = pctyesvotes - 50
      6. Create auxiliary variables

    Returns the cleaned DataFrame.
    """
    print("=" * 70)
    print("SECTION 1: DATA CLEANING")
    print("=" * 70)

    df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
    print(f"Raw dataset: {len(df)} measures")

    # --- string cleaning ---
    df['type_clean'] = df['type'].str.strip()
    df['year_num'] = pd.to_numeric(df['year'].str.strip(), errors='coerce')
    df['state_clean'] = df['state'].str.strip()
    df['electiontype_clean'] = df['electiontype'].str.strip()

    # --- filter 1: missing / zero vote data ---
    df = df[df['pctyesvotes'].notna() & (df['pctyesvotes'] > 0)]
    print(f"After removing missing/zero votes: {len(df)}")

    # --- filter 2: unknown-status measures ---
    df = df[df['unknownstatus'] == 0]
    print(f"After removing unknown status: {len(df)}")

    # --- filter 3: supermajority / inverse-threshold cases ---
    #     IMPORTANT: use the ORIGINAL 'passed' column, NOT a margin-based
    #     redefinition.  Redefining passed first would make this filter a
    #     no-op (see methodology document for details).
    sup_mask = (df['passed'] == 0) & (df['pctyesvotes'] > 50)
    inv_mask = (df['passed'] == 1) & (df['pctyesvotes'] < 50)
    print(f"Removing {sup_mask.sum()} supermajority cases (passed=0, pct>50)")
    print(f"Removing {inv_mask.sum()} inverse cases (passed=1, pct<50)")
    df = df[~sup_mask & ~inv_mask].copy()

    # --- running variable ---
    df['margin'] = df['pctyesvotes'] - 50.0

    # --- auxiliary variables ---
    df['general_election'] = (df['electiontype_clean'] == 'General').astype(int)
    df['state_id'] = pd.Categorical(df['state_clean']).codes
    df['is_fiscal'] = df[FISCAL_COLS].max(axis=1)

    print(f"\nFinal working dataset: {len(df)} measures")
    for t in ['Initiative', 'Legislative Referendum', 'Popular Referendum', 'Other']:
        n = (df['type_clean'] == t).sum()
        if n > 0:
            print(f"  {t}: {n}")

    return df


# ============================================================
# SECTION 2: SUMMARY STATISTICS
# ============================================================
def summary_statistics(df):
    print("\n" + "=" * 70)
    print("SECTION 2: SUMMARY STATISTICS")
    print("=" * 70)

    for type_val in ['Initiative', 'Legislative Referendum', 'Popular Referendum']:
        sub = df[df['type_clean'] == type_val]
        n_total = len(sub)
        n_pass = (sub['margin'] >= 0).sum()
        print(f"\n{type_val}: N={n_total}, Passed={n_pass} "
              f"({100*n_pass/n_total:.1f}%), Failed={n_total-n_pass}")
        print(f"  Mean margin: {sub['margin'].mean():.2f}, "
              f"SD: {sub['margin'].std():.2f}")
        for w in [1, 2, 5]:
            close = sub[sub['margin'].abs() <= w]
            below = (close['margin'] < 0).sum()
            above = (close['margin'] >= 0).sum()
            print(f"  Within {w}pp: {len(close)} total "
                  f"({below} below, {above} above)")

    # Fiscal breakdown for legislative referrals
    legref = df[df['type_clean'] == 'Legislative Referendum']
    n_fiscal = legref['is_fiscal'].sum()
    n_nonfiscal = len(legref) - n_fiscal
    print(f"\nLegislative Referral fiscal breakdown:")
    print(f"  Fiscal (tax_rev|bond_meas|budgets): {int(n_fiscal)}")
    print(f"  Non-fiscal: {int(n_nonfiscal)}")


# ============================================================
# SECTION 3: EXPORT MARGINS FOR R DENSITY TESTS
# ============================================================
def export_margins(df):
    print("\n" + "=" * 70)
    print("SECTION 3: EXPORT MARGIN CSVs FOR R DENSITY TESTS")
    print("=" * 70)

    # By type
    for type_val in ['Initiative', 'Legislative Referendum', 'Popular Referendum']:
        sub = df[df['type_clean'] == type_val]
        fname = f'margins_{type_val.lower().replace(" ", "_")}.csv'
        sub[['margin']].to_csv(f'{OUTPUT_DIR}/{fname}', index=False)
        print(f"Saved: {fname} (N={len(sub)})")

    df[['margin']].to_csv(f'{OUTPUT_DIR}/margins_all.csv', index=False)
    print(f"Saved: margins_all.csv (N={len(df)})")

    # Fiscal / non-fiscal splits for legislative referrals
    legref = df[df['type_clean'] == 'Legislative Referendum']
    for label, mask in [('fiscal', legref['is_fiscal'] == 1),
                        ('nonfiscal', legref['is_fiscal'] == 0)]:
        sub = legref[mask]
        fname = f'margins_legref_{label}.csv'
        sub[['margin']].to_csv(f'{OUTPUT_DIR}/{fname}', index=False)
        print(f"Saved: {fname} (N={len(sub)})")

    # Fiscal / non-fiscal splits for initiatives (as comparison)
    init = df[df['type_clean'] == 'Initiative']
    for label, mask in [('fiscal', init['is_fiscal'] == 1),
                        ('nonfiscal', init['is_fiscal'] == 0)]:
        sub = init[mask]
        fname = f'margins_init_{label}.csv'
        sub[['margin']].to_csv(f'{OUTPUT_DIR}/{fname}', index=False)
        print(f"Saved: {fname} (N={len(sub)})")


# ============================================================
# SECTION 4: COVARIATE BALANCE TESTS
# ============================================================
def run_balance_test(data, covariate):
    """
    Run a single covariate balance test using rdrobust.

    Estimates the discontinuity in Pr(covariate = 1) at margin = 0,
    using MSE-optimal bandwidth, triangular kernel, and state-clustered SEs.
    """
    y = data[covariate].values.astype(float)
    x = data['margin'].values.astype(float)
    cl = data['state_id'].values

    try:
        result = rdrobust(y, x, c=0, kernel='tri', cluster=cl,
                          bwselect='mserd')
        bw = result.bws.iloc[0, 0]

        # Post-hoc reliability check: positive cases within bandwidth
        within_bw = data[data['margin'].abs() <= bw]
        n_pos_bw = int(within_bw[covariate].sum())
        n_pos_left = int(within_bw[within_bw['margin'] < 0][covariate].sum())
        n_pos_right = int(within_bw[within_bw['margin'] >= 0][covariate].sum())

        return {
            'covariate': covariate,
            'coef': result.coef.iloc[0, 0],
            'se_robust': result.se.iloc[2, 0],
            'pv_robust': result.pv.iloc[2, 0],
            'ci_lower': result.ci.iloc[2, 0],
            'ci_upper': result.ci.iloc[2, 1],
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


def run_balance_battery(df, subsample_label, subsample_data,
                        covariates, include_year=True):
    """
    Run the full covariate balance battery on a subsample.

    Returns list of result dicts for reliable covariates.
    """
    print(f"\n--- {subsample_label} (N={len(subsample_data)}) ---")
    print(f"{'Covariate':<25} {'Coef':>8} {'SE_r':>8} {'p_rob':>8} "
          f"{'BW':>6} {'pos':>5} {'(L/R)':>9} {'Sig':>4} {'Flag':>12}")
    print("-" * 95)

    results = []
    for cov in covariates:
        r = run_balance_test(subsample_data, cov)
        results.append(r)

        if r['success']:
            sig = ('***' if r['pv_robust'] < 0.001 else
                   '**' if r['pv_robust'] < 0.01 else
                   '*' if r['pv_robust'] < 0.05 else '')
            flag = '' if r['reliable'] else '[unreliable]'
            print(f"{cov:<25} {r['coef']:>8.4f} {r['se_robust']:>8.4f} "
                  f"{r['pv_robust']:>8.4f} {r['bw']:>6.1f} "
                  f"{r['n_pos_bw']:>5} ({r['n_pos_left']:>3}/{r['n_pos_right']:<3}) "
                  f"{sig:>4} {flag:>12}")
        else:
            print(f"{cov:<25} FAILED: {r.get('error', '')[:50]}")

    # Year (continuous covariate — no small-cell issue)
    if include_year and 'year_num' in subsample_data.columns:
        try:
            result = rdrobust(
                subsample_data['year_num'].values.astype(float),
                subsample_data['margin'].values.astype(float),
                c=0, kernel='tri',
                cluster=subsample_data['state_id'].values,
                bwselect='mserd')
            coef = result.coef.iloc[0, 0]
            se_r = result.se.iloc[2, 0]
            pv_r = result.pv.iloc[2, 0]
            bw = result.bws.iloc[0, 0]
            sig = ('***' if pv_r < 0.001 else '**' if pv_r < 0.01
                   else '*' if pv_r < 0.05 else '')
            print(f"{'year_num':<25} {coef:>8.2f} {se_r:>8.3f} "
                  f"{pv_r:>8.4f} {bw:>6.1f} {'—':>5} {'—':>9} {sig:>4}")
            results.append({
                'covariate': 'year_num', 'coef': coef,
                'se_robust': se_r, 'pv_robust': pv_r, 'bw': bw,
                'success': True, 'reliable': True,
            })
        except Exception as e:
            print(f"{'year_num':<25} FAILED: {e}")

    # Summary counts
    reliable = [r for r in results if r.get('reliable')]
    n_sig = sum(1 for r in reliable if r['pv_robust'] < 0.05)
    n_rel = len(reliable)
    print(f"\n  Reliable covariates: {n_rel}")
    print(f"  Significant at 5% (raw): {n_sig}/{n_rel} "
          f"(expected by chance: {n_rel * 0.05:.1f})")

    return results


def multiple_testing_correction(results, label):
    """Apply Holm-Bonferroni and BH corrections to reliable results."""
    reliable = [r for r in results
                if r.get('reliable') and r.get('success')]
    if not reliable:
        print(f"  {label}: no reliable results to correct")
        return

    pvals = [r['pv_robust'] for r in reliable]
    names = [r['covariate'] for r in reliable]

    rej_holm, pv_holm, _, _ = multipletests(pvals, method='holm')
    rej_bh, pv_bh, _, _ = multipletests(pvals, method='fdr_bh')

    print(f"\n  {label} — Multiple testing corrections "
          f"({len(reliable)} reliable covariates):")
    print(f"  {'Covariate':<25} {'p_raw':<8} {'p_Holm':<8} {'p_BH':<8}")
    print("  " + "-" * 49)
    for i in range(len(reliable)):
        sig_h = '**' if pv_holm[i] < 0.05 else ''
        print(f"  {names[i]:<25} {pvals[i]:<8.4f} "
              f"{pv_holm[i]:<8.4f} {pv_bh[i]:<8.4f} {sig_h}")

    n_holm = sum(1 for p in pv_holm if p < 0.05)
    n_bh = sum(1 for p in pv_bh if p < 0.05)
    print(f"  Significant at 5%: {n_holm} (Holm), {n_bh} (BH)")


def covariate_balance_section(df):
    """Run all covariate balance tests (Section 4)."""
    print("\n" + "=" * 70)
    print("SECTION 4: COVARIATE BALANCE TESTS")
    print("=" * 70)

    all_covs = ALL_TOPIC_COLS + ['general_election']
    balance_results = {}

    # 4a: By type (Initiative, Leg.Ref, Popular Ref)
    for type_val in ['Initiative', 'Legislative Referendum',
                     'Popular Referendum']:
        sub = df[df['type_clean'] == type_val].copy()
        results = run_balance_battery(df, type_val, sub, all_covs)
        balance_results[type_val] = results

    # 4b: Multiple testing corrections
    print("\n" + "-" * 70)
    print("Multiple Testing Corrections")
    print("-" * 70)
    for type_val in balance_results:
        multiple_testing_correction(balance_results[type_val], type_val)

    return balance_results


# ============================================================
# SECTION 5: FISCAL / NON-FISCAL SAMPLE SPLIT (H2 test)
# ============================================================
def fiscal_split_section(df):
    """
    Split legislative referrals into fiscal / non-fiscal subsamples
    and run covariate balance tests on each.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: FISCAL / NON-FISCAL SAMPLE SPLIT (H2)")
    print("=" * 70)

    legref = df[df['type_clean'] == 'Legislative Referendum'].copy()
    legref_fiscal = legref[legref['is_fiscal'] == 1].copy()
    legref_nonfiscal = legref[legref['is_fiscal'] == 0].copy()

    print(f"Legislative referrals: {len(legref)}")
    print(f"  Fiscal: {len(legref_fiscal)}")
    print(f"  Non-fiscal: {len(legref_nonfiscal)}")

    all_covs = ALL_TOPIC_COLS + ['general_election']
    split_results = {}

    for label, sub in [('Fiscal Leg.Ref', legref_fiscal),
                        ('Non-fiscal Leg.Ref', legref_nonfiscal)]:
        results = run_balance_battery(df, label, sub, all_covs)
        split_results[label] = results
        multiple_testing_correction(results, label)

    # Simpson's paradox check for natresourc and localgov
    print("\n" + "-" * 70)
    print("Simpson's Paradox Check: natresourc, localgov")
    print("-" * 70)
    for cov in ['natresourc', 'localgov']:
        print(f"\n  {cov} prevalence above vs below threshold:")
        for label, sub in [('All Leg.Ref', legref),
                            ('Fiscal', legref_fiscal),
                            ('Non-fiscal', legref_nonfiscal)]:
            for h in [5, 10]:
                close = sub[sub['margin'].abs() <= h]
                above = close[close['margin'] >= 0][cov].mean()
                below = close[close['margin'] < 0][cov].mean()
                print(f"    {label:<15} h={h}pp: above={above:.4f}, "
                      f"below={below:.4f}, diff={above - below:+.4f}")

    # Initiative fiscal/non-fiscal split (PLACEBO CONTROL)
    # If agenda-setting drives the non-fiscal imbalances in leg.ref,
    # then the same split on initiatives (no agenda-setter) should
    # show NO such pattern.
    print("\n" + "-" * 70)
    print("PLACEBO: Initiative Fiscal / Non-Fiscal Split")
    print("-" * 70)

    init = df[df['type_clean'] == 'Initiative'].copy()
    init_fiscal = init[init['is_fiscal'] == 1].copy()
    init_nonfiscal = init[init['is_fiscal'] == 0].copy()

    print(f"Initiatives: {len(init)}")
    print(f"  Fiscal: {len(init_fiscal)}")
    print(f"  Non-fiscal: {len(init_nonfiscal)}")

    for label, sub in [('Fiscal Init. (placebo)', init_fiscal),
                        ('Non-fiscal Init. (placebo)', init_nonfiscal)]:
        results = run_balance_battery(df, label, sub, all_covs)
        split_results[label] = results

    return split_results


# ============================================================
# SECTION 6: INTERACTION TEST (Initiative vs Leg.Ref)
# ============================================================
def interaction_test_section(df, sig_covs=None):
    """
    Test whether the covariate discontinuity differs significantly
    between initiatives and legislative referrals.

    Runs rdrobust separately on each type (each with its own
    MSE-optimal bandwidth and clustered SEs), then computes the
    difference.  Since the subsamples are independent:
        delta = beta_legref - beta_init
        SE(delta) = sqrt(SE_legref^2 + SE_init^2)
        z = delta / SE(delta)
    """
    print("\n" + "=" * 70)
    print("SECTION 6: INTERACTION TEST (Initiative vs Legislative Referendum)")
    print("=" * 70)

    init = df[df['type_clean'] == 'Initiative'].copy()
    legref = df[df['type_clean'] == 'Legislative Referendum'].copy()

    print(f"Initiatives: N={len(init)}")
    print(f"Legislative Referrals: N={len(legref)}")

    # Test covariates that are reliable in BOTH types
    all_covs = ALL_TOPIC_COLS + ['general_election']
    reliable_both = []
    for cov in all_covs:
        ok = True
        for sub in [init, legref]:
            within_bw = sub[sub['margin'].abs() <= 5]
            if within_bw[cov].sum() < SMALL_CELL_THRESHOLD:
                ok = False
        if ok:
            reliable_both.append(cov)

    print(f"Covariates reliable in both types: {len(reliable_both)}")

    print(f"\n{'Covariate':<22} {'β_init':>8} {'SE_i':>7} {'p_i':>7} | "
          f"{'β_lr':>8} {'SE_lr':>7} {'p_lr':>7} | "
          f"{'δ':>8} {'SE_δ':>7} {'z':>7} {'p_diff':>7} {'sig':>4}")
    print("-" * 110)

    results = []
    for cov in reliable_both:
        try:
            r_i = rdrobust(init[cov].values.astype(float),
                           init['margin'].values.astype(float), c=0,
                           kernel='tri',
                           cluster=init['state_id'].values,
                           bwselect='mserd')
            coef_i = r_i.coef.iloc[0, 0]
            se_i = r_i.se.iloc[2, 0]
            pv_i = r_i.pv.iloc[2, 0]
            bw_i = r_i.bws.iloc[0, 0]

            r_lr = rdrobust(legref[cov].values.astype(float),
                            legref['margin'].values.astype(float), c=0,
                            kernel='tri',
                            cluster=legref['state_id'].values,
                            bwselect='mserd')
            coef_lr = r_lr.coef.iloc[0, 0]
            se_lr = r_lr.se.iloc[2, 0]
            pv_lr = r_lr.pv.iloc[2, 0]
            bw_lr = r_lr.bws.iloc[0, 0]

            # Difference (independent samples)
            delta = coef_lr - coef_i
            se_delta = np.sqrt(se_lr**2 + se_i**2)
            z = delta / se_delta
            p_diff = 2 * (1 - stats.norm.cdf(abs(z)))

            sig = ('***' if p_diff < 0.001 else '**' if p_diff < 0.01
                   else '*' if p_diff < 0.05 else '')

            print(f"{cov:<22} {coef_i:>8.4f} {se_i:>7.4f} {pv_i:>7.4f} | "
                  f"{coef_lr:>8.4f} {se_lr:>7.4f} {pv_lr:>7.4f} | "
                  f"{delta:>8.4f} {se_delta:>7.4f} {z:>7.3f} "
                  f"{p_diff:>7.4f} {sig:>4}")

            results.append({
                'covariate': cov, 'coef_i': coef_i, 'se_i': se_i,
                'pv_i': pv_i, 'coef_lr': coef_lr, 'se_lr': se_lr,
                'pv_lr': pv_lr, 'delta': delta, 'se_delta': se_delta,
                'z': z, 'p_diff': p_diff,
                'bw_i': bw_i, 'bw_lr': bw_lr
            })
        except Exception as e:
            print(f"{cov:<22} FAILED: {str(e)[:50]}")

    # Summary
    print(f"\n--- Significant differences at 5% ---")
    sig_results = [r for r in results if r['p_diff'] < 0.05]
    if sig_results:
        for r in sig_results:
            print(f"  {r['covariate']}: δ={r['delta']:.4f}, "
                  f"z={r['z']:.3f}, p={r['p_diff']:.4f} "
                  f"(BW_init={r['bw_i']:.1f}, BW_lr={r['bw_lr']:.1f})")
    else:
        print("  (none)")

    return results


# ============================================================
# SECTION 7: PLACEBO CUTOFF TESTS
# ============================================================
def placebo_section(df, sig_covs):
    """
    Run balance tests at placebo thresholds for covariates that were
    significant at the true threshold (50%).
    """
    print("\n" + "=" * 70)
    print("SECTION 7: PLACEBO CUTOFF TESTS")
    print("=" * 70)

    if not sig_covs:
        print("No significant covariates — skipping placebo tests.")
        return

    legref = df[df['type_clean'] == 'Legislative Referendum'].copy()
    cutoffs = [40, 42, 44, 45, 46, 48, 50, 52, 54, 55, 56, 58, 60]
    print(f"Covariates: {sig_covs}")
    print(f"Cutoffs: {cutoffs}")

    for cov in sig_covs:
        print(f"\n  {cov}:")
        print(f"  {'Cutoff':<8} {'Coef':<9} {'SE':<9} {'p-value':<9} {'Sig':<5}")
        print("  " + "-" * 40)

        placebo = []
        for cutoff in cutoffs:
            margin_p = legref['pctyesvotes'] - cutoff
            try:
                result = rdrobust(
                    legref[cov].values.astype(float),
                    margin_p.values.astype(float),
                    c=0, kernel='tri',
                    cluster=legref['state_id'].values,
                    bwselect='mserd')
                coef = result.coef.iloc[0, 0]
                se = result.se.iloc[2, 0]
                pv = result.pv.iloc[2, 0]
                sig = ('***' if pv < 0.01 else '**' if pv < 0.05
                       else '*' if pv < 0.1 else '')
                marker = ' <-- TRUE' if cutoff == 50 else ''
                print(f"  {cutoff:<8} {coef:<9.3f} {se:<9.3f} "
                      f"{pv:<9.4f} {sig:<5}{marker}")
                placebo.append({'cutoff': cutoff, 'coef': coef,
                                'se': se, 'pv': pv})
            except Exception:
                print(f"  {cutoff:<8} FAILED")
                placebo.append({'cutoff': cutoff, 'coef': np.nan,
                                'se': np.nan, 'pv': np.nan})

        # Placebo plot
        if GENERATE_PLOTS:
            valid = [p for p in placebo if not np.isnan(p['coef'])]
            if valid:
                fig, ax = plt.subplots(figsize=(8, 5))
                xs = [p['cutoff'] for p in valid]
                ys = [p['coef'] for p in valid]
                lo = [p['coef'] - 1.96 * p['se'] for p in valid]
                hi = [p['coef'] + 1.96 * p['se'] for p in valid]
                ax.fill_between(xs, lo, hi, alpha=0.2, color='steelblue')
                ax.plot(xs, ys, 'o-', color='steelblue', markersize=6)
                ax.axhline(y=0, color='gray', linewidth=0.5)
                ax.axvline(x=50, color='red', linestyle='--', linewidth=2,
                           label='True threshold (50%)')
                ax.set_xlabel('Placebo Cutoff (%)')
                ax.set_ylabel(f'Estimated Discontinuity in {cov}')
                ax.set_title(f'Placebo Test: {cov} (Legislative Referenda)')
                ax.legend()
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/fig_placebo_{cov}.png',
                            dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: fig_placebo_{cov}.png")


# ============================================================
# SECTION 8: LEAVE-ONE-STATE-OUT ROBUSTNESS
# ============================================================
def leave_one_out_section(df, sig_covs):
    print("\n" + "=" * 70)
    print("SECTION 8: LEAVE-ONE-STATE-OUT ROBUSTNESS")
    print("=" * 70)

    if not sig_covs:
        print("No significant covariates — skipping LOO.")
        return

    legref = df[df['type_clean'] == 'Legislative Referendum'].copy()
    states = legref['state_clean'].unique()

    for cov in sig_covs:
        print(f"\n  Leave-one-state-out: {cov}")
        loo = []
        for state in states:
            sub = legref[legref['state_clean'] != state]
            try:
                result = rdrobust(
                    sub[cov].values.astype(float),
                    sub['margin'].values.astype(float),
                    c=0, kernel='tri',
                    cluster=sub['state_id'].values,
                    bwselect='mserd')
                loo.append({'state': state,
                            'coef': result.coef.iloc[0, 0],
                            'pv': result.pv.iloc[2, 0]})
            except Exception:
                loo.append({'state': state, 'coef': np.nan, 'pv': np.nan})

        valid = [r for r in loo if not np.isnan(r['coef'])]
        coefs = [r['coef'] for r in valid]
        print(f"    Coef range: [{min(coefs):.4f}, {max(coefs):.4f}]")
        print(f"    Always sig at 5%: "
              f"{all(r['pv'] < 0.05 for r in valid)}")
        for r in valid:
            if r['pv'] >= 0.05:
                print(f"    Drops to insig when excluding: {r['state']} "
                      f"(coef={r['coef']:.4f}, p={r['pv']:.4f})")

        if GENERATE_PLOTS and valid:
            fig, ax = plt.subplots(figsize=(10, 6))
            idx = np.argsort([r['coef'] for r in valid])
            ss = [valid[i]['state'] for i in idx]
            cc = [valid[i]['coef'] for i in idx]
            co = ['red' if valid[i]['pv'] >= 0.05 else 'steelblue'
                  for i in idx]
            ax.barh(range(len(ss)), cc, color=co, alpha=0.7)
            ax.set_yticks(range(len(ss)))
            ax.set_yticklabels(ss, fontsize=6)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel(f'Estimated Discontinuity in {cov}')
            ax.set_title(f'Leave-One-State-Out: {cov}\n'
                         f'(Red = drops below 5%)')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/fig_loo_{cov}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: fig_loo_{cov}.png")


# ============================================================
# SECTION 9: FIXED-BANDWIDTH ROBUSTNESS
# ============================================================
def fixed_bandwidth_section(df, sig_covs):
    print("\n" + "=" * 70)
    print("SECTION 9: FIXED-BANDWIDTH ROBUSTNESS")
    print("=" * 70)

    bandwidths = [3, 5, 7, 10]
    types = ['Initiative', 'Legislative Referendum']

    for type_val in types:
        sub_type = df[df['type_clean'] == type_val]
        print(f"\n--- {type_val} ---")
        print(f"{'Covariate':<20} {'h':>4} {'N':>6} {'Coef':>8} "
              f"{'SE':>8} {'p_rob':>8} {'Sig':>4}")
        print("-" * 60)

        for cov in sig_covs:
            for h in bandwidths:
                sub = sub_type[sub_type['margin'].abs() <= h].copy()
                n_obs = len(sub)
                n_pos = sub[cov].sum()
                if n_obs < 20 or n_pos < 5 or (n_obs - n_pos) < 5:
                    print(f"{cov:<20} {h:>4} {n_obs:>6} "
                          f"{'(insufficient)':>30}")
                    continue
                try:
                    result = rdrobust(
                        sub[cov].values.astype(float),
                        sub['margin'].values.astype(float),
                        c=0, h=h, kernel='triangular',
                        cluster=sub['state_id'].values)
                    coef = result.coef.iloc[0, 0]
                    se_r = result.se.iloc[2, 0]
                    pv_r = result.pv.iloc[2, 0]
                    sig = ('***' if pv_r < 0.001 else '**' if pv_r < 0.01
                           else '*' if pv_r < 0.05 else '')
                    print(f"{cov:<20} {h:>4} {n_obs:>6} {coef:>8.4f} "
                          f"{se_r:>8.4f} {pv_r:>8.4f} {sig:>4}")
                except Exception as e:
                    print(f"{cov:<20} {h:>4} {n_obs:>6} "
                          f"{'ERROR: ' + str(e)[:25]:>30}")


# ============================================================
# SECTION 10: OMNIBUS F-TEST WITH RANDOMISATION INFERENCE
# ============================================================
def compute_f_stat(D, X):
    """F-statistic for H0: all slopes = 0 in D ~ X."""
    n, K = X.shape
    if n <= K + 1:
        return np.nan
    reg = LinearRegression().fit(X, D)
    D_hat = reg.predict(X)
    SS_res = np.sum((D - D_hat) ** 2)
    SS_tot = np.sum((D - np.mean(D)) ** 2)
    if SS_tot == 0:
        return np.nan
    R2 = 1 - SS_res / SS_tot
    if R2 >= 1.0:
        return np.inf
    return (R2 / K) / ((1 - R2) / (n - K - 1))


def omnibus_section(df):
    print("\n" + "=" * 70)
    print("SECTION 10: OMNIBUS F-TEST WITH RANDOMISATION INFERENCE")
    print("=" * 70)

    bandwidths = [3, 5, 7, 10, 13, 15]
    types = ['Initiative', 'Legislative Referendum']

    print(f"Permutations: {N_PERMUTATIONS}")
    print("NOTE: Using reliable covariates only (post ±5pp screening + year)")

    omnibus_results = {}
    for type_val in types:
        sub_type = df[df['type_clean'] == type_val].copy()

        # Determine reliable covariates for this type (matching Part 2)
        all_covs_screen = ALL_TOPIC_COLS + ['general_election']
        reliable = []
        for cov in all_covs_screen:
            w5 = sub_type[sub_type['margin'].abs() <= 5]
            if w5[cov].sum() >= SMALL_CELL_THRESHOLD:
                reliable.append(cov)
        reliable_with_year = reliable + ['year_num']

        print(f"\n--- {type_val} (N={len(sub_type)}, "
              f"K={len(reliable_with_year)} reliable covariates) ---")
        print(f"{'BW':>6} {'N':>6} {'K':>4} {'R²':>8} "
              f"{'F_obs':>8} {'p_rand':>8} {'Sig':>4}")
        print("-" * 50)

        for h in bandwidths:
            sub = sub_type[sub_type['margin'].abs() <= h].copy()
            covs_avail = [c for c in reliable_with_year
                          if c in sub.columns]
            X = sub[covs_avail].values.astype(float)
            D = (sub['margin'] >= 0).values.astype(float)

            # Remove zero-variance columns
            mask = X.std(axis=0) > 0
            X = X[:, mask]
            n, K = X.shape

            if n <= K + 10:
                print(f"{h:>6} {n:>6} {K:>4} {'(insufficient)':>30}")
                continue

            F_obs = compute_f_stat(D, X)
            if np.isnan(F_obs):
                print(f"{h:>6} {n:>6} {K:>4} {'(failed)':>30}")
                continue

            # R²
            reg = LinearRegression().fit(X, D)
            SS_res = np.sum((D - reg.predict(X)) ** 2)
            SS_tot = np.sum((D - D.mean()) ** 2)
            R2 = 1 - SS_res / SS_tot

            # Permutation null
            null_F = np.array([
                compute_f_stat(np.random.permutation(D), X)
                for _ in range(N_PERMUTATIONS)
            ])
            p_rand = np.mean(null_F >= F_obs)

            sig = ('***' if p_rand < 0.001 else '**' if p_rand < 0.01
                   else '*' if p_rand < 0.05 else '')
            print(f"{h:>6} {n:>6} {K:>4} {R2:>8.4f} "
                  f"{F_obs:>8.3f} {p_rand:>8.4f} {sig:>4}")

            if type_val not in omnibus_results:
                omnibus_results[type_val] = []
            omnibus_results[type_val].append({
                'h': h, 'N': n, 'K': K, 'R2': R2,
                'F': F_obs, 'p': p_rand
            })

    # Generate omnibus graph
    if GENERATE_PLOTS and omnibus_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        colors = {'Initiative': '#2ca02c',
                  'Legislative Referendum': '#d62728'}
        markers = {'Initiative': 'o',
                   'Legislative Referendum': 's'}
        plot_labels = {'Initiative': 'Initiative',
                       'Legislative Referendum': 'Leg. Referral'}

        for tv in omnibus_results:
            hs = [r['h'] for r in omnibus_results[tv]]
            Fs = [r['F'] for r in omnibus_results[tv]]
            ps = [r['p'] for r in omnibus_results[tv]]

            ax1.plot(hs, Fs, marker=markers[tv], color=colors[tv],
                     label=plot_labels[tv], linewidth=2, markersize=8)
            ax2.plot(hs, ps, marker=markers[tv], color=colors[tv],
                     label=plot_labels[tv], linewidth=2, markersize=8)

        ax1.axhline(y=1, color='gray', linestyle=':', linewidth=1,
                    alpha=0.7)
        ax1.set_xlabel('Bandwidth (pp)', fontsize=12)
        ax1.set_ylabel('F-statistic', fontsize=12)
        ax1.set_title('Omnibus F-statistic by Bandwidth', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.set_xticks(bandwidths)

        ax2.axhline(y=0.05, color='#d62728', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=r'$\alpha = 0.05$')
        ax2.set_xlabel('Bandwidth (pp)', fontsize=12)
        ax2.set_ylabel('Randomization p-value', fontsize=12)
        ax2.set_title('Omnibus p-value by Bandwidth', fontsize=13)
        ax2.legend(fontsize=11)
        ax2.set_xticks(bandwidths)

        # Annotate the h=10 leg.ref result if present
        lr_res = omnibus_results.get('Legislative Referendum', [])
        h10 = [r for r in lr_res if r['h'] == 10]
        if h10:
            ax2.annotate(f"p = {h10[0]['p']:.4f}",
                         xy=(10, h10[0]['p']),
                         xytext=(11.5, 0.08), fontsize=10,
                         fontweight='bold',
                         arrowprops=dict(arrowstyle='->',
                                         color='black', lw=1.5))

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/fig_omnibus.png',
                    dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: fig_omnibus.png")

    return omnibus_results


# ============================================================
# SECTION 11: DENSITY HISTOGRAMS
# ============================================================
def density_histograms(df):
    if not GENERATE_PLOTS:
        return
    print("\n" + "=" * 70)
    print("SECTION 11: DENSITY HISTOGRAMS")
    print("=" * 70)

    # Full-range
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, t in zip(axes,
                     ['Initiative', 'Legislative Referendum',
                      'Popular Referendum']):
        sub = df[df['type_clean'] == t]
        bins = np.arange(-30.5, 31, 1)
        ax.hist(sub['margin'].values, bins=bins, edgecolor='black',
                alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Vote Margin (pp from 50%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{t}\n(N={len(sub)})')
        ax.set_xlim(-31, 31)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_density_full.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_density_full.png")

    # Zoomed ±10pp
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, t in zip(axes,
                     ['Initiative', 'Legislative Referendum',
                      'Popular Referendum']):
        sub = df[df['type_clean'] == t]
        bins = np.arange(-10.5, 11, 1)
        ax.hist(sub['margin'].values, bins=bins, edgecolor='black',
                alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Vote Margin (pp from 50%)')
        ax.set_ylabel('Count')
        ax.set_title(f'{t} (Zoomed)')
        ax.set_xlim(-11, 11)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fig_density_zoomed.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig_density_zoomed.png")


# ============================================================
# SECTION 12: GENERATE R SCRIPT FOR rddensity
# ============================================================
def generate_r_script():
    print("\n" + "=" * 70)
    print("SECTION 12: R SCRIPT FOR DENSITY TESTS")
    print("=" * 70)

    r_script = r"""
# rddensity tests for statewide ballot measures
# Run with: Rscript run_rddensity.R
# Requires: install.packages("rddensity")

library(rddensity)

run_test <- function(filepath, label) {
  cat("\n===", label, "===\n")
  data <- read.csv(filepath)
  margins <- data$margin
  cat("N =", length(margins), "\n")
  result <- rddensity(X = margins, c = 0)
  summary(result)
  cat("\nDensity ratio (right/left):",
      result$hat$right / result$hat$left, "\n")
  return(result)
}

# Main samples
run_test("margins_all.csv",                       "All Measures")
run_test("margins_initiative.csv",                "Initiatives")
run_test("margins_legislative_referendum.csv",    "Legislative Referenda")
run_test("margins_popular_referendum.csv",        "Popular Referenda")

# Fiscal / non-fiscal splits
run_test("margins_legref_fiscal.csv",    "Leg.Ref — Fiscal")
run_test("margins_legref_nonfiscal.csv", "Leg.Ref — Non-fiscal")
run_test("margins_init_fiscal.csv",      "Initiative — Fiscal")
run_test("margins_init_nonfiscal.csv",   "Initiative — Non-fiscal")
"""
    with open(f'{OUTPUT_DIR}/run_rddensity.R', 'w') as f:
        f.write(r_script)
    print("Saved: run_rddensity.R")
    print("NOTE: Run this R script after this Python script completes.")


# ============================================================
# SECTION 13: EXPORT FULL RESULTS FOR APPENDIX
# ============================================================
def export_appendix_csv(all_results):
    """
    Export all covariate balance results across all subsamples
    as a single CSV for the appendix.
    """
    print("\n" + "=" * 70)
    print("SECTION 13: EXPORT FULL BALANCE RESULTS (APPENDIX)")
    print("=" * 70)

    rows = []
    for split_label, results in all_results.items():
        # Compute Holm corrections within this split
        reliable = [r for r in results
                    if r.get('reliable') and r.get('success')]
        if reliable:
            pvals = [r['pv_robust'] for r in reliable]
            _, pv_holm, _, _ = multipletests(pvals, method='holm')
            holm_map = {r['covariate']: ph
                        for r, ph in zip(reliable, pv_holm)}
        else:
            holm_map = {}

        for r in results:
            row = {'split': split_label, 'covariate': r.get('covariate')}
            if r.get('success'):
                row['beta1'] = r.get('coef')
                row['se_robust'] = r.get('se_robust')
                row['pv_robust'] = r.get('pv_robust')
                row['bandwidth'] = r.get('bw')
                row['n_pos_bw'] = r.get('n_pos_bw', '')
                row['n_eff_left'] = r.get('n_eff_left', '')
                row['n_eff_right'] = r.get('n_eff_right', '')
                row['reliable'] = r.get('reliable', False)
                row['p_holm'] = holm_map.get(r['covariate'], '')
            else:
                row['beta1'] = ''
                row['se_robust'] = ''
                row['pv_robust'] = ''
                row['bandwidth'] = ''
                row['n_pos_bw'] = ''
                row['n_eff_left'] = ''
                row['n_eff_right'] = ''
                row['reliable'] = False
                row['p_holm'] = ''
            rows.append(row)

    df_out = pd.DataFrame(rows)
    outpath = f'{OUTPUT_DIR}/appendix_covariate_balance_full.csv'
    df_out.to_csv(outpath, index=False)
    print(f"Saved: {outpath} ({len(df_out)} rows)")
    print(f"Splits: {df_out['split'].nunique()}")
    print(f"Covariates per split: ~{len(df_out) // df_out['split'].nunique()}")


# ============================================================
# MAIN
# ============================================================
def main():
    # 1. Clean data
    df = clean_data(DATA_PATH)

    # 2. Summary statistics
    summary_statistics(df)

    # 3. Export margins for R
    export_margins(df)

    # 4. Covariate balance tests (by type)
    balance_results = covariate_balance_section(df)

    # Identify significant covariates for downstream tests
    legref_results = balance_results.get('Legislative Referendum', [])
    sig_covs = [r['covariate'] for r in legref_results
                if r.get('reliable') and r.get('success')
                and r['pv_robust'] < 0.05
                and r['covariate'] != 'year_num']
    print(f"\nSignificant covariates for downstream tests: {sig_covs}")

    # 5. Fiscal / non-fiscal split (H2)
    split_results = fiscal_split_section(df)

    # 6. Interaction test
    interaction_test_section(df)

    # 7. Placebo cutoff tests
    placebo_section(df, sig_covs)

    # 8. Leave-one-state-out
    leave_one_out_section(df, sig_covs)

    # 9. Fixed-bandwidth robustness
    fixed_bandwidth_section(df, sig_covs)

    # 10. Omnibus F-test
    omnibus_section(df)

    # 11. Density histograms
    density_histograms(df)

    # 12. R script
    generate_r_script()

    # 13. Export full results for appendix
    all_results = {}
    all_results.update(balance_results)
    all_results.update(split_results)
    export_appendix_csv(all_results)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}/")
    print("Next step: Run run_rddensity.R in R with the margin CSVs.")


if __name__ == '__main__':
    main()
