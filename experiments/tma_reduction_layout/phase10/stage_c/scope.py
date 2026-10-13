"""Scope of support for the unchanged prototype, with all uncertainty retained."""
import argparse
from collections import Counter
from experiments.tma_reduction_layout.phase10.stage_c import common as c


def derive():
    analysis = c.read(c.OUT / 'results.json')
    gate = c.read(c.OUT / 'artifact_gate.json')
    targets = {}
    for target, data in analysis['targets'].items():
        rows = [r for r in analysis['rows'] if r['target'] == target]
        primary = [r for r in rows if r['fraction'] == 1]
        groups = {}
        for cohort in ('held_out', 'anchor', 'guard_control', 'cost_fallback_probe'):
            subset = [r for r in primary if r['cohort'] == cohort]
            groups[cohort] = {
                'pairs': len(subset), 'decisions': dict(Counter(r['status'] for r in subset)), 'changed_CUBINs':
                sum(not r['same_CUBIN'] for r in subset)
            }
        targets[target] = {
            'validation': data, 'primary_cohorts': groups, 'all_grid_decisions':
            {str(f): dict(Counter(r['status']
                                  for r in rows
                                  if r['fraction'] == f))
             for f in (1, 4, 16)}, 'resource_strata': dict(Counter(r['resource_stratum'] for r in primary)),
            'unavailable': gate[target]['unavailable']
        }
    unresolved = [r for r in analysis['rows'] if r['status'] == 'UNRESOLVED']
    regressions = [r for r in analysis['rows'] if r['status'] == 'REGRESSION']
    return {
        'targets':
        targets, 'classified_regressions':
        regressions, 'unresolved':
        unresolved, 'unresolved_bands_allowing_over_3_percent_loss': [r for r in unresolved if r['band'][0] < -3],
        'changed_code_unresolved_bands_allowing_over_3_percent_loss':
        [r for r in unresolved if r['band'][0] < -3 and not r['same_CUBIN']], 'same_code_positive_classifications':
        [r for r in analysis['rows']
         if r['same_CUBIN'] and r['status'] == 'BENEFIT'], 'universal_non_regression_established':
        False, 'performance_tuning_from_this_cohort':
        False, 'readiness':
        'Supports review of the frozen structural cost prototype on these tested cells; unresolved and unavailable coverage prevents a universal safety or benefit claim.'
    }


def summary(result):
    text = '# Stage C scope — Actual compiler patch\n\n'
    text += ('This is an actual-patch contrast with the same upstream and assembler. '
             'All 372 compiler attempts and 158,400 exact-binary samples are retained; '
             '352 compiler attempts pass full-output correctness and 20 exceed SM120 shared-memory limits. '
             'No native rebuild, rule adjustment, outcome-selected exclusion or timing rerun occurred.\n\n')
    text += '| Target | Eligible held-out pairs | Held-out benefit | Within ±3% | Unresolved | Anchor benefit |\n| --- | --- | --- | --- | --- | --- |\n'
    for target, data in result['targets'].items():
        held = data['primary_cohorts']['held_out']
        anchor = data['primary_cohorts']['anchor']['decisions']
        counts = held['decisions']
        text += f'| {target} | {held["pairs"]} | {counts.get("BENEFIT", 0)} | {counts.get("WITHIN_PRACTICAL_BAND", 0)} | {counts.get("UNRESOLVED", 0)} | {anchor.get("BENEFIT", 0)} |\n'
    text += (f'\nThe fixed descriptive bands classify {len(result["classified_regressions"])} regressions at any '
             f'of the three grids. They leave {len(result["unresolved"])} case/grid effects unresolved, including '
             f'{len(result["unresolved_bands_allowing_over_3_percent_loss"])} bands allowing losses beyond 3% '
             f'({len(result["changed_code_unresolved_bands_allowing_over_3_percent_loss"])} with changed CUBINs). '
             'Zero classified regressions is not proof that all cases avoid a regression. '
             'These are three-process bands without familywise significance.\n\n')
    text += ('At the full grid, the canonical M32/N64/8-warps device path decreases time by '
             '41.170% on H100, 21.438% on B200 and 17.753% on SM120; all three exceed the '
             'frozen practical benefit threshold. The host path is a benefit on H100 and B200 '
             '(45.801%, 45.099%), while the SM120 host effect is unresolved (mean 16.052%, '
             'band [-6.671%, 38.776%]). The original M32/N128/4-warps host case decreases '
             'time by 19.813% on B200; its H100 2.094% and SM120 0.409% contrasts remain '
             'within the practical band.\n\n')
    text += ('The held-out full-grid subset gives 2 H100 and 13 B200 benefit classifications, '
             'and no SM120 held-out benefit classifications. Most SM120 held-out effects are '
             'within the practical band; two are unresolved. Ten SM120 pairs remain unavailable '
             '(eight held-out FP32/sum paths and two large-input BF16 fallback probes), with '
             'the same resource failures in baseline and patch. All six guard controls per '
             'architecture preserve identical CUBINs. Resource-changing and local-memory '
             'cases remain included rather than screened out.\n\n')
    text += ('The rule has broader evidence than a fixed-vector shape heuristic: it selects from '
             'structural instruction costs, preserves guards and cost-dominated defaults, and '
             'shows benefit on several unseen cells and on the canonical device path across '
             'three architectures. The declared dtype/op subset, finite-value oracle, tested '
             'warp counts and resource availability do not cover all eligible programs or '
             'floating-point edge cases. B200/SM100 is not GB300/SM103. This phase does not '
             'attribute the original historical compiler regression to a particular commit '
             'or isolate bank-conflict causality.\n\n')
    text += ('Before a PR, review the model assumptions and remaining uncertain/resource-limited '
             'coverage, then decide whether another independently frozen confirmation is needed. '
             'No further phase, native change, PR creation/update or reviewer message is started. '
             'User approval remains required before any PR.\n\n')
    text += '[All case/grid results](results.json) · [Full primary table](summary.md) · [Scope data](scope.json)\n'
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    result = derive()
    text = summary(result)
    if args.validate:
        assert (c.OUT / 'scope.json').read_bytes() == c.encode(result)
        assert (c.OUT / 'scope.md').read_text() == text
    else:
        c.write_once(c.OUT / 'scope.json', result)
        c.write_once(c.OUT / 'scope.md', text.encode())
    print('Scope PASS: no universal safety claim; all uncertainty retained')


if __name__ == '__main__':
    main()
