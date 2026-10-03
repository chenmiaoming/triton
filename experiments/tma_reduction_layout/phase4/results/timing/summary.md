# Phase 4 Stage C — Frozen-Binary Timing and Pre-Registered Analysis

Starting HEAD: `1db7ee4e6eb9be0d545a497b4c70492029f8c398`. Raw timing commit: `c962ad8bdc568b7f65a937ab6bd9aa5d76613ffd`.

18 eligible cases (3 PRIMARY, 15 SECONDARY); 2 exclusions absent. Three separately dispatched single-use Modal H100 benchmark containers. Each invocation has 324 conditions (108 canonical, 216 repeated), 10 rounds × 10 scalar CUDA-event samples per condition: 972 invocation-conditions, 9,720 visits, 97,200 samples (32,400 canonical; 64,800 repeated).

Invalid infrastructure/protocol records retained: 2 (2 Modal App startup failures before any dispatch or GPU sample; 0 invalid dispatched benchmark invocations). Every timed and warmup launch is guarded against the Stage B archived CUBIN SHA; canonical reuses its exact binary across B, repeated across R/B. No timing-kernel compilation. Three untimed warmup launches per condition, in first frozen round order, were fixed before collecting outcomes.

GPU UUIDs, in invocation order: `GPU-9283044d-1616-0ec1-89b0-47447c6c1465`, `GPU-9c80eab8-3f56-4e50-d548-95c8cfa03089`, `GPU-19f13a77-e920-863a-247b-57ccabc1c67a`. Replication: distinct invocations on multiple physical UUIDs.

| Invocation | Modal profile | GPU / CC | Driver API / driver | Queried libcudart / Torch CUDA build | PState start→end | SM / memory clock MHz start→end | Power W start→end | Temperature C start→end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | miaomingc | NVIDIA H100 80GB HBM3 / [9, 0] | 13000 / 580.95.05 | 12060 / 13.0 | P0→P0 | 345/2619→1980/2619 | 70.43→339.90 | 28→37 |
| 2 | miaomingc | NVIDIA H100 80GB HBM3 / [9, 0] | 13000 / 580.95.05 | 12060 / 13.0 | P0→P0 | 345/2619→1980/2619 | 70.53→350.80 | 29→37 |
| 3 | miaomingc | NVIDIA H100 80GB HBM3 / [9, 0] | 13000 / 580.95.05 | 12060 / 13.0 | P0→P0 | 345/2619→1980/2619 | 68.54→322.95 | 28→35 |

All displayed G and Δg(1) values are mean ± sample SD in ns/additional CTA. This is a marginal grid-time slope, not single-CTA latency. The frozen t band (t=4.302652729911275, df=2; half-width=t×SD/√3) is used only for sign resolution. Continuous values remain unrounded for association and rank calculations.

## Complete eligible cohort

| Case | Class | Origin | G canonical mean ± SD | G category | Min canonical R² | Δg(1) mean ± SD | D category | Min repeated R² | Sign relationship | Taxonomy | Outlier score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | SECONDARY | NEW_GENERALIZATION | 0.00232538306 ± 0.0214938417 | SIGN_UNRESOLVED | 0.999997483 | -0.0275528821 ± 0.0159784724 | SIGN_UNRESOLVED | 0.999994136 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0.214285714 |
| M128_N128_w8 | SECONDARY | NEW_GENERALIZATION | -0.0182294787 ± 0.0157000017 | SIGN_UNRESOLVED | 0.99999643 | -0.00676627245 ± 0.0284854809 | SIGN_UNRESOLVED | 0.999997576 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0.357142857 |
| M128_N16_w4 | SECONDARY | NEW_GENERALIZATION | 0.0424106312 ± 0.00200100003 | POSITIVE | 0.999963269 | -0.0105097175 ± 0.00384187169 | NEGATIVE | 0.999973779 | RESOLVED_OPPOSITION | OPPOSED_CANONICAL_POSITIVE | 0.214285714 |
| M128_N16_w8 | SECONDARY | NEW_GENERALIZATION | 1.10758454 ± 0.00370249611 | POSITIVE | 0.999998835 | 0.580055054 ± 0.00439871675 | POSITIVE | 0.999972151 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0 |
| M128_N32_w4 | SECONDARY | NEW_GENERALIZATION | 0.0132068592 ± 0.0103496943 | SIGN_UNRESOLVED | 0.999993273 | 0.00118575372 ± 0.00511774185 | SIGN_UNRESOLVED | 0.99999292 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0.285714286 |
| M128_N32_w8 | SECONDARY | NEW_GENERALIZATION | 0.63720713 ± 0.0140556519 | POSITIVE | 0.999981021 | -0.00304597896 ± 0.00736040367 | SIGN_UNRESOLVED | 0.999971752 | ONE_SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 0.285714286 |
| M128_N64_w8 | PRIMARY | NEW_GENERALIZATION | -0.0119047118 ± 0.0109253971 | SIGN_UNRESOLVED | 0.999991716 | -0.0127653536 ± 0.00697436981 | SIGN_UNRESOLVED | 0.999996708 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0 |
| M32_N128_w4 | SECONDARY | ANCHOR_EXISTING | 0.00458051848 ± 0.00748428329 | SIGN_UNRESOLVED | 0.999982051 | -0.00220885947 ± 0.00436750756 | SIGN_UNRESOLVED | 0.999989344 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0.285714286 |
| M32_N128_w8 | SECONDARY | NEW_GENERALIZATION | 0.854980466 ± 0.00591421509 | POSITIVE | 0.999993714 | 0.186476909 ± 0.00880456701 | POSITIVE | 0.999958117 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0.142857143 |
| M32_N32_w4 | SECONDARY | NEW_GENERALIZATION | 0.160853785 ± 0.00598794651 | POSITIVE | 0.999778182 | 0.231142976 ± 0.0111477806 | POSITIVE | 0.99935821 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0.142857143 |
| M32_N64_w4 | SECONDARY | NEW_GENERALIZATION | 0.0657087076 ± 0.0049436423 | POSITIVE | 0.999975145 | -0.0042085135 ± 0.00497235356 | SIGN_UNRESOLVED | 0.999975936 | ONE_SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 0.0714285714 |
| M32_N64_w8 | PRIMARY | ANCHOR_EXISTING | 1.41469028 ± 0.0090380954 | POSITIVE | 0.999990422 | 1.04471261 ± 0.00598556047 | POSITIVE | 0.999952264 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0 |
| M64_N128_w8 | SECONDARY | NEW_GENERALIZATION | -0.0179733369 ± 0.0233514283 | SIGN_UNRESOLVED | 0.999997982 | -0.0163689232 ± 0.0171644703 | SIGN_UNRESOLVED | 0.999995683 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0 |
| M64_N16_w4 | SECONDARY | NEW_GENERALIZATION | 0.115234305 ± 0.00946882754 | POSITIVE | 0.999488564 | 0.308268184 ± 0.00705912972 | POSITIVE | 0.998747148 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0.285714286 |
| M64_N32_w4 | SECONDARY | NEW_GENERALIZATION | 0.0738932017 ± 0.008919918 | POSITIVE | 0.999980807 | -0.0161133282 ± 0.00820705758 | SIGN_UNRESOLVED | 0.999978034 | ONE_SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 0.428571429 |
| M64_N32_w8 | SECONDARY | NEW_GENERALIZATION | 1.0411551 ± 0.00393054554 | POSITIVE | 0.999993429 | 0.228213426 ± 0.00645317201 | POSITIVE | 0.999969811 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0.142857143 |
| M64_N64_w4 | SECONDARY | NEW_GENERALIZATION | -0.00276689943 ± 0.0114423493 | SIGN_UNRESOLVED | 0.999995366 | -0.00702195415 ± 0.0104632291 | SIGN_UNRESOLVED | 0.999986265 | BOTH_SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 0.142857143 |
| M64_N64_w8 | PRIMARY | NEW_GENERALIZATION | 1.14311292 ± 0.0124995436 | POSITIVE | 0.999994226 | 0.280924556 ± 0.0152761297 | POSITIVE | 0.999949159 | RESOLVED_AGREEMENT | A_MECHANISM_ALIGNED_POSITIVE | 0 |

## PRIMARY

n=3; Spearman(mean Δg(1), mean G canonical)=1.

PRIMARY n=3<5: cross-shape generalization is not established, regardless of observed rho. Leave-one-case-out results: M128_N64_w8=UNDEFINED, M32_N64_w8=UNDEFINED, M64_N64_w8=UNDEFINED; defined=0, undefined=3.

New-generalization-only n=2, Spearman=UNDEFINED. The anchor remains in PRIMARY; new-only sensitivity is undefined below n=3.

| D category / G category | POSITIVE | NEGATIVE | SIGN_UNRESOLVED |
| --- | --- | --- | --- |
| POSITIVE | 2 | 0 | 0 |
| NEGATIVE | 0 | 0 | 0 |
| SIGN_UNRESOLVED | 0 | 0 | 1 |

Resolved sign agreement: 2/2 (rate=1). SIGN_UNRESOLVED pair coverage: 1/3. Three-category exact agreement: 3/3.

Coverage: `{"M": {"128": 1, "32": 1, "64": 1}, "N": {"64": 3}, "lanePart_transition": {"4->2": 3}, "num_warps": {"8": 3}}`.

Nine-grid case distribution: `{"A_MECHANISM_ALIGNED_POSITIVE": 2, "B_MECHANISM_NEUTRAL_UNRESOLVED": 1}`.

Rank-disagreement audit (score=|average rank(D)−average rank(G)|/(n−1); descending score, then case ID):

| Case | Origin | D rank | G rank | Score |
| --- | --- | --- | --- | --- |
| M128_N64_w8 | NEW_GENERALIZATION | 1 | 1 | 0 |
| M32_N64_w8 | ANCHOR_EXISTING | 3 | 3 | 0 |
| M64_N64_w8 | NEW_GENERALIZATION | 2 | 2 | 0 |

Top min(5,n): M128_N64_w8, M32_N64_w8, M64_N64_w8. Every case is retained.

## SECONDARY

n=15; Pearson(mean Δg(1), mean G canonical)=0.703075497.

SECONDARY is the broader, lower-equivalence-strength descriptive stratum. Observed Pearson direction: positive. No post-hoc correlation threshold or support category is applied. PRIMARY and SECONDARY are separate evidence strata within a structured factorial cohort.

| D category / G category | POSITIVE | NEGATIVE | SIGN_UNRESOLVED |
| --- | --- | --- | --- |
| POSITIVE | 5 | 0 | 0 |
| NEGATIVE | 1 | 0 | 0 |
| SIGN_UNRESOLVED | 3 | 0 | 6 |

Resolved sign agreement: 5/6 (rate=0.833333333). SIGN_UNRESOLVED pair coverage: 9/15. Three-category exact agreement: 11/15.

Coverage: `{"M": {"128": 6, "32": 4, "64": 5}, "N": {"128": 5, "16": 3, "32": 5, "64": 2}, "lanePart_transition": {"16->8": 3, "2->1": 5, "4->2": 2, "8->4": 5}, "num_warps": {"4": 9, "8": 6}}`.

Nine-grid case distribution: `{"A_MECHANISM_ALIGNED_POSITIVE": 5, "B_MECHANISM_NEUTRAL_UNRESOLVED": 6, "E_OTHER_MECHANISM_CANDIDATE": 3, "OPPOSED_CANONICAL_POSITIVE": 1}`.

Rank-disagreement audit (score=|average rank(D)−average rank(G)|/(n−1); descending score, then case ID):

| Case | Origin | D rank | G rank | Score |
| --- | --- | --- | --- | --- |
| M64_N32_w4 | NEW_GENERALIZATION | 3 | 9 | 0.428571429 |
| M128_N128_w8 | NEW_GENERALIZATION | 6 | 1 | 0.357142857 |
| M128_N32_w4 | NEW_GENERALIZATION | 10 | 6 | 0.285714286 |
| M128_N32_w8 | NEW_GENERALIZATION | 8 | 12 | 0.285714286 |
| M32_N128_w4 | ANCHOR_EXISTING | 9 | 5 | 0.285714286 |
| M64_N16_w4 | NEW_GENERALIZATION | 14 | 10 | 0.285714286 |
| M128_N128_w4 | NEW_GENERALIZATION | 1 | 4 | 0.214285714 |
| M128_N16_w4 | NEW_GENERALIZATION | 4 | 7 | 0.214285714 |
| M32_N128_w8 | NEW_GENERALIZATION | 11 | 13 | 0.142857143 |
| M32_N32_w4 | NEW_GENERALIZATION | 13 | 11 | 0.142857143 |
| M64_N32_w8 | NEW_GENERALIZATION | 12 | 14 | 0.142857143 |
| M64_N64_w4 | NEW_GENERALIZATION | 5 | 3 | 0.142857143 |
| M32_N64_w4 | NEW_GENERALIZATION | 7 | 8 | 0.0714285714 |
| M128_N16_w8 | NEW_GENERALIZATION | 15 | 15 | 0 |
| M64_N128_w8 | NEW_GENERALIZATION | 2 | 2 | 0 |

Top min(5,n): M64_N32_w4, M128_N128_w8, M128_N32_w4, M128_N32_w8, M32_N128_w4. Every case is retained.

## Anchor temporal comparison

M32_N64_w8: Phase 4 G=1.41469028 ± 0.0090380954, Δg(1)=1.04471261 ± 0.00598556047.

Phase 3 historical Δg(1)=1.09288969 ± 0.0117706581; its Phase 2 canonical baseline G=1.45958897 ± 0.0233241007. Fresh Phase 4 descriptive D/G=0.738474439. Descriptive temporal comparison only. Phase 4 uses its fresh canonical denominator; no historical-denominator attribution or additive causal claim.

## Full counterexample and structural audit

All nine-grid classifications are listed, including aligned, unresolved, masked, and opposed cases. No category is dropped.

| Case | G | D | D/G categories | Taxonomy | M×N / warps / lanePart | Warp partition default→4; canonical / repeated | Body equivalence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | 0.00232538306 ± 0.0214938417 | -0.0275528821 ± 0.0159784724 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 128×128 / 4 / 2->1 | canonical: 4→4; repeated: 4→4 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N128_w8 | -0.0182294787 ± 0.0157000017 | -0.00676627245 ± 0.0284854809 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 128×128 / 8 / 2->1 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N16_w4 | 0.0424106312 ± 0.00200100003 | -0.0105097175 ± 0.00384187169 | NEGATIVE/POSITIVE | OPPOSED_CANONICAL_POSITIVE | 128×16 / 4 / 16->8 | canonical: 4→4; repeated: 4→4 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N16_w8 | 1.10758454 ± 0.00370249611 | 0.580055054 ± 0.00439871675 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 128×16 / 8 / 16->8 | canonical: 8→8; repeated: 8→8 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N32_w4 | 0.0132068592 ± 0.0103496943 | 0.00118575372 ± 0.00511774185 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 128×32 / 4 / 8->4 | canonical: 4→4; repeated: 4→4 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N32_w8 | 0.63720713 ± 0.0140556519 | -0.00304597896 ± 0.00736040367 | SIGN_UNRESOLVED/POSITIVE | E_OTHER_MECHANISM_CANDIDATE | 128×32 / 8 / 8->4 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M128_N64_w8 | -0.0119047118 ± 0.0109253971 | -0.0127653536 ± 0.00697436981 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 128×64 / 8 / 4->2 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'EXACT_SEQUENCE_EQUIVALENT'} |
| M32_N128_w4 | 0.00458051848 ± 0.00748428329 | -0.00220885947 ± 0.00436750756 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 32×128 / 4 / 2->1 | canonical: 4→4; repeated: 4→4 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M32_N128_w8 | 0.854980466 ± 0.00591421509 | 0.186476909 ± 0.00880456701 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 32×128 / 8 / 2->1 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M32_N32_w4 | 0.160853785 ± 0.00598794651 | 0.231142976 ± 0.0111477806 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 32×32 / 4 / 8->4 | canonical: 4→4; repeated: 4→4 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M32_N64_w4 | 0.0657087076 ± 0.0049436423 | -0.0042085135 ± 0.00497235356 | SIGN_UNRESOLVED/POSITIVE | E_OTHER_MECHANISM_CANDIDATE | 32×64 / 4 / 4->2 | canonical: 4→4; repeated: 4→4 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M32_N64_w8 | 1.41469028 ± 0.0090380954 | 1.04471261 ± 0.00598556047 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 32×64 / 8 / 4->2 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'EXACT_SEQUENCE_EQUIVALENT'} |
| M64_N128_w8 | -0.0179733369 ± 0.0233514283 | -0.0163689232 ± 0.0171644703 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 64×128 / 8 / 2->1 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M64_N16_w4 | 0.115234305 ± 0.00946882754 | 0.308268184 ± 0.00705912972 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 64×16 / 4 / 16->8 | canonical: 4→4; repeated: 4→4 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M64_N32_w4 | 0.0738932017 ± 0.008919918 | -0.0161133282 ± 0.00820705758 | SIGN_UNRESOLVED/POSITIVE | E_OTHER_MECHANISM_CANDIDATE | 64×32 / 4 / 8->4 | canonical: 4→4; repeated: 4→4 | {'4': 'PIPELINED_OPCODE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M64_N32_w8 | 1.0411551 ± 0.00393054554 | 0.228213426 ± 0.00645317201 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 64×32 / 8 / 8->4 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M64_N64_w4 | -0.00276689943 ± 0.0114423493 | -0.00702195415 ± 0.0104632291 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 64×64 / 4 / 4->2 | canonical: 4→4; repeated: 4→4 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'PIPELINED_OPCODE_EQUIVALENT'} |
| M64_N64_w8 | 1.14311292 ± 0.0124995436 | 0.280924556 ± 0.0152761297 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 64×64 / 8 / 4->2 | canonical: 8→8; repeated: 8→8 | {'4': 'EXACT_SEQUENCE_EQUIVALENT', 'default': 'EXACT_SEQUENCE_EQUIVALENT'} |

## Fit diagnostics

All 324 OLS fits use exactly three B points and retain all cases, slopes, intercepts, R² and residuals in results.json. No R² cutoff is introduced. These are the five lowest finite R² fits per harness for inspection; constant-response fits have undefined R² and remain in the full results.

canonical:

| Case | Invocation | Candidate | R | R² | Residuals (us) |
| --- | --- | --- | --- | --- | --- |
| M64_N16_w4 | 1 | 4 | None | 0.999488564 | -0.710856702, 1.06628505, -0.355428351 |
| M64_N16_w4 | 2 | 4 | None | 0.99952572 | -0.685714185, 1.02857128, -0.342857093 |
| M64_N16_w4 | 3 | 4 | None | 0.999611457 | -0.621713698, 0.932570547, -0.310856849 |
| M32_N32_w4 | 3 | 4 | None | 0.999778182 | -0.42057197, 0.630857955, -0.210285985 |
| M32_N32_w4 | 2 | 4 | None | 0.999828061 | -0.372571605, 0.558857407, -0.186285802 |

repeated:

| Case | Invocation | Candidate | R | R² | Residuals (us) |
| --- | --- | --- | --- | --- | --- |
| M64_N16_w4 | 2 | 4 | 0 | 0.998747148 | -0.473143267, 0.7097149, -0.236571633 |
| M64_N16_w4 | 2 | default | 0 | 0.999121365 | -0.393142125, 0.589713188, -0.196571063 |
| M64_N16_w4 | 1 | default | 0 | 0.999260337 | -0.358857214, 0.538285822, -0.179428607 |
| M32_N32_w4 | 2 | default | 0 | 0.99935821 | -0.315428578, 0.473142868, -0.157714289 |
| M64_N16_w4 | 1 | 4 | 0 | 0.99939205 | -0.324570973, 0.48685646, -0.162285487 |

All-eligible pooled signs are descriptive only: `{"SIGN_UNRESOLVED_pair_coverage": {"count": 10, "denominator": 18}, "axis_order": "delta_g1/G_canonical", "matrix": {"NEGATIVE": {"NEGATIVE": 0, "POSITIVE": 1, "SIGN_UNRESOLVED": 0}, "POSITIVE": {"NEGATIVE": 0, "POSITIVE": 7, "SIGN_UNRESOLVED": 0}, "SIGN_UNRESOLVED": {"NEGATIVE": 0, "POSITIVE": 3, "SIGN_UNRESOLVED": 7}}, "resolved_sign_agreement": {"denominator": 8, "numerator": 7, "rate": 0.875}, "three_category_exact_agreement": {"denominator": 18, "numerator": 14}}`.

## Fidelity and scientific status

Raw integrity and corruption probes: raw_validation.json. Independent rational-moment/OLS/rank analysis recomputation and adverse-outcome synthetic acceptance: validation.json. Final verification commands: validate_evidence.py --self-test; validate_evidence.py; validate_preregistration.py; validate_artifact_gate.py; validate_timing_raw.py; validate_timing_analysis.py. All six commands must pass after the analysis commit before the normal push. No raw file is changed after the first raw timing commit.

Phase 4 PRIMARY status: `INSUFFICIENT_PRIMARY_COHORT_SIZE_FOR_ESTABLISHED_GENERALIZATION`. H2a historical status remains SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b and H2c remain UNVERIFIED. No production heuristic is proposed, no upstream PR is modified, and Stage D is not started.
