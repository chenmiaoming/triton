# PHASE 5 HELD-OUT TIMING REPORT

Starting HEAD: `56128d73233b234dacd191c569971b418680b857`. Raw commit: `fa91a5c2d39bf65b0b18938ffbf81113137a3adb`.

13 eligible cases: 2 PRIMARY, 11 SECONDARY. Six artifact exclusions absent. Three independent single-use Modal H100 invocations; 234 conditions/invocation (78 canonical, 156 repeated); 702 invocation-level conditions; 70,200 scalar CUDA-event samples. 10 rounds × 10 one-kernel samples; three untimed warmups per condition/invocation.

UUIDs: GPU-f93999f3-450b-dc8e-de2c-6cb2d6a8f620, GPU-5402b572-d234-eb08-c3f5-fa4abaf6a3d0, GPU-e0c67887-7140-23e5-c818-fe1174a5cc86. Replication: distinct invocations on multiple physical UUIDs.

| Invocation | GPU / CC | Driver | CUDA runtime | Temperature C start→end | Power W start→end | SM / memory clock MHz start→end |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | NVIDIA H100 80GB HBM3 / [9, 0] | 580.95.05 | 12060 | 34→44 | 71.48→421.32 | 345/2619→1980/2619 |
| 2 | NVIDIA H100 80GB HBM3 / [9, 0] | 580.95.05 | 12060 | 24→33 | 69.96→391.30 | 345/2619→1980/2619 |
| 3 | NVIDIA H100 80GB HBM3 / [9, 0] | 580.95.05 | 12060 | 29→38 | 69.89→390.23 | 345/2619→1980/2619 |

All runtime CUBIN SHAs match frozen Stage B bytes; no JIT, Triton import, ptxas or recompilation. Exact full descriptor allocation and registered R/B dimensions retained. Invalid attempt records: 6. Initial import-path startup failure occurred before benchmark entry; original source snapshot and full logs remain retained. A subsequent INVALID_PROTOCOL_RUN return was lost after a local invalid-attempt filename collision; consumed Modal output was unavailable on retrieval, so its original error/partial samples cannot be recovered. Its call ID, logs and source snapshot remain retained and none of its samples is accepted. Three later cuTensorMapEncodeTiled failures retain complete returns with zero warmups/visits. Repair reproduces frozen compiler getTMABlockShapeTiled message-box clipping at 256, without altering full descriptor bounds or kernel bytes. All repairs rerun the entire invocation; no outcome-based repeats.

All differential values are ns/additional CTA, mean ± sample SD across three invocations. Sign resolution uses frozen t(df=2)=4.302652729911275; unresolved signs are not practical equivalence.

| Case | Class | Warps | G mean±SD | g0 mean±SD | D mean±SD | G/D categories | A prediction / abs error | B prediction / abs error | C prediction / abs error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M16_N128_w4 | SECONDARY | 4 | 0.10412030146 ± 0.00717991554064 | -0.00295294523225 ± 0.00419823484864 | 0.0465029729856 ± 0.00304472453765 | POSITIVE/POSITIVE | 0.215541538348 / 0.111421236888 | 0.199950458248 / 0.0958301567878 | 0.00730816521985 / 0.0968121362402 |
| M16_N128_w8 | SECONDARY | 8 | 1.19356859938 ± 0.00814890481838 | 0.00125559533141 ± 0.00326879966913 | 0.886160700291 ± 0.00933590487668 | POSITIVE/POSITIVE | 1.41821878049 / 0.224650181102 | 1.41222484762 / 0.21865624824 | 1.40662896658 / 0.213060367198 |
| M16_N64_w4 | SECONDARY | 4 | 0.1116303511 ± 0.00263261044512 | -0.000744047870934 ± 0.00221720774882 | 0.0729398997074 ± 0.0114198087772 | POSITIVE/POSITIVE | 0.253408265571 / 0.141777914471 | 0.252971086296 / 0.141340735196 | 0.065459749569 / 0.0461706015314 |
| M256_N128_w8 | SECONDARY | 8 | -0.0107882529153 ± 0.0168468325901 | -0.00144180895794 ± 0.0143560127484 | -0.0155552178122 ± 0.0246581323721 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.126652969431 / 0.137441222346 | 0.123386061373 / 0.134174314288 | 0.379257795265 / 0.390046048181 |
| M256_N16_w4 | SECONDARY | 4 | -0.0210426107861 ± 0.02970348168 | 0.000372145742793 ± 0.00335427097338 | 0.00239472661806 ± 0.00385699778291 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.152363433304 / 0.17340604409 | 0.161632791012 / 0.182675401798 | 0.0025517043266 / 0.0235943151127 |
| M256_N16_w8 | SECONDARY | 8 | 0.365745834296 ± 0.0144433257946 | -0.00685922497637 ± 0.00142329782335 | -0.00106947915074 ± 0.0108477461266 | POSITIVE/SIGN_UNRESOLVED | 0.147401505003 / 0.218344329293 | 0.105031856169 / 0.260713978127 | 0.323892810191 / 0.0418530241054 |
| M256_N32_w8 | SECONDARY | 8 | -0.0131834288671 ± 0.00884726066074 | -0.00999805360195 ± 0.00632298557953 | 0.0107188606351 ± 0.00499246909638 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.164286442189 / 0.177469871056 | 0.0991779080793 / 0.112361336946 | 0.295549398987 / 0.308732827854 |
| M256_N64_w4 | SECONDARY | 4 | -6.98362003148e-05 ± 0.00527487126306 | -0.00811477158485 ± 0.0168520248968 | -0.00018631648605 ± 0.0100295834383 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.148666496205 / 0.148736332406 | 0.0972947697557 / 0.097364605956 | -0.111984356989 / 0.111914520789 |
| M256_N64_w8 | PRIMARY | 8 | -0.00830069828745 ± 0.0146498942104 | 0.00381327308354 ± 0.0095936341073 | -0.00434792604037 ± 0.00912776378696 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.142705646994 / 0.151006345281 | 0.176752912849 / 0.185053611137 | 0.460748137331 / 0.469048835619 |
| M512_N16_w8 | SECONDARY | 8 | 0.0044643413622 ± 0.0045895365008 | -0.00148818236041 ± 0.000967533548179 | 0.00225537362331 ± 0.00197834535112 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.152163832121 / 0.147699490759 | 0.148130280711 / 0.143665939349 | 0.398238784593 / 0.393774443231 |
| M512_N32_w4 | SECONDARY | 4 | 0.00360408930213 ± 0.0105115711355 | -0.00223237640014 ± 0.00570804984698 | -0.00051107105849 ± 0.00214457804645 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.148201336509 / 0.144597247207 | 0.138912369949 / 0.135308280647 | -0.0349229315839 / 0.038527020886 |
| M512_N32_w8 | SECONDARY | 8 | -0.0133465315836 ± 0.012208674189 | 0.00176725647872 ± 0.000805742641835 | -0.00274368565962 ± 0.00117207415023 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.145003468278 / 0.158349999861 | 0.16437707649 / 0.177723608074 | 0.435584991608 / 0.448931523191 |
| M512_N64_w8 | PRIMARY | 8 | -0.00267382780467 ± 0.0125672279341 | -0.0149273633031 ± 0.0272103980968 | 0.00846310805563 ± 0.0348313766586 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | 0.161055432453 / 0.163729260258 | 0.0607442621717 / 0.0634180899764 | 0.22819309495 / 0.230866922755 |

All individual invocation values, medians/means/SD/IQR/min/max, 234 OLS fits, intercepts, slopes, R², three residuals, G/g0/g1/D/G-D and every squared prediction error are preserved at full JSON precision in results.json.

| Case | Squared error A | Squared error B | Squared error C |
| --- | --- | --- | --- |
| M16_N128_w4 | 0.0124146920297 | 0.00918341894997 | 0.00937258972339 |
| M16_N128_w8 | 0.050467703869 | 0.0478105548945 | 0.0453947200705 |
| M16_N64_w4 | 0.0201009770317 | 0.0199772034257 | 0.00213172444577 |
| M256_N128_w8 | 0.0188900895999 | 0.0180027466146 | 0.152135919701 |
| M256_N16_w4 | 0.030069656127 | 0.0333703024221 | 0.000556691705638 |
| M256_N16_w8 | 0.0476742461344 | 0.0679717783907 | 0.00175167562677 |
| M256_N32_w8 | 0.0314955551326 | 0.0126250700404 | 0.0953159589949 |
| M256_N64_w4 | 0.0221224965775 | 0.00947986649297 | 0.0125248599634 |
| M256_N64_w8 | 0.0228029163152 | 0.0342448389947 | 0.220006810195 |
| M512_N16_w8 | 0.0218151395705 | 0.0206399021289 | 0.155058312142 |
| M512_N32_w4 | 0.0209083638998 | 0.0183083308117 | 0.00148433133835 |
| M512_N32_w8 | 0.0250747224561 | 0.0315856808668 | 0.201539512515 |
| M512_N64_w8 | 0.0268072706647 | 0.00402185413625 | 0.0532995360223 |

## Locked coefficients and confirmatory decisions

`{"Stage_D_source_result_SHA256": "1b284511ac840c11efdcdba2a87f752b7a759eb51783dd01edde669560c7b2cc", "Stage_D_source_result_path": "experiments/tma_reduction_layout/phase4/results/residual_analysis/results.json", "coefficient_source": "Frozen Phase 5 protocol.json; no refit/calibration", "coefficients": {"A": {"D": 1.43234225450178, "intercept": 0.14893336518099015}, "B": {"D": 1.4079214075667394, "g0": 7.152669582225426, "intercept": 0.15559936860663273}, "C": {"D": 1.0999810571362003, "g0": 13.161075863171975, "intercept": -0.004980287943348076, "warp8": 0.4203242852172698}}, "preregistration_source_bindings_SHA256": "f5b64213fcaefc58895d102e79a7a8f93b8592a327d1a431e78dada60c79b051", "protocol_SHA256": "b6f22944d88c55579af4b7b67ebf7237a8688061260d2fa9520bd04cd400093b"}`

Case is the prediction unit: target mean G, predictors mean D and mean g0. Same Model B metrics and same all-eligible n=13 population in H5_02 and H5_03. No refit/calibration/tolerance/significance layer.

| Model | n | MAE | RMSE |
| --- | --- | --- | --- |
| A | 13 | 0.16143303654 | 0.16423337743 |
| B | 13 | 0.149868177425 | 0.158653358996 |
| C | 13 | 0.216410198976 | 0.270408898699 |

H5_01_EXACT_BODY_DIRECTIONAL_TRACKING: **INCONCLUSIVE_BY_COVERAGE**. `{"PRIMARY_n": 2, "Spearman_computed": false, "pre_timing_feasibility": "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING", "required": 5, "status": "INCONCLUSIVE_BY_COVERAGE"}`

H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT: **FALSIFIED**. `{"Model_A": {"MAE": 0.16143303653981128, "RMSE": 0.1642333774302596, "n": 13, "unit": "one held-out case; three-invocation means"}, "Model_B": {"MAE": 0.1498681774247632, "RMSE": 0.15865335899601996, "n": 13, "unit": "one held-out case; three-invocation means"}, "n": 13, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES", "status": "FALSIFIED"}`

H5_03_WARP_REGIME_CONTEXT_PREDICTION: **FALSIFIED**. `{"Model_B": {"MAE": 0.1498681774247632, "RMSE": 0.15865335899601996, "n": 13, "unit": "one held-out case; three-invocation means"}, "Model_C": {"MAE": 0.21641019897635447, "RMSE": 0.27040889869914536, "n": 13, "unit": "one held-out case; three-invocation means"}, "eligible_by_warps": {"4": 5, "8": 8}, "n": 13, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES", "status": "FALSIFIED"}`

Adding g0 in locked Model B lowered both held-out MAE and RMSE relative to locked Model A, contradicting H5_02's registered non-improvement prediction.

Locked Model C did not improve either registered error metric relative to the same locked Model B, contradicting H5_03's registered prospective extension prediction.

H5_01 remains outcome-independent INCONCLUSIVE_BY_COVERAGE (PRIMARY n=2<5); no n=2 Spearman computed.

## Prediction failures (descriptive only)

| Case | Model | Class | M×N / warps | Absolute error | Squared error |
| --- | --- | --- | --- | --- | --- |
| M256_N64_w8 | C | PRIMARY | 256×64 / 8 | 0.469048835619 | 0.220006810195 |
| M512_N32_w8 | C | SECONDARY | 512×32 / 8 | 0.448931523191 | 0.201539512515 |
| M512_N16_w8 | C | SECONDARY | 512×16 / 8 | 0.393774443231 | 0.155058312142 |
| M256_N128_w8 | C | SECONDARY | 256×128 / 8 | 0.390046048181 | 0.152135919701 |
| M256_N32_w8 | C | SECONDARY | 256×32 / 8 | 0.308732827854 | 0.0953159589949 |

All 39 case/model prediction errors are ranked and retained. No counterexample, opposed sign or poor fit removed.

## Fit diagnostics

canonical:

| Case | Invocation | Candidate | R | R² | Three residuals (us) |
| --- | --- | --- | --- | --- | --- |
| M256_N16_w4 | 1 | default | None | 0.999668873155 | 1.11542748553, -1.6731412283, 0.557713742767 |
| M16_N128_w4 | 3 | default | None | 0.999845834249 | 0.441143555301, -0.661715332951, 0.22057177765 |
| M16_N128_w4 | 1 | default | None | 0.999873584227 | 0.40000091706, -0.60000137559, 0.20000045853 |
| M16_N64_w4 | 2 | 4 | None | 0.999933078301 | -0.22857104029, 0.342856560435, -0.114285520145 |
| M16_N64_w4 | 1 | 4 | None | 0.999971082862 | -0.150857759374, 0.22628663906, -0.0754288796868 |

repeated:

| Case | Invocation | Candidate | R | R² | Three residuals (us) |
| --- | --- | --- | --- | --- | --- |
| M16_N64_w4 | 2 | default | 1 | 0.999641570865 | -0.272000208497, 0.408000312746, -0.136000104249 |
| M16_N64_w4 | 1 | default | 0 | 0.999651202685 | -0.233143301947, 0.34971495292, -0.116571650973 |
| M16_N64_w4 | 2 | 4 | 1 | 0.999707168231 | -0.226286372968, 0.339429559452, -0.113143186484 |
| M16_N64_w4 | 1 | 4 | 1 | 0.999710926035 | -0.223999044725, 0.335998567087, -0.111999522362 |
| M16_N64_w4 | 1 | default | 1 | 0.999718427974 | -0.244570629937, 0.366855944906, -0.122285314969 |

## Interpretation and validation

Deterministic prospective prediction comparisons do not establish a causal mechanism or production policy. G-D remains a cross-harness descriptive residual. Barrier copies may interact with live ranges/registers/scheduling; no zero differential assumption. Phase 4 PRIMARY n=3 generalization remains not established; H2a historical SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b/H2c UNVERIFIED. No production heuristic, Coalesce.cpp change, or PR #11991 modification.

Raw manifest/validation: raw_manifest.json and raw_validation.json. Independent rational-moment/OLS/prediction validation and corruption/synthetic branches: validation.json. Final old/new validator suite: validator_suite.json. All baseline experiment bytes and first-commit raw bytes remain identical. The unmodified Phase 5 Stage B validator explicitly rejects a later phase5_timing directory. It is replayed in a detached checkout of trusted baseline 56128d...; the Stage C validator separately proves every one of the 3514 prior experiment files in the current checkout equals that baseline. The initial stage-scope failure and complete diagnostic output are retained in validator_suite.json; no frozen checker or gate was changed. executed_schedule.json retains the original Stage B preview's descriptive metadata; executed=true and the three VALID_PROTOCOL_RUN records identify actual execution. The master condition dictionary retains the full frozen structural domain; only eligibility-filtered round orders are launched. Raw and analysis commit SHAs are verified separately in the final Git report. Stop after this Stage C.
