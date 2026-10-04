# Phase 6 Stage A — Existing-evidence diagnostic

31 cases recomputed from six historical raw invocations, with exact-rational OLS sums. All 39 Phase 5 frozen-model errors retained. No GPU, compiler execution, new timing or model refit.

| Case | G | G sign | D | D sign | Preload B/thread d/4 | Registers d→4 | Blocks/SM | Across-harness residency matched |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M32_N64_w8 | 1.414690283733459 | POSITIVE | 1.044712613899416 | POSITIVE | {"4": 16, "default": 16} | canonical:29→22; single:27→21; repeated:32→23 | canonical:{"4": 8, "default": 8}; single:{"4": 8, "default": 8}; repeated:{"4": 8, "default": 8} | True |
| M64_N64_w8 | 1.1431129245292615 | POSITIVE | 0.2809245556601949 | POSITIVE | {"4": 32, "default": 32} | canonical:29→22; single:26→21; repeated:32→23 | canonical:{"4": 8, "default": 8}; single:{"4": 8, "default": 8}; repeated:{"4": 8, "default": 8} | True |
| M128_N64_w8 | -0.011904711798359871 | SIGN_UNRESOLVED | -0.012765353641866225 | SIGN_UNRESOLVED | {"4": 64, "default": 64} | canonical:29→22; single:26→22; repeated:32→24 | canonical:{"4": 8, "default": 8}; single:{"4": 8, "default": 8}; repeated:{"4": 8, "default": 8} | True |
| M256_N64_w8 | -0.008300698287452354 | SIGN_UNRESOLVED | -0.004347926040368932 | SIGN_UNRESOLVED | {"4": 128, "default": 128} | canonical:32→36; single:39→37; repeated:39→30 | canonical:{"4": 6, "default": 6}; single:{"4": 6, "default": 6}; repeated:{"4": 6, "default": 6} | True |
| M512_N64_w8 | -0.002673827804669552 | SIGN_UNRESOLVED | 0.008463108055624483 | SIGN_UNRESOLVED | {"4": 256, "default": 256} | canonical:42→35; single:42→32; repeated:80→32 | canonical:{"4": 3, "default": 3}; single:{"4": 3, "default": 3}; repeated:{"4": 3, "default": 3} | True |

Units: ns/additional CTA; not single-CTA latency. Five historical PRIMARY cases are a retrospective diagnostic family, never a newly confirmed n=5 cohort.

| Model / descriptive group | n | MAE | RMSE | Mean prediction−G |
| --- | --- | --- | --- | --- |
| A:ALL_13 | 13 | 0.1614330365398114 | 0.16423337743025965 | 0.127841601263982 |
| A:M256_OR_512_W8 | 7 | 0.16486293126493126 | 0.16671615735964024 | 0.10247883718124813 |
| A:W4 | 5 | 0.1439877550122638 | 0.14533835396451278 | 0.1439877550122638 |
| A:W8 | 8 | 0.17233633749452865 | 0.175009872486662 | 0.1177502551713059 |
| B:ALL_13 | 13 | 0.1498681774247627 | 0.15865335899601946 | 0.10975833463603761 |
| B:M256_OR_512_W8 | 7 | 0.1538729825566501 | 0.1643566988388159 | 0.0793832745204464 |
| B:W4 | 5 | 0.13050383607701896 | 0.1344017277437272 | 0.13050383607701896 |
| B:W8 | 8 | 0.16197089076710253 | 0.17208371003194306 | 0.09679239623542427 |
| C:ALL_13 | 13 | 0.21641019897635552 | 0.2704088986991442 | 0.1648290753530044 |
| C:M256_OR_512_W8 | 7 | 0.32617908927652095 | 0.3543823047011042 | 0.31422108238926416 |
| C:W4 | 5 | 0.06340371891181987 | 0.07220830586093943 | -0.053965992866733536 |
| C:W8 | 8 | 0.3120392490166903 | 0.3399452980384045 | 0.3015759929903406 |

Model C's fixed warp8 contribution is examined alongside all its other frozen terms. This failure audit identifies a context-transfer question, not a causal warp explanation.

Next experiment: reuse all 30 archived binaries in canonical/single/repeated harnesses; measure unspecialized runtime R=0,1,2,4,8 in one repeated CUBIN per candidate. Check all resources and full family matches before timing. Freeze a fresh cohort before seeing any new outcomes.

- Outcome-informed diagnostic selection; all five cases already observed. No new confirmation.
- Changing M changes payload/resources/body together. No isolated M/lane/warp causality.
- Opcode sequence equivalence is not operand/dataflow or whole-program equivalence.
- Same blocks/SM does not establish same register live ranges or scheduling.
- SIGN_UNRESOLVED is not practical equivalence; G-D is descriptive.
- No model refit; original H5_01/02/03 decisions and H2 statuses unchanged.
