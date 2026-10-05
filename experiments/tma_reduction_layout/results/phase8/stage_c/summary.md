# Phase8 stage_c — Frozen runtime descriptor results

OLD_CASE_DIAGNOSTIC

ns/additional CTA; layout gaps and runtime descriptor-mode contrasts; not intrinsic component latency

| Case | Class | G | Host static | Device static | Host switch | Device switch | Switch mode effect |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1024_N32_w16 | PRIMARY | -0.0418526940935 ± 0.0737794183347 (SIGN_UNRESOLVED) | -0.0103705349916 ± 0.0129376796442 (SIGN_UNRESOLVED) | -0.07914769369 ± 0.0398524331448 (SIGN_UNRESOLVED) | 0.00106887282094 ± 0.0334648825691 (SIGN_UNRESOLVED) | -0.0634762771023 ± 0.0361947504809 (SIGN_UNRESOLVED) | -0.0645451499232 ± 0.00691134309258 (NEGATIVE) |
| M128_N32_w16 | PRIMARY | 2.92764122735 ± 0.121665970687 (POSITIVE) | 3.20949594165 ± 0.18266773553 (POSITIVE) | 2.82094043498 ± 0.305607046537 (POSITIVE) | 3.0816126712 ± 0.104526043229 (POSITIVE) | 3.03334252835 ± 0.0404790118879 (POSITIVE) | -0.0482701428458 ± 0.145004908114 (SIGN_UNRESOLVED) |
| M2048_N32_w16 | PRIMARY | 2.42747747896 ± 0.263691583628 (POSITIVE) | -0.0511317055151 ± 0.0355838800342 (SIGN_UNRESOLVED) | 2.57154447137 ± 0.00758564229272 (POSITIVE) | -0.0599419373819 ± 0.0175868052657 (NEGATIVE) | 2.08196130795 ± 0.0972480441154 (POSITIVE) | 2.14190324533 ± 0.0825089252515 (POSITIVE) |
| M512_N64_w16 | PRIMARY | -0.0080923373822 ± 0.0362779633194 (SIGN_UNRESOLVED) | -0.00785911695511 ± 0.0103025111404 (SIGN_UNRESOLVED) | -0.00909191536872 ± 0.0122700211675 (SIGN_UNRESOLVED) | 0.010510164083 ± 0.0226475346977 (SIGN_UNRESOLVED) | -0.0267386678301 ± 0.010250831765 (NEGATIVE) | -0.0372488319131 ± 0.0327958592234 (SIGN_UNRESOLVED) |

| Scope | Hypothesis | n | Reference MAE/RMSE | Alternative MAE/RMSE | Decision | Practical threshold |
| --- | --- | --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER | 4 | {'MAE': 0.6757287321436567, 'RMSE': 1.2463096397268876} | {'MAE': 0.12287184636945588, 'RMSE': 0.18122462198197856} | DIAGNOSTIC_ONLY | True |
| ALL_ELIGIBLE | H8_02_DESCRIPTOR_INCREMENT | 4 | {'MAE': 0.6980448195766558, 'RMSE': 1.2473909521569193} | {'MAE': 0.1600922782277595, 'RMSE': 0.20639522665028215} | DIAGNOSTIC_ONLY | True |
| PRIMARY | H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER | 4 | {'MAE': 0.6757287321436567, 'RMSE': 1.2463096397268876} | {'MAE': 0.12287184636945588, 'RMSE': 0.18122462198197856} | DIAGNOSTIC_ONLY | True |
| PRIMARY | H8_02_DESCRIPTOR_INCREMENT | 4 | {'MAE': 0.6980448195766558, 'RMSE': 1.2473909521569193} | {'MAE': 0.1600922782277595, 'RMSE': 0.20639522665028215} | DIAGNOSTIC_ONLY | True |

Every invocation value, sign band, OLS intercept/R²/residual, per-case prediction error, leave-one-out result and resource context is retained in results.json.

- All admitted cases retained; coefficient1/intercept0, no refit or favorable-outcome repeats.
- Mode0/1 share CUBIN/module/function/resource allocation; their descriptor issue paths and runtime branch differ.
- Runtime intervention is not pure tensormap construction latency, a causal share, or equivalence between different binaries.
- Three independent processes may share GPU UUIDs; small/sign-unresolved/negative effects remain in every analysis.
- NCU aggregates are separate descriptive evidence; its duration never enters these event fits.
- Fresh cohort extends warp count; coverage does not establish arbitrary shape/warp generalization.
- H2b/H2c remain UNVERIFIED; no production heuristic or compiler change.
