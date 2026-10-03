# STAGE D REPORT

OUTCOME-INFORMED EXPLORATORY ANALYSIS + HELD-OUT CONFIRMATORY EXTENSION PREREGISTRATION

Stage C is the development/hypothesis-generation set. No Stage D finding is retroactively preregistered or confirmed by Stage C. No Modal/GPU, compilation, new timing, production-code/heuristic change or PR update was performed.

Starting HEAD: `11310ee3b6c92e188c66999a5dacd1ece87d7079`. All pre-existing experiment files are checked byte-for-byte against that Git tree. Stage C remains PRIMARY=3, SECONDARY=15, EXCLUDE=2; PRIMARY n=3<5 remains insufficient for established generalization. H2a historical remains SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b/H2c remain UNVERIFIED.

G-D is a cross-harness descriptive residual, NOT an additive causal remainder.

fixed-harness differential; may include pre-loop LocalLoad, fixed work and compiler/register/scheduling interactions

All values below are mean ± sample SD over the three paired invocation differentials, in ns/additional CTA. Invocation values for G/g0/g1/D/G-D, all extracted sequences/multisets, resource fields, every model/LOOCV fold and all correlations are retained in results.json.

## Complete 18-case mechanism table

| Case | Class / origin | G | g0 | g1 | D | G-D | G/D categories | Taxonomy | Lane transition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | SECONDARY/NEW_GENERALIZATION | 0.00232538306 ± 0.0214938417 | 0.00685906257 ± 0.00603325197 | -0.0206938196 ± 0.0101602353 | -0.0275528821 ± 0.0159784724 | 0.0298782652 ± 0.0289230121 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 2->1 |
| M128_N128_w8 | SECONDARY/NEW_GENERALIZATION | -0.0182294787 ± 0.0157000017 | -0.0166714277 ± 0.0247874255 | -0.0234377001 ± 0.0209025885 | -0.00676627245 ± 0.0284854809 | -0.0114632063 ± 0.0161855839 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 2->1 |
| M128_N16_w4 | SECONDARY/NEW_GENERALIZATION | 0.0424106312 ± 0.00200100003 | 2.32976812e-05 ± 0.00294328171 | -0.0104864198 ± 0.00436755425 | -0.0105097175 ± 0.00384187169 | 0.0529203487 ± 0.00332082145 | POSITIVE/NEGATIVE | OPPOSED_CANONICAL_POSITIVE | 16->8 |
| M128_N16_w8 | SECONDARY/NEW_GENERALIZATION | 1.10758454 ± 0.00370249611 | 0.00141827849 ± 0.00348230856 | 0.581473333 ± 0.00511019415 | 0.580055054 ± 0.00439871675 | 0.527529488 ± 0.00577513213 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 16->8 |
| M128_N32_w4 | SECONDARY/NEW_GENERALIZATION | 0.0132068592 ± 0.0103496943 | 0.00179035115 ± 0.00722401919 | 0.00297610487 ± 0.00871918291 | 0.00118575372 ± 0.00511774185 | 0.0120211055 ± 0.005273092 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 8->4 |
| M128_N32_w8 | SECONDARY/NEW_GENERALIZATION | 0.63720713 ± 0.0140556519 | 0.00186004118 ± 0.00758531587 | -0.00118593778 ± 0.00584563725 | -0.00304597896 ± 0.00736040367 | 0.640253109 ± 0.00717159361 | POSITIVE/SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 8->4 |
| M128_N64_w8 | PRIMARY/NEW_GENERALIZATION | -0.0119047118 ± 0.0109253971 | 0.00427853376 ± 0.00539955024 | -0.00848681988 ± 0.00680105214 | -0.0127653536 ± 0.00697436981 | 0.000860641844 ± 0.0176170427 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 4->2 |
| M32_N128_w4 | SECONDARY/ANCHOR_EXISTING | 0.00458051848 ± 0.00748428329 | -0.00602219751 ± 0.00222484365 | -0.00823105698 ± 0.00347306723 | -0.00220885947 ± 0.00436750756 | 0.00678937795 ± 0.00948399789 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 2->1 |
| M32_N128_w8 | SECONDARY/NEW_GENERALIZATION | 0.854980466 ± 0.00591421509 | -0.00211596108 ± 0.00153670946 | 0.184360948 ± 0.00815590422 | 0.186476909 ± 0.00880456701 | 0.668503557 ± 0.00372935247 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 2->1 |
| M32_N32_w4 | SECONDARY/NEW_GENERALIZATION | 0.160853785 ± 0.00598794651 | -0.00476655075 ± 0.00133138505 | 0.226376425 ± 0.0124746354 | 0.231142976 ± 0.0111477806 | -0.0702891909 ± 0.00539297923 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 8->4 |
| M32_N64_w4 | SECONDARY/NEW_GENERALIZATION | 0.0657087076 ± 0.0049436423 | 0.00102305161 ± 0.0032043934 | -0.00318546188 ± 0.00571064106 | -0.0042085135 ± 0.00497235356 | 0.0699172211 ± 0.00420975764 | POSITIVE/SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 4->2 |
| M32_N64_w8 | PRIMARY/ANCHOR_EXISTING | 1.41469028 ± 0.0090380954 | 0.00460380804 ± 0.00435169117 | 1.04931642 ± 0.00350658926 | 1.04471261 ± 0.00598556047 | 0.36997767 ± 0.00958754455 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 4->2 |
| M64_N128_w8 | SECONDARY/NEW_GENERALIZATION | -0.0179733369 ± 0.0233514283 | -0.0004186599 ± 0.013071078 | -0.0167875831 ± 0.00678895916 | -0.0163689232 ± 0.0171644703 | -0.00160441362 ± 0.0234002907 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 2->1 |
| M64_N16_w4 | SECONDARY/NEW_GENERALIZATION | 0.115234305 ± 0.00946882754 | -0.00218559697 ± 0.00352679128 | 0.306082587 ± 0.00580979553 | 0.308268184 ± 0.00705912972 | -0.193033879 ± 0.00684673518 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 16->8 |
| M64_N32_w4 | SECONDARY/NEW_GENERALIZATION | 0.0738932017 ± 0.008919918 | 0.00462710843 ± 0.00431144264 | -0.0114862197 ± 0.00562566463 | -0.0161133282 ± 0.00820705758 | 0.0900065299 ± 0.0171210259 | POSITIVE/SIGN_UNRESOLVED | E_OTHER_MECHANISM_CANDIDATE | 8->4 |
| M64_N32_w8 | SECONDARY/NEW_GENERALIZATION | 1.0411551 ± 0.00393054554 | -0.00378999435 ± 0.00252890327 | 0.224423432 ± 0.00460297151 | 0.228213426 ± 0.00645317201 | 0.812941678 ± 0.00997126998 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 8->4 |
| M64_N64_w4 | SECONDARY/NEW_GENERALIZATION | -0.00276689943 ± 0.0114423493 | -0.00134866154 ± 0.00545032649 | -0.00837061569 ± 0.00503541972 | -0.00702195415 ± 0.0104632291 | 0.00425505472 ± 0.0208152775 | SIGN_UNRESOLVED/SIGN_UNRESOLVED | B_MECHANISM_NEUTRAL_UNRESOLVED | 4->2 |
| M64_N64_w8 | PRIMARY/NEW_GENERALIZATION | 1.14311292 ± 0.0124995436 | 0.00346444397 ± 0.00821343623 | 0.284389 ± 0.0123361135 | 0.280924556 ± 0.0152761297 | 0.862188369 ± 0.0175697648 | POSITIVE/POSITIVE | A_MECHANISM_ALIGNED_POSITIVE | 4->2 |

E/P below retains EXACT_SEQUENCE_EQUIVALENT versus PIPELINED_OPCODE_EQUIVALENT relative to the canonical body. The canonical equivalence column is canonical-to-single reproduction, not default-to-cand4 body equivalence. Instruction counts/multisets and sequence order classifications are separate features. Deltas mean default−cand4, never weighted instruction costs.

| Case | M/N/warps | Vec d/4; warpPart d/4 | Canonical equivalence d/4 | Repeated equivalence d/4 | Preload d/4 | Registers canonical/repeated d→4 | Blocks/SM canonical/repeated d→4 | Initial LL count delta canonical/repeated | LL bytes/thread canonical d/4; repeated d/4 | Canonical shfl/bar/ld/st/max/cvt/ldmatrix/selp deltas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | 128/128/4 | 8/4; 4/4 | E/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:36→31; repeated:41→32 | canonical:6→6; repeated:6→6 | -16/-16 | canonical:256/256; repeated:256/256 | 16/4/2/2/14/4/0/0 |
| M128_N128_w8 | 128/128/8 | 8/4; 8/8 | E/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:38→36; repeated:39→32 | canonical:6→6; repeated:6→6 | -8/-8 | canonical:128/128; repeated:128/128 | 20/4/2/2/18/4/0/0 |
| M128_N16_w4 | 128/16/4 | 8/4; 4/4 | P/P | P/P | EXACT/EXACT | canonical:31→21; repeated:30→21 | canonical:16→16; repeated:16→16 | -2/-2 | canonical:32/32; repeated:32/32 | 28/2/1/1/24/4/0/0 |
| M128_N16_w8 | 128/16/8 | 8/4; 8/8 | P/P | P/P | EXACT/EXACT | canonical:31→21; repeated:30→21 | canonical:8→8; repeated:8→8 | -1/-1 | canonical:16/16; repeated:16/16 | 32/2/1/1/28/4/0/0 |
| M128_N32_w4 | 128/32/4 | 8/4; 4/4 | P/P | P/P | EXACT/EXACT | canonical:30→22; repeated:29→22 | canonical:16→16; repeated:16→16 | -4/-4 | canonical:64/64; repeated:64/64 | 24/2/1/1/20/4/0/0 |
| M128_N32_w8 | 128/32/8 | 8/4; 8/8 | P/E | P/E | EXACT/EXACT | canonical:29→22; repeated:29→23 | canonical:8→8; repeated:8→8 | -2/-2 | canonical:32/32; repeated:32/32 | 28/0/0/0/24/4/0/0 |
| M128_N64_w8 | 128/64/8 | 8/4; 8/8 | E/E | E/E | EXACT/DIFFERENT_ENCODING | canonical:29→22; repeated:32→24 | canonical:8→8; repeated:8→8 | -4/-4 | canonical:64/64; repeated:64/64 | 24/4/2/2/34/4/0/0 |
| M32_N128_w4 | 32/128/4 | 8/4; 4/4 | E/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:32→25; repeated:32→24 | canonical:16→16; repeated:16→16 | -4/-4 | canonical:64/64; repeated:64/64 | 16/4/2/2/14/4/0/0 |
| M32_N128_w8 | 32/128/8 | 8/4; 8/8 | E/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:32→21; repeated:32→25 | canonical:8→8; repeated:8→8 | -2/-2 | canonical:32/32; repeated:32/32 | 20/4/2/2/18/4/0/0 |
| M32_N32_w4 | 32/32/4 | 8/4; 4/4 | P/P | P/P | EXACT/EXACT | canonical:29→20; repeated:29→17 | canonical:16→16; repeated:16→16 | -1/-1 | canonical:16/16; repeated:16/16 | 24/2/1/1/20/4/0/0 |
| M32_N64_w4 | 32/64/4 | 8/4; 4/4 | P/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:28→22; repeated:28→22 | canonical:16→16; repeated:16→16 | -2/-2 | canonical:32/32; repeated:32/32 | 20/0/0/0/22/4/0/0 |
| M32_N64_w8 | 32/64/8 | 8/4; 8/8 | E/E | E/E | EXACT/DIFFERENT_ENCODING | canonical:29→22; repeated:32→23 | canonical:8→8; repeated:8→8 | -1/-1 | canonical:16/16; repeated:16/16 | 24/4/2/2/22/4/0/0 |
| M64_N128_w8 | 64/128/8 | 8/4; 8/8 | E/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:32→24; repeated:32→25 | canonical:8→8; repeated:8→8 | -4/-4 | canonical:64/64; repeated:64/64 | 20/4/2/2/18/4/0/0 |
| M64_N16_w4 | 64/16/4 | 8/4; 4/4 | P/P | P/P | EXACT/EXACT | canonical:31→21; repeated:30→21 | canonical:16→16; repeated:16→16 | -1/-1 | canonical:16/16; repeated:16/16 | 28/2/1/1/24/4/0/0 |
| M64_N32_w4 | 64/32/4 | 8/4; 4/4 | P/P | P/P | EXACT/EXACT | canonical:29→22; repeated:29→17 | canonical:16→16; repeated:16→16 | -2/-2 | canonical:32/32; repeated:32/32 | 24/2/1/1/20/4/0/0 |
| M64_N32_w8 | 64/32/8 | 8/4; 8/8 | P/E | P/E | EXACT/EXACT | canonical:29→22; repeated:29→23 | canonical:8→8; repeated:8→8 | -1/-1 | canonical:16/16; repeated:16/16 | 28/0/0/0/24/4/0/0 |
| M64_N64_w4 | 64/64/4 | 8/4; 4/4 | P/E | P/E | EXACT/DIFFERENT_ENCODING | canonical:28→22; repeated:28→24 | canonical:16→16; repeated:16→16 | -4/-4 | canonical:64/64; repeated:64/64 | 20/0/0/0/30/4/0/0 |
| M64_N64_w8 | 64/64/8 | 8/4; 8/8 | E/E | E/E | EXACT/DIFFERENT_ENCODING | canonical:29→22; repeated:32→23 | canonical:8→8; repeated:8→8 | -2/-2 | canonical:32/32; repeated:32/32 | 24/4/2/2/26/4/0/0 |

All shared fields remain distinct: driver static SMEM, dynamic launch SMEM and cuobjdump-reported SHARED. Here driver static is 0 and raw cuobjdump SHARED is 1024; neither is substituted for the other or used to infer occupancy. All LocalLoad sequences and resource fields follow; complete body fingerprint sequences and multisets are archived in results.json.

| Case / harness | Lane reduction ratio | Register delta | Dynamic SMEM d/4 | Driver static SMEM d/4 | cuobjdump SHARED d/4 | LocalLoad opcode sequence d/4 | Vector widths d/4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4/canonical | 2.0 | 5 | 32776/32776 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] |
| M128_N128_w4/repeated | 2.0 | 9 | 32776/32776 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] |
| M128_N128_w8/canonical | 2.0 | 2 | 32776/32776 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4, 4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] |
| M128_N128_w8/repeated | 2.0 | 7 | 32776/32776 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4, 4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] |
| M128_N16_w4/canonical | 2.0 | 10 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M128_N16_w4/repeated | 2.0 | 9 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M128_N16_w8/canonical | 2.0 | 10 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M128_N16_w8/repeated | 2.0 | 9 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M128_N32_w4/canonical | 2.0 | 8 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M128_N32_w4/repeated | 2.0 | 7 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M128_N32_w8/canonical | 2.0 | 7 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M128_N32_w8/repeated | 2.0 | 6 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M128_N64_w8/canonical | 2.0 | 7 | 16392/16392 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2] |
| M128_N64_w8/repeated | 2.0 | 8 | 16392/16392 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M32_N128_w4/canonical | 2.0 | 7 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2] |
| M32_N128_w4/repeated | 2.0 | 8 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M32_N128_w8/canonical | 2.0 | 11 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4] / [2, 2, 2, 2] |
| M32_N128_w8/repeated | 2.0 | 7 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M32_N32_w4/canonical | 2.0 | 9 | 2056/2056 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M32_N32_w4/repeated | 2.0 | 12 | 2056/2056 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M32_N64_w4/canonical | 2.0 | 6 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4] / [2, 2, 2, 2] |
| M32_N64_w4/repeated | 2.0 | 6 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M32_N64_w8/canonical | 2.0 | 7 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32"] | [4] / [2, 2] |
| M32_N64_w8/repeated | 2.0 | 9 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M64_N128_w8/canonical | 2.0 | 8 | 16392/16392 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2] |
| M64_N128_w8/repeated | 2.0 | 7 | 16392/16392 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M64_N16_w4/canonical | 2.0 | 10 | 2056/2056 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M64_N16_w4/repeated | 2.0 | 9 | 2056/2056 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M64_N32_w4/canonical | 2.0 | 7 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M64_N32_w4/repeated | 2.0 | 12 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |
| M64_N32_w8/canonical | 2.0 | 7 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M64_N32_w8/repeated | 2.0 | 6 | 4104/4104 | 0/0 | 1024/1024 | ["ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16"] | [4] / [4, 4] |
| M64_N64_w4/canonical | 2.0 | 6 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4, 4, 4] / [2, 2, 2, 2, 2, 2, 2, 2] |
| M64_N64_w4/repeated | 2.0 | 4 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4, 4, 4] / [4, 4, 4, 4, 4, 4, 4, 4] |
| M64_N64_w8/canonical | 2.0 | 7 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"] | [4, 4] / [2, 2, 2, 2] |
| M64_N64_w8/repeated | 2.0 | 9 | 8200/8200 | 0/0 | 1024/1024 | ["ld.shared.v4.b32", "ld.shared.v4.b32"] / ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"] | [4, 4] / [4, 4, 4, 4] |

## Only the registered exploratory model family

Only A/B/C, unweighted OLS on case means; all 18 cases retained; PRIMARY/SECONDARY separate; singular fit/fold UNDEFINED, never partially aggregate failed LOOCV; no p-values, significance, winner threshold or causal inference

A: G~1+D; B: G~1+D+g0; C: G~1+D+g0+I(warps==8). The QR rank tolerance 1e-12 is numerical only. All fits use case means, with all invocation values retained separately; no p-values or significance claims.

| Scope | Model | n | OLS coefficients | In-sample R² | LOOCV MAE | LOOCV RMSE | Defined/undefined folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | A | 18 | {"D": 1.43234225450178, "intercept": 0.14893336518099015} | 0.62102066 | 0.28920348 | 0.357357235 | 18/0 |
| ALL_ELIGIBLE | B | 18 | {"D": 1.4079214075667394, "g0": 7.152669582225426, "intercept": 0.15559936860663273} | 0.626569425 | 0.301458624 | 0.36889089 | 18/0 |
| ALL_ELIGIBLE | C | 18 | {"D": 1.0999810571362003, "g0": 13.161075863171975, "intercept": -0.004980287943348076, "warp8": 0.4203242852172698} | 0.779641321 | 0.2510526 | 0.336840286 | 18/0 |
| PRIMARY | A | 3 | {"D": 1.1495351439899855, "intercept": 0.34556873483765427} | 0.686195808 | 1.51387935 | 1.74591295 | 3/0 |
| PRIMARY | B | 3 | {"D": 1.6071256841253092, "g0": -838.9996386976379, "intercept": 3.598299094775085} | 1 | UNDEFINED | UNDEFINED | 0/3 |
| PRIMARY | C | 3 | UNDEFINED | UNDEFINED | UNDEFINED | UNDEFINED | 0/3 |
| SECONDARY | A | 15 | {"D": 1.648275847952593, "intercept": 0.11360704486548989} | 0.494315154 | 0.235797016 | 0.321098562 | 15/0 |
| SECONDARY | B | 15 | {"D": 1.6552654267361786, "g0": 6.804435467864977, "intercept": 0.12187991809381883} | 0.502343568 | 0.23794842 | 0.323230995 | 15/0 |
| SECONDARY | C | 15 | {"D": 1.2369471500589966, "g0": 20.017877368860542, "intercept": -0.012178234808927439, "warp8": 0.47907300980686424} | 0.783907949 | 0.209095472 | 0.277283686 | 15/0 |

PRIMARY B is a saturated three-point fit; R²=1 is not validation and all its LOOCV folds are undefined. PRIMARY C is insufficient/singular. On all 18 development cases B has slightly higher LOOCV error than A, while C has lower error than A/B. These observations only motivate hypotheses; no model winner or causal warp claim.

## Groupwise descriptive audit

EXACT means all canonical-single and repeated candidate body reproductions are E; PIPELINED means at least one P. Preload EXACT means both candidates EXACT; otherwise DIFFERENT_ENCODING is present. Neither aggregation upgrades a Stage C class.

M:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 7 | False | 0.253228622 | 0.0743715147 | 0.178857107 | 1/2 | 5 |
| 32 | 5 | False | 0.500162752 | 0.291183025 | 0.208979727 | 3/3 | 2 |
| 64 | 6 | False | 0.392109216 | 0.129650327 | 0.26245889 | 3/3 | 3 |

N:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 5 | False | 0.16513671 | 0.0267159943 | 0.138420716 | 1/1 | 4 |
| 16 | 3 | False | 0.42174316 | 0.292604507 | 0.129138653 | 2/3 | 0 |
| 32 | 5 | False | 0.385263216 | 0.0882765697 | 0.296986646 | 2/2 | 3 |
| 64 | 5 | False | 0.521768061 | 0.26032827 | 0.261439791 | 2/2 | 3 |

equivalence_group:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EXACT | 3 | False | 0.848632832 | 0.437623939 | 0.411008894 | 2/2 | 1 |
| PIPELINED | 15 | False | 0.272011395 | 0.0961030582 | 0.175908336 | 5/6 | 9 |

lane_transition:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16->8 | 3 | False | 0.42174316 | 0.292604507 | 0.129138653 | 2/3 | 0 |
| 2->1 | 5 | False | 0.16513671 | 0.0267159943 | 0.138420716 | 1/1 | 4 |
| 4->2 | 5 | False | 0.521768061 | 0.26032827 | 0.261439791 | 2/2 | 3 |
| 8->4 | 5 | False | 0.385263216 | 0.0882765697 | 0.296986646 | 2/2 | 3 |

num_warps:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 9 | False | 0.052827388 | 0.0525535176 | 0.000273870389 | 2/3 | 6 |
| 8 | 9 | False | 0.683402547 | 0.253492892 | 0.429909655 | 5/5 | 4 |

preload_group:

| Group | n | small-n | Mean G | Mean D | Mean G-D | Resolved sign matches/denominator | Unresolved pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DIFFERENT_ENCODING | 10 | False | 0.343452386 | 0.143522132 | 0.199930254 | 3/3 | 7 |
| EXACT | 8 | False | 0.398943195 | 0.164899546 | 0.234043649 | 4/5 | 3 |

## Exploratory correlations

D+g0 is a descriptive constructed score; no causal decomposition

| Scope | Score versus G | Pearson | Spearman |
| --- | --- | --- | --- |
| ALL_ELIGIBLE | D | 0.78804864 | 0.787409701 |
| ALL_ELIGIBLE | D_plus_g0 | 0.789313419 | 0.851393189 |
| ALL_ELIGIBLE | g0 | 0.213191598 | 0.184726522 |
| PRIMARY | D | 0.828369367 | 1 |
| PRIMARY | D_plus_g0 | 0.827850261 | 1 |
| PRIMARY | g0 | -0.0625406409 | 0.5 |
| SECONDARY | D | 0.703075497 | 0.714285714 |
| SECONDARY | D_plus_g0 | 0.705507634 | 0.821428571 |
| SECONDARY | g0 | 0.066170491 | 0.0964285714 |

## Focused opposition/unresolved/neutral audit

resolved_opposition: All three have exact preloads, the same lane transition and the same canonical/repeated register deltas. The opposed case has 32B/thread preload versus 16B in both neighbors, larger per-thread M work than the warp8 neighbor, and different absolute/family shuffle structure. These distinguish execution contexts but do not identify a cause; g0 does not capture all body/context interactions.

```json
{
  "canonical_family_delta_target_reference": [
    {
      "bar.sync": 2,
      "cvt.*": 4,
      "ld.shared*": 1,
      "ldmatrix*": 0,
      "max.*": 24,
      "selp.*": 0,
      "shfl.*": 28,
      "st.shared*": 1
    },
    {
      "bar.sync": 2,
      "cvt.*": 4,
      "ld.shared*": 1,
      "ldmatrix*": 0,
      "max.*": 24,
      "selp.*": 0,
      "shfl.*": 28,
      "st.shared*": 1
    }
  ],
  "canonical_preload_payload_default_target_reference": [
    32,
    16
  ],
  "metric_mean_target_minus_reference": {
    "D": -0.31877790109189963,
    "G": -0.07282367386367339,
    "g0": 0.002208894654488998,
    "g1": -0.3165690064374106,
    "residual": 0.2459542272282262
  },
  "reference": "M64_N16_w4",
  "register_deltas_target_reference": {
    "canonical": [
      10,
      10
    ],
    "repeated": [
      9,
      9
    ]
  },
  "target": "M128_N16_w4",
  "warps_target_reference": [
    4,
    4
  ]
}
```

```json
{
  "canonical_family_delta_target_reference": [
    {
      "bar.sync": 2,
      "cvt.*": 4,
      "ld.shared*": 1,
      "ldmatrix*": 0,
      "max.*": 24,
      "selp.*": 0,
      "shfl.*": 28,
      "st.shared*": 1
    },
    {
      "bar.sync": 2,
      "cvt.*": 4,
      "ld.shared*": 1,
      "ldmatrix*": 0,
      "max.*": 28,
      "selp.*": 0,
      "shfl.*": 32,
      "st.shared*": 1
    }
  ],
  "canonical_preload_payload_default_target_reference": [
    32,
    16
  ],
  "metric_mean_target_minus_reference": {
    "D": -0.5905647716038702,
    "G": -1.065173911402304,
    "g0": -0.001394980808115657,
    "g1": -0.591959752411986,
    "residual": -0.47460913979843383
  },
  "reference": "M128_N16_w8",
  "register_deltas_target_reference": {
    "canonical": [
      10,
      10
    ],
    "repeated": [
      9,
      9
    ]
  },
  "target": "M128_N16_w4",
  "warps_target_reference": [
    4,
    8
  ]
}
```

canonical_positive_D_unresolved: The three cases span both warp regimes and both EXACT/DIFFERENT_ENCODING preload groups. Their small fixed-harness differentials coexist with positive G and unresolved D. Initial load width/count, body instruction mix, register allocation and scheduling interactions are candidate explanations; encoding mismatch and warp count alone do not distinguish all three.

```json
{
  "canonical_family_delta_target_reference": [
    {
      "bar.sync": 0,
      "cvt.*": 4,
      "ld.shared*": 0,
      "ldmatrix*": 0,
      "max.*": 24,
      "selp.*": 0,
      "shfl.*": 28,
      "st.shared*": 0
    },
    {
      "bar.sync": 2,
      "cvt.*": 4,
      "ld.shared*": 1,
      "ldmatrix*": 0,
      "max.*": 20,
      "selp.*": 0,
      "shfl.*": 24,
      "st.shared*": 1
    }
  ],
  "canonical_preload_payload_default_target_reference": [
    32,
    32
  ],
  "metric_mean_target_minus_reference": {
    "D": 0.013067349192126711,
    "G": 0.56331392832882,
    "g0": -0.0027670672525780633,
    "g1": 0.010300281939548647,
    "residual": 0.5502465791366933
  },
  "reference": "M64_N32_w4",
  "register_deltas_target_reference": {
    "canonical": [
      7,
      7
    ],
    "repeated": [
      6,
      12
    ]
  },
  "target": "M128_N32_w8",
  "warps_target_reference": [
    8,
    4
  ]
}
```

```json
{
  "canonical_family_delta_target_reference": [
    {
      "bar.sync": 0,
      "cvt.*": 4,
      "ld.shared*": 0,
      "ldmatrix*": 0,
      "max.*": 22,
      "selp.*": 0,
      "shfl.*": 20,
      "st.shared*": 0
    },
    {
      "bar.sync": 4,
      "cvt.*": 4,
      "ld.shared*": 2,
      "ldmatrix*": 0,
      "max.*": 22,
      "selp.*": 0,
      "shfl.*": 24,
      "st.shared*": 2
    }
  ],
  "canonical_preload_payload_default_target_reference": [
    32,
    16
  ],
  "metric_mean_target_minus_reference": {
    "D": -1.0489211273947843,
    "G": -1.348981576094437,
    "g0": -0.003580756432105166,
    "g1": -1.0525018838268894,
    "residual": -0.3000604486996528
  },
  "reference": "M32_N64_w8",
  "register_deltas_target_reference": {
    "canonical": [
      6,
      7
    ],
    "repeated": [
      6,
      9
    ]
  },
  "target": "M32_N64_w4",
  "warps_target_reference": [
    4,
    8
  ]
}
```

M128_N16_w4: G=0.0424106312 ± 0.00200100003; g0=2.32976812e-05 ± 0.00294328171; D=-0.0105097175 ± 0.00384187169; residual=0.0529203487 ± 0.00332082145.

canonical LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 2, "cvt.*": 4, "ld.shared*": 1, "ldmatrix*": 0, "max.*": 32, "selp.*": 0, "shfl.*": 20, "st.shared*": 1}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 56, "selp.*": 0, "shfl.*": 48, "st.shared*": 2}}`.

repeated LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 2, "cvt.*": 4, "ld.shared*": 1, "ldmatrix*": 0, "max.*": 32, "selp.*": 0, "shfl.*": 20, "st.shared*": 1}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 56, "selp.*": 0, "shfl.*": 48, "st.shared*": 2}}`.

M128_N32_w8: G=0.63720713 ± 0.0140556519; g0=0.00186004118 ± 0.00758531587; D=-0.00304597896 ± 0.00736040367; residual=0.640253109 ± 0.00717159361.

canonical LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 4, "cvt.*": 4, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 32, "selp.*": 0, "shfl.*": 20, "st.shared*": 2}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 56, "selp.*": 0, "shfl.*": 48, "st.shared*": 2}}`.

repeated LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 4, "cvt.*": 4, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 32, "selp.*": 0, "shfl.*": 20, "st.shared*": 2}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 56, "selp.*": 0, "shfl.*": 48, "st.shared*": 2}}`.

M32_N64_w4: G=0.0657087076 ± 0.0049436423; g0=0.00102305161 ± 0.0032043934; D=-0.0042085135 ± 0.00497235356; residual=0.0699172211 ± 0.00420975764.

canonical LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"]}`; family counts d/4: `{"4": {"bar.sync": 4, "cvt.*": 4, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 18, "selp.*": 0, "shfl.*": 12, "st.shared*": 2}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 40, "selp.*": 0, "shfl.*": 32, "st.shared*": 2}}`.

repeated LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 4, "cvt.*": 4, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 18, "selp.*": 0, "shfl.*": 12, "st.shared*": 2}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 40, "selp.*": 0, "shfl.*": 32, "st.shared*": 2}}`.

M64_N32_w4: G=0.0738932017 ± 0.008919918; g0=0.00462710843 ± 0.00431144264; D=-0.0161133282 ± 0.00820705758; residual=0.0900065299 ± 0.0171210259.

canonical LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 2, "cvt.*": 4, "ld.shared*": 1, "ldmatrix*": 0, "max.*": 28, "selp.*": 0, "shfl.*": 16, "st.shared*": 1}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 48, "selp.*": 0, "shfl.*": 40, "st.shared*": 2}}`.

repeated LocalLoad d/4: `{"default": ["ld.shared.v4.b32", "ld.shared.v4.b32"], "4": ["ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16", "ld.shared.v4.b16"]}`; family counts d/4: `{"4": {"bar.sync": 2, "cvt.*": 4, "ld.shared*": 1, "ldmatrix*": 0, "max.*": 28, "selp.*": 0, "shfl.*": 16, "st.shared*": 1}, "default": {"bar.sync": 4, "cvt.*": 8, "ld.shared*": 2, "ldmatrix*": 0, "max.*": 48, "selp.*": 0, "shfl.*": 40, "st.shared*": 2}}`.

Retain all seven neutral cases and their signed continuous means/SDs. They may locate expression boundaries, but SIGN_UNRESOLVED is not practical equivalence and cannot justify declaring the optimization irrelevant.

| Neutral case | G | g0 | D | G-D | Lane | Preload group |
| --- | --- | --- | --- | --- | --- | --- |
| M128_N128_w4 | 0.00232538306 ± 0.0214938417 | 0.00685906257 ± 0.00603325197 | -0.0275528821 ± 0.0159784724 | 0.0298782652 ± 0.0289230121 | 2->1 | DIFFERENT_ENCODING |
| M128_N128_w8 | -0.0182294787 ± 0.0157000017 | -0.0166714277 ± 0.0247874255 | -0.00676627245 ± 0.0284854809 | -0.0114632063 ± 0.0161855839 | 2->1 | DIFFERENT_ENCODING |
| M128_N32_w4 | 0.0132068592 ± 0.0103496943 | 0.00179035115 ± 0.00722401919 | 0.00118575372 ± 0.00511774185 | 0.0120211055 ± 0.005273092 | 8->4 | EXACT |
| M128_N64_w8 | -0.0119047118 ± 0.0109253971 | 0.00427853376 ± 0.00539955024 | -0.0127653536 ± 0.00697436981 | 0.000860641844 ± 0.0176170427 | 4->2 | DIFFERENT_ENCODING |
| M32_N128_w4 | 0.00458051848 ± 0.00748428329 | -0.00602219751 ± 0.00222484365 | -0.00220885947 ± 0.00436750756 | 0.00678937795 ± 0.00948399789 | 2->1 | DIFFERENT_ENCODING |
| M64_N128_w8 | -0.0179733369 ± 0.0233514283 | -0.0004186599 ± 0.013071078 | -0.0163689232 ± 0.0171644703 | -0.00160441362 ± 0.0234002907 | 2->1 | DIFFERENT_ENCODING |
| M64_N64_w4 | -0.00276689943 ± 0.0114423493 | -0.00134866154 ± 0.00545032649 | -0.00702195415 ± 0.0104632291 | 0.00425505472 ± 0.0208152775 | 4->2 | DIFFERENT_ENCODING |

## Outcome-informed hypotheses, for new held-out data only

### H5_01_EXACT_BODY_DIRECTIONAL_TRACKING

OUTCOME_INFORMED_HYPOTHESIS

Within the newly gated exact-sequence scope, G and D have a positive continuous rank association.

Observed development motivation: `{"PRIMARY_Spearman_G_D": 1.0, "case_ids": ["M128_N64_w8", "M32_N64_w8", "M64_N64_w8"], "limitation": "Development n=3, one anchor and two new shapes; LOO/new-only undefined. This is not established generalization."}`.

Mechanistic rationale: Exact body reproduction preserves the frozen instruction sequence projection; variation in a body advantage may track canonical marginal cost within that equivalence scope, subject to context interactions.

Variables: G, D, exact_sequence_gate, coverage.

Future support: Held-out PRIMARY n>=5 with nonconstant G/D and positive Spearman(G,D) supports the directional prediction; report full sign agreement, effect sizes and coverage, without automatic established-generalization status.

Future falsification: Held-out PRIMARY n>=5 with nonconstant G/D and negative Spearman(G,D) contradicts the directional prediction.

Inconclusive: PRIMARY n<5, constant/undefined correlation, or rho=0; no rescue by adding development data or upgrading PIPELINED cases.

Scope: Every Phase 5 PRIMARY case admitted by the future frozen-equivalence artifact gate, across the full 24-source domain; no N64/w8 structural filter.

### H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT

OUTCOME_INFORMED_HYPOTHESIS

Adding the fixed-harness differential alone does not improve prospective prediction over the body-only linear model.

Observed development motivation: `{"LOOCV_A_MAE": 0.2892034801332597, "LOOCV_A_RMSE": 0.35735723462498914, "LOOCV_B_MAE": 0.3014586237507142, "LOOCV_B_RMSE": 0.36889089020064864, "focused_cases": ["M128_N16_w4", "M128_N32_w8", "M32_N64_w4", "M64_N32_w4"]}`.

Mechanistic rationale: Small R=0 differentials coexist with opposed/unresolved-body canonical outcomes. Fixed work need not capture register/live-range/scheduling interactions between the repeated and canonical execution contexts.

Variables: G, D, g0, locked_model_A_predictions, locked_model_B_predictions.

Future support: Using the locked development A/B coefficients on all held-out timing-eligible cases (n>=5), MAE_B>=MAE_A and RMSE_B>=RMSE_A support the specified non-improvement prediction.

Future falsification: On that same held-out set, MAE_B<MAE_A and RMSE_B<RMSE_A contradict the specified non-improvement prediction.

Inconclusive: Mixed MAE/RMSE ordering, fewer than five eligible cases, invalid protocol or undefined predictions; no refitting or performance-driven exclusions.

Scope: All Phase 5 timing-eligible PRIMARY/SECONDARY cases; separately display strata and coverage. A/B are frozen descriptive predictors, not causal cost components.

### H5_03_WARP_REGIME_CONTEXT_PREDICTION

OUTCOME_INFORMED_HYPOTHESIS

The registered warp-regime covariate carries additional predictive context beyond D and g0 on held-out extrapolation cases.

Observed development motivation: `{"LOOCV_B_MAE": 0.3014586237507142, "LOOCV_B_RMSE": 0.36889089020064864, "LOOCV_C_MAE": 0.25105259967684806, "LOOCV_C_RMSE": 0.3368402855122466, "Model_C_warp8_coefficient": 0.4203242852172698}`.

Mechanistic rationale: Warp count changes per-thread work and inter-warp execution context. The covariate may proxy those changes or other correlated codegen features; it is not an identified causal warp effect or an always-warp8 rule.

Variables: G, D, g0, I(num_warps==8), locked_model_B_predictions, locked_model_C_predictions.

Future support: Using locked development B/C coefficients on all held-out timing-eligible cases with at least three cases in each warp regime, MAE_C<MAE_B and RMSE_C<RMSE_B support the specified predictive extension.

Future falsification: Under that coverage, MAE_C>=MAE_B and RMSE_C>=RMSE_B contradict the specified predictive extension.

Inconclusive: Mixed MAE/RMSE ordering, fewer than three cases in either warp regime or undefined predictions; no new model terms or refitting.

Scope: Entire Phase 5 timing-eligible cohort, across both warp regimes and full new-M domain. Does not select structural membership or establish production heuristics.

## Phase 5 preregistration and STOP

Full 24-source transition/layout/inclusion/reason table: ../phase5/preregistration/cohort_summary.md; future sampling/analysis/hypothesis locks: ../phase5/preregistration/protocol.json. No performance-dependent membership, expected PRIMARY filtering or N64/w8-only domain. Layouts are source-rule predictions, actual artifact/resource status pending.

The frozen Stage A validator allows new paths only under phase4/. Phase 5 code therefore lives in phase4/phase5/ and outputs in phase4/results/phase5/preregistration/. This preserves every Stage A-C byte, including the validator SHA, and retains a distinct Phase 5 statistical role.

Eight validators must pass: Phase 3 self-test/evidence, Stage A, Stage B, Stage C raw/analysis, Stage D residual analysis, Phase 5 preregistration. validation.json includes independent full recomputation, corruption probes, and a synthetic negative D/G, useless-g0, worse-C-LOOCV dataset accepted by fidelity checks.

NO Phase 5 timing observed. NO Phase 5 artifact gate executed. No production heuristic, Coalesce.cpp modification or PR #11991 update. STOP.

## Full held-out 24-transition pool

# Phase 5 — Held-Out Validation Preregistration

held-out confirmatory extension for OUTCOME_INFORMED_HYPOTHESIS; Stage C is development/hypothesis-generation data and cannot confirm these hypotheses

24 source transitions; 19 structurally included; 5 excluded. All layout/resource fields are source-rule predictions. No Phase 5 artifact gate or timing has been executed.

Each layout cell is sizePerThread / threadsPerWarp / warpsPerCTA, with order [2,1,0]. Illegal candidates have no predicted layout.

| Case | M | N | Warps | Default layout | Cand4 layout | Include/exclude | Reasons | Resource/artifact state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M16_N128_w4 | 16 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N128_w8 | 16 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N16_w4 | 16 | 16 | 4 | [1, 1, 2] / [1, 4, 8] / [1, 4, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N16_w8 | 16 | 16 | 8 | [1, 1, 1] / [1, 2, 16] / [1, 8, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N32_w4 | 16 | 32 | 4 | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | EXCLUDE | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | STRUCTURALLY_EXCLUDED |
| M16_N32_w8 | 16 | 32 | 8 | [1, 1, 2] / [1, 2, 16] / [1, 8, 1] | ILLEGAL | EXCLUDE | CAND4_ILLEGAL | STRUCTURALLY_EXCLUDED |
| M16_N64_w4 | 16 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M16_N64_w8 | 16 | 64 | 8 | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | EXCLUDE | IDENTICAL_LAYOUT, LANE_PART_NOT_REDUCED | STRUCTURALLY_EXCLUDED |
| M256_N128_w4 | 256 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N128_w8 | 256 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N16_w4 | 256 | 16 | 4 | [1, 1, 8] / [1, 16, 2] / [1, 4, 1] | [1, 1, 4] / [1, 8, 4] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N16_w8 | 256 | 16 | 8 | [1, 1, 8] / [1, 16, 2] / [1, 8, 1] | [1, 1, 4] / [1, 8, 4] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N32_w4 | 256 | 32 | 4 | [1, 1, 8] / [1, 8, 4] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N32_w8 | 256 | 32 | 8 | [1, 1, 8] / [1, 8, 4] / [1, 8, 1] | [1, 1, 4] / [1, 4, 8] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N64_w4 | 256 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M256_N64_w8 | 256 | 64 | 8 | [1, 1, 8] / [1, 4, 8] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N128_w4 | 512 | 128 | 4 | [1, 1, 8] / [1, 2, 16] / [1, 4, 1] | [1, 1, 4] / [1, 1, 32] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N128_w8 | 512 | 128 | 8 | [1, 1, 8] / [1, 2, 16] / [1, 8, 1] | [1, 1, 4] / [1, 1, 32] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N16_w4 | 512 | 16 | 4 | [1, 1, 8] / [1, 16, 2] / [1, 4, 1] | [1, 1, 4] / [1, 8, 4] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N16_w8 | 512 | 16 | 8 | [1, 1, 8] / [1, 16, 2] / [1, 8, 1] | [1, 1, 4] / [1, 8, 4] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N32_w4 | 512 | 32 | 4 | [1, 1, 8] / [1, 8, 4] / [1, 4, 1] | [1, 1, 4] / [1, 4, 8] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N32_w8 | 512 | 32 | 8 | [1, 1, 8] / [1, 8, 4] / [1, 8, 1] | [1, 1, 4] / [1, 4, 8] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N64_w4 | 512 | 64 | 4 | [1, 1, 8] / [1, 4, 8] / [1, 4, 1] | [1, 1, 4] / [1, 2, 16] / [1, 4, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |
| M512_N64_w8 | 512 | 64 | 8 | [1, 1, 8] / [1, 4, 8] / [1, 8, 1] | [1, 1, 4] / [1, 2, 16] / [1, 8, 1] | INCLUDE | PASSED_STAGE_A_STRUCTURAL_RULE | STRUCTURAL_OR_RESOURCE_PENDING |

Coverage: `{"M": {"16": 3, "256": 8, "512": 8}, "N": {"128": 6, "16": 4, "32": 4, "64": 5}, "lane_transition": {"16->8": 4, "2->1": 6, "4->2": 5, "8->4": 4}, "num_warps": {"4": 10, "8": 9}}`.

M16 is one power-of-two below the development M32/64/128 range; M256/512 are above it. Deliberate extrapolation, never prune for expected PRIMARY membership or performance.

The field compiled_num_warps is required by the reused Stage A selector but is explicitly a predicted source attribute here, not a claim of compilation. Actual Stage B artifacts must confirm it.

M512/N128 retains B_DESC=65536: 8GiB BF16 input per case and 128KiB logical shared tile. Actual compiled shared allocation, register/spill, occupancy and allocation safety remain pending. No descriptor/grid/resource policy is adjusted.

PRIMARY and SECONDARY will be assigned only by the future artifact gate; no predicted PRIMARY count or N64/w8 filter.

Full future sampling, analysis, locked prediction coefficients and hypothesis support/falsification observations are in protocol.json. Stage C cannot confirm Stage D-generated hypotheses.

NO Phase 5 timing observed. NO Phase 5 artifact gate executed. STOP until a separate artifact-gate authorization.
