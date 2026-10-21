module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: f32, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v9: i32 {tt.divisibility = 16 : i32}, %v10: i32 {tt.divisibility = 16 : i32}, %v11: i32 {tt.divisibility = 16 : i32}, %v12: i32 {tt.divisibility = 16 : i32}, %v13: i32 {tt.divisibility = 16 : i32}, %v14: i32 {tt.divisibility = 16 : i32}, %v15: i32 {tt.divisibility = 16 : i32}, %v16: i32 {tt.divisibility = 16 : i32}, %v17: i32 {tt.divisibility = 16 : i32}, %v18: i32 {tt.divisibility = 16 : i32}, %v19: i32 {tt.divisibility = 16 : i32}, %v20: i32 {tt.divisibility = 16 : i32}, %v21: i32 {tt.divisibility = 16 : i32}, %v22: i32 {tt.divisibility = 16 : i32}, %v23: i32 {tt.divisibility = 16 : i32}, %v24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v25 = arith.constant 1 : i32
%v26 = arith.constant dense<128> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v27 = arith.constant 32 : i32
%v28 = arith.constant 0xFF800000 : f32
%v29 = arith.constant 0.000000e+00 : f32
%v30 = arith.constant dense<0> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v31 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v32 = arith.constant 128 : i64
%v33 = arith.constant dense<0.000000e+00> : tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v34 = arith.constant 64 : i32
%v35 = arith.constant 31 : i32
%v36 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v37 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v38 = tt.get_program_id x : i32
%v39 = tt.get_program_id y : i32
%v40 = tt.get_program_id z : i32
%v41 = arith.extsi %v38 : i32 to i64
%v42 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v43 = arith.cmpi slt, %v42, %v26 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v44 = tt.addptr %v4, %v41 : !tt.ptr<i32>, i64
%v45 = tt.load %v44 : !tt.ptr<i32>
%v46 = tt.addptr %v44, %v25 : !tt.ptr<i32>, i32
%v47 = tt.load %v46 : !tt.ptr<i32>
%v48 = arith.subi %v47, %v45 : i32
%v49 = tt.addptr %v8, %v41 : !tt.ptr<i32>, i64
%v50 = tt.load %v49 : !tt.ptr<i32>
%v51 = arith.extsi %v9 : i32 to i64
%v52 = arith.muli %v41, %v51 : i64
%v53 = arith.muli %v39, %v10 : i32
%v54 = arith.extsi %v53 : i32 to i64
%v55 = arith.addi %v52, %v54 : i64
%v56 = arith.extsi %v42 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v57 = tt.splat %v55 : i64 -> tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v58 = arith.addi %v57, %v56 : tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v59 = arith.subi %v50, %v25 : i32
%v60 = arith.addi %v48, %v59 : i32
%v61 = arith.divsi %v60, %v50 : i32
%v62 = arith.addi %v61, %v35 : i32
%v63 = arith.divsi %v62, %v27 : i32
%v64 = arith.muli %v63, %v27 : i32
%v65 = arith.muli %v64, %v40 : i32
%v66 = arith.addi %v65, %v64 : i32
%v67 = arith.minsi %v66, %v48 : i32
%v68 = arith.cmpi sgt, %v67, %v65 : i32
scf.if %v68 {
%v69 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v70 = tt.addptr %v69, %v58 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = tt.load %v70, %v43, %v33 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.splat %v67 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.addptr %v5, %v45 : !tt.ptr<i32>, i32
%v75 = tt.splat %v74 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v76 = tt.splat %v11 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v77 = arith.muli %v39, %v12 : i32
%v78 = tt.splat %v77 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v79 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v80 = tt.expand_dims %v79 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v81 = ttg.convert_layout %v80 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v82 = tt.broadcast %v81 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v83 = tt.splat %v67 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v84 = ttg.convert_layout %v43 : tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v85 = tt.expand_dims %v84 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v86 = ttg.convert_layout %v85 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v87 = tt.broadcast %v86 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v88 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v89 = ttg.convert_layout %v71 : tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v90 = tt.expand_dims %v89 {axis = 0 : i32} : tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v91 = ttg.convert_layout %v90 : tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v92 = tt.broadcast %v91 : tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v93 = tt.splat %v3 : f32 -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v94 = tt.splat %v13 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v95 = arith.muli %v39, %v14 : i32
%v96 = tt.splat %v95 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v97 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v98:3 = scf.for %v99 = %v65 to %v67 step %v34 iter_args(%v100 = %v28, %v101 = %v29, %v102 = %v36) -> (f32, f32, tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) : i32 {
%v103 = tt.splat %v99 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v104 = arith.addi %v103, %v72 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v105 = arith.cmpi slt, %v104, %v73 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v106 = tt.addptr %v75, %v104 : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v107 = tt.load %v106, %v105, %v30 : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v108 = ttg.convert_layout %v107 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v109 = tt.expand_dims %v108 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v110 = ttg.convert_layout %v109 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v111 = arith.muli %v110, %v76 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v112 = arith.addi %v111, %v78 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = tt.broadcast %v112 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v114 = ttg.convert_layout %v113 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v115 = arith.addi %v114, %v82 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v116 = ttg.convert_layout %v104 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v117 = tt.expand_dims %v116 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v118 = ttg.convert_layout %v117 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = arith.cmpi slt, %v118, %v83 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v120 = tt.broadcast %v119 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v121 = ttg.convert_layout %v120 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v122 = arith.andi %v121, %v87 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v123 = tt.addptr %v88, %v115 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v124 = ttg.convert_layout %v123 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v125 = ttg.convert_layout %v122 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = tt.load %v124, %v125, %v37 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = ttg.convert_layout %v126 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v128 = arith.mulf %v92, %v127 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v129 = arith.extf %v128 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> to tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v130 = "tt.reduce"(%v129) <{axis = 1 : i32}> ({
^bb0(%v131: f32, %v132: f32):
%v133 = arith.addf %v131, %v132 : f32
tt.reduce.return %v133 : f32
}) : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>>
%v134 = ttg.convert_layout %v130 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v135 = arith.mulf %v134, %v93 : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v136 = arith.select %v105, %v135, %v31 : tensor<64xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v137 = arith.muli %v110, %v94 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v138 = arith.addi %v137, %v96 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v139 = tt.broadcast %v138 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v140 = ttg.convert_layout %v139 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v141 = arith.addi %v140, %v82 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v142 = tt.addptr %v97, %v141 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v143 = ttg.convert_layout %v142 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v144 = tt.load %v143, %v125, %v37 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v145 = ttg.convert_layout %v144 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v146 = "tt.reduce"(%v136) <{axis = 0 : i32}> ({
^bb0(%v147: f32, %v148: f32):
%v149 = arith.maxnumf %v147, %v148 : f32
tt.reduce.return %v149 : f32
}) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v150 = arith.maxnumf %v146, %v100 : f32
%v151 = arith.subf %v100, %v150 : f32
%v152 = math.exp %v151 : f32
%v153 = tt.splat %v150 : f32 -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v154 = arith.subf %v136, %v153 : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v155 = math.exp %v154 : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v156 = tt.splat %v152 : f32 -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v157 = arith.mulf %v102, %v156 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v158 = ttg.convert_layout %v155 : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v159 = tt.expand_dims %v158 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v160 = ttg.convert_layout %v159 : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v161 = arith.extf %v145 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> to tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v162 = tt.broadcast %v160 : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v163 = ttg.convert_layout %v162 : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v164 = arith.mulf %v163, %v161 : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v165 = "tt.reduce"(%v164) <{axis = 0 : i32}> ({
^bb0(%v166: f32, %v167: f32):
%v168 = arith.addf %v166, %v167 : f32
tt.reduce.return %v168 : f32
}) : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>>
%v169 = ttg.convert_layout %v165 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v170 = arith.addf %v157, %v169 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v171 = arith.mulf %v101, %v152 : f32
%v172 = "tt.reduce"(%v155) <{axis = 0 : i32}> ({
^bb0(%v173: f32, %v174: f32):
%v175 = arith.addf %v173, %v174 : f32
tt.reduce.return %v175 : f32
}) : (tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v176 = arith.addf %v171, %v172 : f32
scf.yield %v150, %v176, %v170 : f32, f32, tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
}
%v177 = arith.extsi %v19 : i32 to i64
%v178 = arith.muli %v41, %v177 : i64
%v179 = arith.muli %v39, %v20 : i32
%v180 = arith.extsi %v179 : i32 to i64
%v181 = arith.addi %v178, %v180 : i64
%v182 = arith.muli %v40, %v21 : i32
%v183 = arith.extsi %v182 : i32 to i64
%v184 = arith.addi %v181, %v183 : i64
%v185 = tt.splat %v184 : i64 -> tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v186 = arith.addi %v185, %v56 : tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v187 = tt.splat %v6 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v188 = tt.addptr %v187, %v186 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v189 = tt.splat %v98#1 : f32 -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v190 = arith.divf %v98#2, %v189 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v188, %v190, %v43 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v191 = arith.divsi %v184, %v32 : i64
%v192 = tt.addptr %v7, %v191 : !tt.ptr<f32>, i64
%v193 = math.log %v98#1 : f32
%v194 = arith.addf %v98#0, %v193 : f32
tt.store %v192, %v194 : !tt.ptr<f32>
}
tt.return
}
}
