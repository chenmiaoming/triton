module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: f32, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v9: i32 {tt.divisibility = 16 : i32}, %v10: i32 {tt.divisibility = 16 : i32}, %v11: i32 {tt.divisibility = 16 : i32}, %v12: i32 {tt.divisibility = 16 : i32}, %v13: i32 {tt.divisibility = 16 : i32}, %v14: i32 {tt.divisibility = 16 : i32}, %v15: i32 {tt.divisibility = 16 : i32}, %v16: i32 {tt.divisibility = 16 : i32}, %v17: i32 {tt.divisibility = 16 : i32}, %v18: i32 {tt.divisibility = 16 : i32}, %v19: i32 {tt.divisibility = 16 : i32}, %v20: i32 {tt.divisibility = 16 : i32}, %v21: i32 {tt.divisibility = 16 : i32}, %v22: i32 {tt.divisibility = 16 : i32}, %v23: i32 {tt.divisibility = 16 : i32}, %v24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v25 = arith.constant dense<128> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v26 = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v27 = arith.constant 1 : i32
%v28 = arith.constant 31 : i32
%v29 = arith.constant 32 : i32
%v30 = arith.constant 0xFF800000 : f32
%v31 = arith.constant 0.000000e+00 : f32
%v32 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v33 = arith.constant dense<0.000000e+00> : tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v34 = arith.constant 64 : i32
%v35 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v36 = arith.constant dense<0.000000e+00> : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v37 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v38 = arith.constant 128 : i64
%v39 = tt.get_program_id x : i32
%v40 = tt.get_program_id y : i32
%v41 = tt.get_program_id z : i32
%v42 = arith.extsi %v39 : i32 to i64
%v43 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v44 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v45 = arith.cmpi slt, %v43, %v25 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v46 = arith.cmpi slt, %v44, %v26 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v47 = tt.addptr %v4, %v42 : !tt.ptr<i32>, i64
%v48 = tt.load %v47 : !tt.ptr<i32>
%v49 = tt.addptr %v47, %v27 : !tt.ptr<i32>, i32
%v50 = tt.load %v49 : !tt.ptr<i32>
%v51 = arith.subi %v50, %v48 : i32
%v52 = tt.addptr %v8, %v42 : !tt.ptr<i32>, i64
%v53 = tt.load %v52 : !tt.ptr<i32>
%v54 = arith.extsi %v9 : i32 to i64
%v55 = arith.muli %v42, %v54 : i64
%v56 = arith.muli %v40, %v10 : i32
%v57 = arith.extsi %v56 : i32 to i64
%v58 = arith.addi %v55, %v57 : i64
%v59 = arith.extsi %v43 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v60 = tt.splat %v58 : i64 -> tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = arith.addi %v60, %v59 : tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = arith.subi %v53, %v27 : i32
%v63 = arith.addi %v51, %v62 : i32
%v64 = arith.divsi %v63, %v53 : i32
%v65 = arith.addi %v64, %v28 : i32
%v66 = arith.divsi %v65, %v29 : i32
%v67 = arith.muli %v66, %v29 : i32
%v68 = arith.muli %v67, %v41 : i32
%v69 = arith.addi %v68, %v67 : i32
%v70 = arith.minsi %v69, %v51 : i32
%v71 = arith.cmpi sgt, %v70, %v68 : i32
scf.if %v71 {
%v72 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.addptr %v72, %v61 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.load %v73, %v45, %v33 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75:3 = scf.for %v76 = %v68 to %v70 step %v34 iter_args(%v77 = %v30, %v78 = %v31, %v79 = %v32) -> (f32, f32, tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) : i32 {
%v80 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v81 = tt.splat %v76 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v82 = arith.addi %v81, %v80 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v83 = tt.splat %v70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v84 = arith.cmpi slt, %v82, %v83 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v85 = tt.addptr %v5, %v48 : !tt.ptr<i32>, i32
%v86 = tt.splat %v85 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v87 = tt.addptr %v86, %v82 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v88 = tt.load %v87, %v84, %v35 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v89 = tt.expand_dims %v88 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v90 = tt.splat %v11 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v91 = arith.muli %v89, %v90 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v92 = arith.muli %v40, %v12 : i32
%v93 = tt.splat %v92 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v94 = arith.addi %v91, %v93 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v95 = tt.expand_dims %v44 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v96 = tt.broadcast %v94 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v97 = tt.broadcast %v95 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v98 = arith.addi %v96, %v97 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v99 = tt.expand_dims %v82 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v100 = tt.splat %v70 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v101 = arith.cmpi slt, %v99, %v100 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v102 = ttg.convert_layout %v45 : tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v103 = tt.expand_dims %v102 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v104 = tt.broadcast %v101 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v105 = tt.broadcast %v103 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v106 = arith.andi %v104, %v105 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v107 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v108 = tt.addptr %v107, %v98 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v109 = tt.load %v108, %v106, %v36 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v110 = ttg.convert_layout %v74 : tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v111 = tt.expand_dims %v110 {axis = 0 : i32} : tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v112 = tt.broadcast %v111 : tensor<1x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = arith.mulf %v112, %v109 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v114 = arith.extf %v113 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v115 = "tt.reduce"(%v114) <{axis = 1 : i32}> ({
^bb0(%v116: f32, %v117: f32):
%v118 = arith.addf %v116, %v117 : f32
tt.reduce.return %v118 : f32
}) : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v119 = tt.splat %v3 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v120 = arith.mulf %v115, %v119 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v121 = arith.select %v84, %v120, %v37 : tensor<64xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v122 = tt.splat %v13 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v123 = arith.muli %v89, %v122 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v124 = arith.muli %v40, %v14 : i32
%v125 = tt.splat %v124 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = arith.addi %v123, %v125 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = tt.broadcast %v126 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v128 = arith.addi %v127, %v97 : tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v129 = tt.expand_dims %v46 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v130 = tt.broadcast %v129 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v131 = arith.andi %v104, %v130 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v132 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v133 = tt.addptr %v132, %v128 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v134 = tt.load %v133, %v131, %v36 : tensor<64x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v135 = "tt.reduce"(%v121) <{axis = 0 : i32}> ({
^bb0(%v136: f32, %v137: f32):
%v138 = arith.maxnumf %v136, %v137 : f32
tt.reduce.return %v138 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v139 = arith.maxnumf %v135, %v77 : f32
%v140 = arith.subf %v77, %v139 : f32
%v141 = math.exp %v140 : f32
%v142 = tt.splat %v139 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v143 = arith.subf %v121, %v142 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v144 = math.exp %v143 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v145 = tt.splat %v141 : f32 -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v146 = arith.mulf %v79, %v145 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v147 = tt.expand_dims %v144 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v148 = arith.extf %v134 : tensor<64x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v149 = tt.broadcast %v147 : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v150 = arith.mulf %v149, %v148 : tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v151 = "tt.reduce"(%v150) <{axis = 0 : i32}> ({
^bb0(%v152: f32, %v153: f32):
%v154 = arith.addf %v152, %v153 : f32
tt.reduce.return %v154 : f32
}) : (tensor<64x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v155 = arith.addf %v146, %v151 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v156 = arith.mulf %v78, %v141 : f32
%v157 = "tt.reduce"(%v144) <{axis = 0 : i32}> ({
^bb0(%v158: f32, %v159: f32):
%v160 = arith.addf %v158, %v159 : f32
tt.reduce.return %v160 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v161 = arith.addf %v156, %v157 : f32
scf.yield %v139, %v161, %v155 : f32, f32, tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
}
%v162 = arith.extsi %v19 : i32 to i64
%v163 = arith.muli %v42, %v162 : i64
%v164 = arith.muli %v40, %v20 : i32
%v165 = arith.extsi %v164 : i32 to i64
%v166 = arith.addi %v163, %v165 : i64
%v167 = arith.muli %v41, %v21 : i32
%v168 = arith.extsi %v167 : i32 to i64
%v169 = arith.addi %v166, %v168 : i64
%v170 = arith.extsi %v44 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v171 = tt.splat %v169 : i64 -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v172 = arith.addi %v171, %v170 : tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v173 = ttg.convert_layout %v46 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v174 = tt.splat %v6 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v175 = tt.addptr %v174, %v172 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v176 = ttg.convert_layout %v175 : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v177 = tt.splat %v75#1 : f32 -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v178 = arith.divf %v75#2, %v177 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v179 = ttg.convert_layout %v178 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v176, %v179, %v173 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v180 = arith.divsi %v169, %v38 : i64
%v181 = tt.addptr %v7, %v180 : !tt.ptr<f32>, i64
%v182 = math.log %v75#1 : f32
%v183 = arith.addf %v75#0, %v182 : f32
tt.store %v181, %v183 : !tt.ptr<f32>
}
tt.return
}
}
