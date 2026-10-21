module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: f32, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v9: i32 {tt.divisibility = 16 : i32}, %v10: i32 {tt.divisibility = 16 : i32}, %v11: i32 {tt.divisibility = 16 : i32}, %v12: i32 {tt.divisibility = 16 : i32}, %v13: i32 {tt.divisibility = 16 : i32}, %v14: i32 {tt.divisibility = 16 : i32}, %v15: i32 {tt.divisibility = 16 : i32}, %v16: i32 {tt.divisibility = 16 : i32}, %v17: i32 {tt.divisibility = 16 : i32}, %v18: i32 {tt.divisibility = 16 : i32}, %v19: i32 {tt.divisibility = 16 : i32}, %v20: i32 {tt.divisibility = 16 : i32}, %v21: i32 {tt.divisibility = 16 : i32}, %v22: i32 {tt.divisibility = 16 : i32}, %v23: i32 {tt.divisibility = 16 : i32}, %v24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v25 = arith.constant dense<256> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v26 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v27 = arith.constant 1 : i32
%v28 = arith.constant 31 : i32
%v29 = arith.constant 32 : i32
%v30 = arith.constant 0xFF800000 : f32
%v31 = arith.constant 0.000000e+00 : f32
%v32 = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v33 = arith.constant dense<0.000000e+00> : tensor<256xbf16, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v34 = arith.constant 64 : i32
%v35 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v36 = arith.constant dense<0.000000e+00> : tensor<64x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v37 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v38 = arith.constant 256 : i64
%v39 = tt.get_program_id x : i32
%v40 = tt.get_program_id y : i32
%v41 = tt.get_program_id z : i32
%v42 = arith.extsi %v39 : i32 to i64
%v43 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v44 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v45 = arith.cmpi slt, %v43, %v25 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v46 = arith.cmpi slt, %v44, %v26 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
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
%v59 = arith.extsi %v43 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v60 = tt.splat %v58 : i64 -> tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = arith.addi %v60, %v59 : tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
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
%v72 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.addptr %v72, %v61 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.load %v73, %v45, %v33 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75:3 = scf.for %v76 = %v68 to %v70 step %v34 iter_args(%v77 = %v30, %v78 = %v31, %v79 = %v32) -> (f32, f32, tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) : i32 {
%v80 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v81 = tt.splat %v76 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v82 = arith.addi %v81, %v80 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v83 = tt.splat %v70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v84 = arith.cmpi slt, %v82, %v83 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v85 = tt.addptr %v5, %v48 : !tt.ptr<i32>, i32
%v86 = tt.splat %v85 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v87 = tt.addptr %v86, %v82 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v88 = tt.load %v87, %v84, %v35 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v89 = tt.expand_dims %v88 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v90 = tt.splat %v11 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v91 = arith.muli %v89, %v90 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v92 = arith.muli %v40, %v12 : i32
%v93 = tt.splat %v92 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v94 = arith.addi %v91, %v93 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v95 = ttg.convert_layout %v43 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v96 = tt.expand_dims %v95 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v97 = tt.broadcast %v94 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v98 = tt.broadcast %v96 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v99 = arith.addi %v97, %v98 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v100 = tt.expand_dims %v82 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v101 = tt.splat %v70 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v102 = arith.cmpi slt, %v100, %v101 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v103 = ttg.convert_layout %v45 : tensor<256xi1, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v104 = tt.expand_dims %v103 {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v105 = tt.broadcast %v102 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v106 = tt.broadcast %v104 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v107 = arith.andi %v105, %v106 : tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v108 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v109 = tt.addptr %v108, %v99 : tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v110 = tt.load %v109, %v107, %v36 : tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v111 = ttg.convert_layout %v74 : tensor<256xbf16, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v112 = tt.expand_dims %v111 {axis = 0 : i32} : tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = tt.broadcast %v112 : tensor<1x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v114 = arith.mulf %v113, %v110 : tensor<64x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v115 = arith.extf %v114 : tensor<64x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v116 = "tt.reduce"(%v115) <{axis = 1 : i32}> ({
^bb0(%v117: f32, %v118: f32):
%v119 = arith.addf %v117, %v118 : f32
tt.reduce.return %v119 : f32
}) : (tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v120 = tt.splat %v3 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v121 = arith.mulf %v116, %v120 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v122 = arith.select %v84, %v121, %v37 : tensor<64xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v123 = ttg.convert_layout %v88 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v124 = tt.expand_dims %v123 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v125 = tt.splat %v13 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = arith.muli %v124, %v125 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = arith.muli %v40, %v14 : i32
%v128 = tt.splat %v127 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v129 = arith.addi %v126, %v128 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v130 = tt.expand_dims %v44 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v131 = tt.broadcast %v129 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v132 = tt.broadcast %v130 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v133 = arith.addi %v131, %v132 : tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v134 = ttg.convert_layout %v82 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v135 = tt.expand_dims %v134 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v136 = arith.cmpi slt, %v135, %v101 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v137 = tt.expand_dims %v46 {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v138 = tt.broadcast %v136 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v139 = tt.broadcast %v137 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v140 = arith.andi %v138, %v139 : tensor<64x256xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v141 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v142 = tt.addptr %v141, %v133 : tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x256xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v143 = tt.load %v142, %v140, %v36 : tensor<64x256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v144 = "tt.reduce"(%v122) <{axis = 0 : i32}> ({
^bb0(%v145: f32, %v146: f32):
%v147 = arith.maxnumf %v145, %v146 : f32
tt.reduce.return %v147 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v148 = arith.maxnumf %v144, %v77 : f32
%v149 = arith.subf %v77, %v148 : f32
%v150 = math.exp %v149 : f32
%v151 = tt.splat %v148 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v152 = arith.subf %v122, %v151 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v153 = math.exp %v152 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v154 = tt.splat %v150 : f32 -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v155 = arith.mulf %v79, %v154 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v156 = ttg.convert_layout %v153 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v157 = tt.expand_dims %v156 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v158 = arith.extf %v143 : tensor<64x256xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v159 = tt.broadcast %v157 : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v160 = arith.mulf %v159, %v158 : tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v161 = "tt.reduce"(%v160) <{axis = 0 : i32}> ({
^bb0(%v162: f32, %v163: f32):
%v164 = arith.addf %v162, %v163 : f32
tt.reduce.return %v164 : f32
}) : (tensor<64x256xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v165 = arith.addf %v155, %v161 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v166 = arith.mulf %v78, %v150 : f32
%v167 = "tt.reduce"(%v153) <{axis = 0 : i32}> ({
^bb0(%v168: f32, %v169: f32):
%v170 = arith.addf %v168, %v169 : f32
tt.reduce.return %v170 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v171 = arith.addf %v166, %v167 : f32
scf.yield %v148, %v171, %v165 : f32, f32, tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
}
%v172 = arith.extsi %v19 : i32 to i64
%v173 = arith.muli %v42, %v172 : i64
%v174 = arith.muli %v40, %v20 : i32
%v175 = arith.extsi %v174 : i32 to i64
%v176 = arith.addi %v173, %v175 : i64
%v177 = arith.muli %v41, %v21 : i32
%v178 = arith.extsi %v177 : i32 to i64
%v179 = arith.addi %v176, %v178 : i64
%v180 = arith.extsi %v44 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<256xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v181 = tt.splat %v179 : i64 -> tensor<256xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v182 = arith.addi %v181, %v180 : tensor<256xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v183 = ttg.convert_layout %v46 : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<256xi1, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v184 = tt.splat %v6 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v185 = tt.addptr %v184, %v182 : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<256xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v186 = ttg.convert_layout %v185 : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v187 = tt.splat %v75#1 : f32 -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v188 = arith.divf %v75#2, %v187 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v189 = ttg.convert_layout %v188 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<256xf32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v186, %v189, %v183 : tensor<256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v190 = arith.divsi %v179, %v38 : i64
%v191 = tt.addptr %v7, %v190 : !tt.ptr<f32>, i64
%v192 = math.log %v75#1 : f32
%v193 = arith.addf %v75#0, %v192 : f32
tt.store %v191, %v193 : !tt.ptr<f32>
}
tt.return
}
}
