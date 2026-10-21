module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: f32, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v9: i32 {tt.divisibility = 16 : i32}, %v10: i32 {tt.divisibility = 16 : i32}, %v11: i32 {tt.divisibility = 16 : i32}, %v12: i32 {tt.divisibility = 16 : i32}, %v13: i32 {tt.divisibility = 16 : i32}, %v14: i32 {tt.divisibility = 16 : i32}, %v15: i32 {tt.divisibility = 16 : i32}, %v16: i32 {tt.divisibility = 16 : i32}, %v17: i32 {tt.divisibility = 16 : i32}, %v18: i32 {tt.divisibility = 16 : i32}, %v19: i32 {tt.divisibility = 16 : i32}, %v20: i32 {tt.divisibility = 16 : i32}, %v21: i32 {tt.divisibility = 16 : i32}, %v22: i32 {tt.divisibility = 16 : i32}, %v23: i32 {tt.divisibility = 16 : i32}, %v24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v25 = arith.constant dense<0.000000e+00> : tensor<64xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v26 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v27 = arith.constant dense<64> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v28 = arith.constant dense<64> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v29 = arith.constant 1 : i32
%v30 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v31 = arith.constant 32 : i32
%v32 = arith.constant 0xFF800000 : f32
%v33 = arith.constant 0.000000e+00 : f32
%v34 = arith.constant 64 : i64
%v35 = arith.constant 64 : i32
%v36 = arith.constant 31 : i32
%v37 = arith.constant dense<0.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v38 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v39 = tt.get_program_id x : i32
%v40 = tt.get_program_id y : i32
%v41 = tt.get_program_id z : i32
%v42 = arith.extsi %v39 : i32 to i64
%v43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v44 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v45 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v46 = arith.cmpi slt, %v44, %v27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v47 = arith.cmpi slt, %v45, %v28 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v48 = tt.addptr %v4, %v42 : !tt.ptr<i32>, i64
%v49 = tt.load %v48 : !tt.ptr<i32>
%v50 = tt.addptr %v48, %v29 : !tt.ptr<i32>, i32
%v51 = tt.load %v50 : !tt.ptr<i32>
%v52 = arith.subi %v51, %v49 : i32
%v53 = tt.addptr %v8, %v42 : !tt.ptr<i32>, i64
%v54 = tt.load %v53 : !tt.ptr<i32>
%v55 = arith.extsi %v9 : i32 to i64
%v56 = arith.muli %v42, %v55 : i64
%v57 = arith.muli %v40, %v10 : i32
%v58 = arith.extsi %v57 : i32 to i64
%v59 = arith.addi %v56, %v58 : i64
%v60 = arith.extsi %v44 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v61 = arith.extsi %v45 : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<64xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = tt.splat %v59 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v63 = arith.addi %v62, %v60 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v64 = arith.subi %v54, %v29 : i32
%v65 = arith.addi %v52, %v64 : i32
%v66 = arith.divsi %v65, %v54 : i32
%v67 = arith.addi %v66, %v36 : i32
%v68 = arith.divsi %v67, %v31 : i32
%v69 = arith.muli %v68, %v31 : i32
%v70 = arith.muli %v69, %v41 : i32
%v71 = arith.addi %v70, %v69 : i32
%v72 = arith.minsi %v71, %v52 : i32
%v73 = arith.cmpi sgt, %v72, %v70 : i32
scf.if %v73 {
%v74 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v75 = tt.addptr %v74, %v63 : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v76 = tt.load %v75, %v46, %v25 : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v77 = tt.splat %v72 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v78 = tt.addptr %v5, %v49 : !tt.ptr<i32>, i32
%v79 = tt.splat %v78 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v80 = tt.splat %v11 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v81 = arith.muli %v40, %v12 : i32
%v82 = tt.splat %v81 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v83 = tt.expand_dims %v44 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v84 = tt.broadcast %v83 : tensor<1x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v85 = tt.splat %v72 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v86 = tt.expand_dims %v46 {axis = 0 : i32} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v87 = tt.broadcast %v86 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v88 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v89 = tt.expand_dims %v76 {axis = 0 : i32} : tensor<64xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v90 = tt.broadcast %v89 : tensor<1x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v91 = tt.splat %v3 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v92 = tt.splat %v13 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v93 = arith.muli %v40, %v14 : i32
%v94 = tt.splat %v93 : i32 -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v95 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v96:3 = scf.for %v97 = %v70 to %v72 step %v35 iter_args(%v98 = %v32, %v99 = %v33, %v100 = %v37) -> (f32, f32, tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) : i32 {
%v101 = tt.splat %v97 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v102 = arith.addi %v101, %v43 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v103 = arith.cmpi slt, %v102, %v77 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v104 = tt.addptr %v79, %v102 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v105 = tt.load %v104, %v103, %v26 : tensor<64x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v106 = tt.expand_dims %v105 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v107 = arith.muli %v106, %v80 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v108 = arith.addi %v107, %v82 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v109 = tt.broadcast %v108 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v110 = arith.addi %v109, %v84 : tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v111 = tt.expand_dims %v102 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v112 = arith.cmpi slt, %v111, %v85 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = tt.broadcast %v112 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v114 = arith.andi %v113, %v87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v115 = tt.addptr %v88, %v110 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v116 = tt.load %v115, %v114, %v30 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v117 = arith.mulf %v90, %v116 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v118 = arith.extf %v117 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = "tt.reduce"(%v118) <{axis = 1 : i32}> ({
^bb0(%v120: f32, %v121: f32):
%v122 = arith.addf %v120, %v121 : f32
tt.reduce.return %v122 : f32
}) : (tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v123 = arith.mulf %v119, %v91 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v124 = arith.select %v103, %v123, %v38 : tensor<64xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v125 = arith.muli %v106, %v92 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = arith.addi %v125, %v94 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = tt.broadcast %v126 : tensor<64x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v128 = arith.addi %v127, %v84 : tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v129 = tt.addptr %v95, %v128 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v130 = tt.load %v129, %v114, %v30 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v131 = "tt.reduce"(%v124) <{axis = 0 : i32}> ({
^bb0(%v132: f32, %v133: f32):
%v134 = arith.maxnumf %v132, %v133 : f32
tt.reduce.return %v134 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v135 = arith.maxnumf %v131, %v98 : f32
%v136 = arith.subf %v98, %v135 : f32
%v137 = math.exp %v136 : f32
%v138 = tt.splat %v135 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v139 = arith.subf %v124, %v138 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v140 = math.exp %v139 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v141 = tt.splat %v137 : f32 -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v142 = arith.mulf %v100, %v141 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v143 = tt.expand_dims %v140 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v144 = arith.extf %v130 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v145 = tt.broadcast %v143 : tensor<64x1xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v146 = arith.mulf %v145, %v144 : tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v147 = "tt.reduce"(%v146) <{axis = 0 : i32}> ({
^bb0(%v148: f32, %v149: f32):
%v150 = arith.addf %v148, %v149 : f32
tt.reduce.return %v150 : f32
}) : (tensor<64x64xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v151 = arith.addf %v142, %v147 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v152 = arith.mulf %v99, %v137 : f32
%v153 = "tt.reduce"(%v140) <{axis = 0 : i32}> ({
^bb0(%v154: f32, %v155: f32):
%v156 = arith.addf %v154, %v155 : f32
tt.reduce.return %v156 : f32
}) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>) -> f32
%v157 = arith.addf %v152, %v153 : f32
scf.yield %v135, %v157, %v151 : f32, f32, tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
}
%v158 = arith.extsi %v19 : i32 to i64
%v159 = arith.muli %v42, %v158 : i64
%v160 = arith.muli %v40, %v20 : i32
%v161 = arith.extsi %v160 : i32 to i64
%v162 = arith.addi %v159, %v161 : i64
%v163 = arith.muli %v41, %v21 : i32
%v164 = arith.extsi %v163 : i32 to i64
%v165 = arith.addi %v162, %v164 : i64
%v166 = tt.splat %v165 : i64 -> tensor<64xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v167 = arith.addi %v166, %v61 : tensor<64xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v168 = tt.splat %v6 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v169 = tt.addptr %v168, %v167 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<64xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v170 = tt.splat %v96#1 : f32 -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v171 = arith.divf %v96#2, %v170 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v172 = ttg.convert_layout %v171 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v169, %v172, %v47 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v173 = arith.divsi %v165, %v34 : i64
%v174 = tt.addptr %v7, %v173 : !tt.ptr<f32>, i64
%v175 = math.log %v96#1 : f32
%v176 = arith.addf %v96#0, %v175 : f32
tt.store %v174, %v176 : !tt.ptr<f32>
}
tt.return
}
}
