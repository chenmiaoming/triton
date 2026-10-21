module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: i32, %v6: f32, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: i64 {tt.divisibility = 16 : i32}, %v10: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v11 = arith.constant 0 : i32
%v12 = arith.constant 32 : i32
%v13 = arith.constant dense<256> : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v14 = arith.constant dense<1> : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v15 = arith.constant dense<0> : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v16 = arith.constant 1024 : i64
%v17 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v18 = arith.constant 0.000000e+00 : f32
%v19 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v20 = arith.constant 262144 : i64
%v21 = arith.constant 8192 : i32
%v22 = arith.constant dense<256> : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v23 = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v24 = arith.constant 1024 : i32
%v25 = arith.constant 1 : i32
%v26 = arith.constant 2 : i32
%v27 = arith.constant 1023 : i32
%v28 = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v29 = tt.get_program_id x : i32
%v30 = tt.get_program_id y : i32
%v31 = tt.addptr %v10, %v5 : !tt.ptr<i32>, i32
%v32 = tt.load %v31 : !tt.ptr<i32>
%v33 = arith.cmpi sge, %v29, %v32 : i32
cf.cond_br %v33, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v34:2 = scf.while (%v35 = %v11, %v36 = %v5) : (i32, i32) -> (i32, i32) {
%v37 = arith.cmpi slt, %v35, %v36 : i32
scf.condition(%v37) %v35, %v36 : i32, i32
} do {
^bb0(%v35: i32, %v36: i32):
%v38 = arith.addi %v35, %v36 : i32
%v39 = arith.divsi %v38, %v26 : i32
%v40 = tt.addptr %v10, %v39 : !tt.ptr<i32>, i32
%v41 = tt.load %v40 : !tt.ptr<i32>
%v42 = arith.cmpi sle, %v41, %v29 : i32
%v43 = arith.select %v42, %v36, %v39 : i32
%v44 = scf.if %v42 -> (i32) {
%v45 = arith.addi %v39, %v25 : i32
scf.yield %v45 : i32
} else {
scf.yield %v35 : i32
}
scf.yield %v44, %v43 : i32, i32
}
%v46 = arith.subi %v34#0, %v25 : i32
%v47 = tt.addptr %v4, %v46 : !tt.ptr<i32>, i32
%v48 = tt.load %v47 : !tt.ptr<i32>
%v49 = arith.cmpi sle, %v48, %v11 : i32
cf.cond_br %v49, ^bb1, ^bb3
^bb3:
%v50 = arith.addi %v48, %v27 : i32
%v51 = arith.divsi %v50, %v24 : i32
%v52 = arith.muli %v51, %v12 : i32
%v53 = arith.addi %v48, %v52 : i32
%v54 = arith.subi %v53, %v25 : i32
%v55 = arith.divsi %v54, %v52 : i32
%v56 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v57 = tt.splat %v55 : i32 -> tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v58 = arith.cmpi slt, %v56, %v57 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v59 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v60 = arith.cmpi slt, %v59, %v13 : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = arith.select %v60, %v14, %v15 : tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = arith.cmpi ne, %v61, %v15 : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v63 = arith.extsi %v29 : i32 to i64
%v64 = arith.muli %v63, %v16 : i64
%v65 = arith.muli %v30, %v12 : i32
%v66 = arith.extsi %v65 : i32 to i64
%v67 = arith.addi %v64, %v66 : i64
%v68 = arith.extsi %v56 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v69 = tt.splat %v67 : i64 -> tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v70 = arith.addi %v69, %v68 : tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = tt.splat %v2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = tt.addptr %v71, %v70 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.load %v72, %v58, %v17 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.splat %v3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75 = tt.addptr %v74, %v70 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v76 = tt.load %v75, %v58, %v19 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v77 = arith.cmpf ogt, %v76, %v19 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v78 = arith.andi %v58, %v77 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v79 = arith.cmpf une, %v73, %v17 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v80 = arith.andi %v78, %v79 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v81 = arith.select %v80, %v73, %v17 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v82 = tt.reshape %v81 allow_reorder : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v83 = "tt.reduce"(%v82) <{axis = 0 : i32}> ({
^bb0(%v84: f32, %v85: f32):
%v86 = arith.maxnumf %v84, %v85 : f32
tt.reduce.return %v86 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v87 = tt.splat %v83 : f32 -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v88 = arith.subf %v81, %v87 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v89 = math.exp %v88 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v90 = arith.select %v80, %v89, %v19 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v91 = arith.mulf %v76, %v90 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v92 = tt.reshape %v91 allow_reorder : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v93 = "tt.reduce"(%v92) <{axis = 0 : i32}> ({
^bb0(%v94: f32, %v95: f32):
%v96 = arith.addf %v94, %v95 : f32
tt.reduce.return %v96 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v97 = arith.muli %v63, %v20 : i64
%v98 = arith.muli %v30, %v21 : i32
%v99 = arith.extsi %v98 : i32 to i64
%v100 = arith.addi %v97, %v99 : i64
%v101 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v102 = tt.expand_dims %v101 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v103 = ttg.convert_layout %v102 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v104 = arith.muli %v103, %v22 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v105 = arith.extsi %v104 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v106 = tt.splat %v100 : i64 -> tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v107 = arith.addi %v106, %v105 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v108 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v109 = tt.expand_dims %v108 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v110 = ttg.convert_layout %v109 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v111 = arith.extsi %v110 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> to tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v112 = tt.broadcast %v107 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = ttg.convert_layout %v112 : tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v114 = tt.broadcast %v111 : tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v115 = arith.addi %v113, %v114 : tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v116 = ttg.convert_layout %v58 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v117 = tt.expand_dims %v116 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v118 = ttg.convert_layout %v117 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = ttg.convert_layout %v62 : tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v120 = tt.expand_dims %v119 {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v121 = ttg.convert_layout %v120 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v122 = tt.broadcast %v118 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v123 = ttg.convert_layout %v122 : tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v124 = tt.broadcast %v121 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v125 = arith.andi %v123, %v124 : tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v126 = tt.splat %v1 : !tt.ptr<f32> -> tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v127 = tt.addptr %v126, %v115 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v128 = ttg.convert_layout %v127 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v129 = ttg.convert_layout %v125 : tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v130 = tt.load %v128, %v129, %v28 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v131 = ttg.convert_layout %v130 : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v132 = ttg.convert_layout %v90 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v133 = tt.expand_dims %v132 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v134 = ttg.convert_layout %v133 : tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v135 = tt.broadcast %v134 : tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v136 = ttg.convert_layout %v135 : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v137 = arith.mulf %v131, %v136 : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v138 = "tt.reduce"(%v137) <{axis = 0 : i32}> ({
^bb0(%v139: f32, %v140: f32):
%v141 = arith.addf %v139, %v140 : f32
tt.reduce.return %v141 : f32
}) : (tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>>
%v142 = ttg.convert_layout %v138 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>> -> tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v143 = arith.cmpf oeq, %v93, %v18 : f32
%v144 = tt.splat %v93 : f32 -> tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v145 = arith.divf %v142, %v144 : tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v146 = tt.splat %v143 : i1 -> tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v147 = arith.select %v146, %v23, %v145 : tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v148 = arith.muli %v63, %v7 : i64
%v149 = arith.extsi %v30 : i32 to i64
%v150 = arith.muli %v149, %v8 : i64
%v151 = arith.addi %v148, %v150 : i64
%v152 = arith.extsi %v59 : tensor<256xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<256xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v153 = tt.splat %v151 : i64 -> tensor<256xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v154 = arith.addi %v153, %v152 : tensor<256xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v155 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v156 = tt.addptr %v155, %v154 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v157 = arith.truncf %v147 : tensor<256xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<256xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v158 = ttg.convert_layout %v156 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v159 = ttg.convert_layout %v157 : tensor<256xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xbf16, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v160 = ttg.convert_layout %v62 : tensor<256xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<256xi1, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v158, %v159, %v160 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
