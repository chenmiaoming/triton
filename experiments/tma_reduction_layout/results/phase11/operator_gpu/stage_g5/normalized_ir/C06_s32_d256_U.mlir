module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: i32, %v6: f32, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: i64 {tt.divisibility = 16 : i32}, %v10: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v11 = arith.constant dense<256> : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v12 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v13 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v14 = arith.constant dense<0> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v15 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v16 = arith.constant dense<1> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v17 = arith.constant dense<1> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v18 = arith.constant dense<256> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v19 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v20 = arith.constant 32 : i32
%v21 = arith.constant 0 : i32
%v22 = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v23 = arith.constant 1024 : i64
%v24 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v25 = arith.constant 0.000000e+00 : f32
%v26 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v27 = arith.constant 262144 : i64
%v28 = arith.constant 8192 : i32
%v29 = arith.constant 1023 : i32
%v30 = arith.constant 2 : i32
%v31 = arith.constant 1 : i32
%v32 = arith.constant 1024 : i32
%v33 = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v34 = tt.get_program_id x : i32
%v35 = tt.get_program_id y : i32
%v36 = tt.addptr %v10, %v5 : !tt.ptr<i32>, i32
%v37 = tt.load %v36 : !tt.ptr<i32>
%v38 = arith.cmpi sge, %v34, %v37 : i32
cf.cond_br %v38, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v39:2 = scf.while (%v40 = %v21, %v41 = %v5) : (i32, i32) -> (i32, i32) {
%v42 = arith.cmpi slt, %v40, %v41 : i32
scf.condition(%v42) %v40, %v41 : i32, i32
} do {
^bb0(%v40: i32, %v41: i32):
%v43 = arith.addi %v40, %v41 : i32
%v44 = arith.divsi %v43, %v30 : i32
%v45 = tt.addptr %v10, %v44 : !tt.ptr<i32>, i32
%v46 = tt.load %v45 : !tt.ptr<i32>
%v47 = arith.cmpi sle, %v46, %v34 : i32
%v48 = arith.select %v47, %v41, %v44 : i32
%v49 = scf.if %v47 -> (i32) {
%v50 = arith.addi %v44, %v31 : i32
scf.yield %v50 : i32
} else {
scf.yield %v40 : i32
}
scf.yield %v49, %v48 : i32, i32
}
%v51 = arith.subi %v39#0, %v31 : i32
%v52 = tt.addptr %v4, %v51 : !tt.ptr<i32>, i32
%v53 = tt.load %v52 : !tt.ptr<i32>
%v54 = arith.cmpi sle, %v53, %v21 : i32
cf.cond_br %v54, ^bb1, ^bb3
^bb3:
%v55 = arith.addi %v53, %v29 : i32
%v56 = arith.divsi %v55, %v32 : i32
%v57 = arith.muli %v56, %v20 : i32
%v58 = arith.addi %v53, %v57 : i32
%v59 = arith.subi %v58, %v31 : i32
%v60 = arith.divsi %v59, %v57 : i32
%v61 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v62 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v63 = tt.splat %v60 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v64 = tt.splat %v60 : i32 -> tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v65 = arith.cmpi slt, %v61, %v63 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v66 = arith.cmpi slt, %v62, %v64 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v67 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v68 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v69 = arith.cmpi slt, %v67, %v19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v70 = arith.cmpi slt, %v68, %v18 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = arith.select %v69, %v17, %v15 : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v72 = arith.select %v70, %v16, %v14 : tensor<256xi1, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = arith.cmpi ne, %v71, %v15 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v74 = arith.cmpi ne, %v72, %v14 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75 = arith.extsi %v34 : i32 to i64
%v76 = arith.muli %v75, %v23 : i64
%v77 = arith.muli %v35, %v20 : i32
%v78 = arith.extsi %v77 : i32 to i64
%v79 = arith.addi %v76, %v78 : i64
%v80 = arith.extsi %v61 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v81 = arith.extsi %v62 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v82 = tt.splat %v79 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v83 = tt.splat %v79 : i64 -> tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v84 = arith.addi %v82, %v80 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v85 = arith.addi %v83, %v81 : tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v86 = tt.splat %v2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v87 = tt.splat %v2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v88 = tt.addptr %v86, %v84 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v89 = tt.addptr %v87, %v85 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v90 = tt.load %v88, %v65, %v13 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v91 = tt.load %v89, %v66, %v24 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v92 = tt.splat %v3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v93 = tt.splat %v3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v94 = tt.addptr %v92, %v84 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v95 = tt.addptr %v93, %v85 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v96 = tt.load %v94, %v65, %v12 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v97 = tt.load %v95, %v66, %v26 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v98 = arith.cmpf ogt, %v96, %v12 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v99 = arith.cmpf ogt, %v97, %v26 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v100 = arith.andi %v65, %v98 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v101 = arith.andi %v66, %v99 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v102 = arith.cmpf une, %v90, %v13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v103 = arith.cmpf une, %v91, %v24 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v104 = arith.andi %v100, %v102 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v105 = arith.andi %v101, %v103 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v106 = arith.select %v104, %v90, %v13 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v107 = arith.select %v105, %v91, %v24 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v108 = tt.reshape %v107 allow_reorder : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v109 = "tt.reduce"(%v108) <{axis = 0 : i32}> ({
^bb0(%v110: f32, %v111: f32):
%v112 = arith.maxnumf %v110, %v111 : f32
tt.reduce.return %v112 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v113 = tt.splat %v109 : f32 -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v114 = tt.splat %v109 : f32 -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v115 = arith.subf %v106, %v113 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v116 = arith.subf %v107, %v114 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v117 = math.exp %v115 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v118 = math.exp %v116 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v119 = arith.select %v104, %v117, %v12 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v120 = arith.select %v105, %v118, %v26 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v121 = arith.mulf %v97, %v120 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v122 = tt.reshape %v121 allow_reorder : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v123 = "tt.reduce"(%v122) <{axis = 0 : i32}> ({
^bb0(%v124: f32, %v125: f32):
%v126 = arith.addf %v124, %v125 : f32
tt.reduce.return %v126 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v127 = arith.muli %v75, %v27 : i64
%v128 = arith.muli %v35, %v28 : i32
%v129 = arith.extsi %v128 : i32 to i64
%v130 = arith.addi %v127, %v129 : i64
%v131 = tt.expand_dims %v61 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v132 = arith.muli %v131, %v11 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v133 = arith.extsi %v132 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> to tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v134 = tt.splat %v130 : i64 -> tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v135 = arith.addi %v134, %v133 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v136 = tt.expand_dims %v67 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v137 = arith.extsi %v136 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> to tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v138 = tt.broadcast %v135 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v139 = tt.broadcast %v137 : tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v140 = arith.addi %v138, %v139 : tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v141 = tt.expand_dims %v65 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v142 = tt.expand_dims %v73 {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v143 = tt.broadcast %v141 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v144 = tt.broadcast %v142 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v145 = arith.andi %v143, %v144 : tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v146 = tt.splat %v1 : !tt.ptr<f32> -> tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v147 = tt.addptr %v146, %v140 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v148 = tt.load %v147, %v145, %v22 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v149 = tt.expand_dims %v119 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v150 = tt.broadcast %v149 : tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v151 = arith.mulf %v148, %v150 : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v152 = "tt.reduce"(%v151) <{axis = 0 : i32}> ({
^bb0(%v153: f32, %v154: f32):
%v155 = arith.addf %v153, %v154 : f32
tt.reduce.return %v155 : f32
}) : (tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v156 = arith.cmpf oeq, %v123, %v25 : f32
%v157 = tt.splat %v123 : f32 -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v158 = arith.divf %v152, %v157 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v159 = tt.splat %v156 : i1 -> tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v160 = arith.select %v159, %v33, %v158 : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v161 = arith.muli %v75, %v7 : i64
%v162 = arith.extsi %v35 : i32 to i64
%v163 = arith.muli %v162, %v8 : i64
%v164 = arith.addi %v161, %v163 : i64
%v165 = arith.extsi %v68 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v166 = tt.splat %v164 : i64 -> tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v167 = arith.addi %v166, %v165 : tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v168 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v169 = tt.addptr %v168, %v167 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v170 = arith.truncf %v160 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> to tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v171 = ttg.convert_layout %v170 : tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<256xbf16, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v169, %v171, %v74 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
