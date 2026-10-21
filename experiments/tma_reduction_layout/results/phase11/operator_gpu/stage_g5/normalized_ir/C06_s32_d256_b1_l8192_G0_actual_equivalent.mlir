module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: f32, %v6: i64 {tt.divisibility = 16 : i32}, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v10 = arith.constant 1023 : i32
%v11 = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v12 = arith.constant 0.000000e+00 : f32
%v13 = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v14 = arith.constant dense<256> : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v15 = arith.constant 8192 : i32
%v16 = arith.constant 262144 : i64
%v17 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v18 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v19 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v20 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v21 = arith.constant 1024 : i64
%v22 = arith.constant dense<0> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v23 = arith.constant dense<1> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v24 = arith.constant dense<256> : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v25 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v26 = arith.constant dense<1> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v27 = arith.constant dense<256> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v28 = arith.constant 32 : i32
%v29 = arith.constant 1024 : i32
%v30 = arith.constant 2 : i32
%v31 = arith.constant 0 : i32
%v32 = arith.constant 1 : i32
%v33 = tt.get_program_id x : i32
%v34 = tt.get_program_id y : i32
%v35 = tt.addptr %v9, %v32 : !tt.ptr<i32>, i32
%v36 = tt.load %v35 : !tt.ptr<i32>
%v37 = arith.cmpi sge, %v33, %v36 : i32
cf.cond_br %v37, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v38:2 = scf.while (%v39 = %v31, %v40 = %v32) : (i32, i32) -> (i32, i32) {
%v41 = arith.cmpi slt, %v39, %v40 : i32
scf.condition(%v41) %v39, %v40 : i32, i32
} do {
^bb0(%v39: i32, %v40: i32):
%v42 = arith.addi %v39, %v40 : i32
%v43 = arith.divsi %v42, %v30 : i32
%v44 = tt.addptr %v9, %v43 : !tt.ptr<i32>, i32
%v45 = tt.load %v44 : !tt.ptr<i32>
%v46 = arith.cmpi sle, %v45, %v33 : i32
%v47 = arith.select %v46, %v40, %v43 : i32
%v48 = scf.if %v46 -> (i32) {
%v49 = arith.addi %v43, %v32 : i32
scf.yield %v49 : i32
} else {
scf.yield %v39 : i32
}
scf.yield %v48, %v47 : i32, i32
}
%v50 = arith.subi %v38#0, %v32 : i32
%v51 = tt.addptr %v4, %v50 : !tt.ptr<i32>, i32
%v52 = tt.load %v51 : !tt.ptr<i32>
%v53 = arith.cmpi sle, %v52, %v31 : i32
cf.cond_br %v53, ^bb1, ^bb3
^bb3:
%v54 = arith.addi %v52, %v10 : i32
%v55 = arith.divsi %v54, %v29 : i32
%v56 = arith.muli %v55, %v28 : i32
%v57 = arith.addi %v52, %v56 : i32
%v58 = arith.subi %v57, %v32 : i32
%v59 = arith.divsi %v58, %v56 : i32
%v60 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v61 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = tt.splat %v59 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v63 = tt.splat %v59 : i32 -> tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v64 = arith.cmpi slt, %v60, %v62 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v65 = arith.cmpi slt, %v61, %v63 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v66 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v67 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v68 = arith.cmpi slt, %v66, %v27 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v69 = arith.select %v68, %v26, %v25 : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v70 = arith.cmpi ne, %v69, %v25 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v71 = arith.cmpi slt, %v67, %v24 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = arith.select %v71, %v23, %v22 : tensor<256xi1, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = arith.cmpi ne, %v72, %v22 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = arith.extsi %v33 : i32 to i64
%v75 = arith.muli %v74, %v21 : i64
%v76 = arith.muli %v34, %v28 : i32
%v77 = arith.extsi %v76 : i32 to i64
%v78 = arith.addi %v75, %v77 : i64
%v79 = arith.extsi %v60 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v80 = tt.splat %v78 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v81 = arith.addi %v80, %v79 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v82 = arith.extsi %v61 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v83 = tt.splat %v78 : i64 -> tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v84 = arith.addi %v83, %v82 : tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v85 = tt.splat %v2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v86 = tt.addptr %v85, %v81 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v87 = tt.load %v86, %v64, %v20 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v88 = tt.splat %v2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v89 = tt.addptr %v88, %v84 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v90 = tt.load %v89, %v65, %v19 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v91 = tt.splat %v3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v92 = tt.addptr %v91, %v81 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v93 = tt.load %v92, %v64, %v18 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v94 = tt.splat %v3 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v95 = tt.addptr %v94, %v84 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v96 = tt.load %v95, %v65, %v17 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v97 = arith.cmpf ogt, %v93, %v18 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v98 = arith.cmpf ogt, %v96, %v17 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v99 = arith.andi %v64, %v97 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v100 = arith.andi %v65, %v98 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v101 = arith.cmpf une, %v87, %v20 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v102 = arith.cmpf une, %v90, %v19 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v103 = arith.andi %v99, %v101 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v104 = arith.andi %v100, %v102 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v105 = arith.select %v103, %v87, %v20 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v106 = arith.select %v104, %v90, %v19 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v107 = "tt.reduce"(%v106) <{axis = 0 : i32}> ({
^bb0(%v108: f32, %v109: f32):
%v110 = arith.maxnumf %v108, %v109 : f32
tt.reduce.return %v110 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v111 = tt.splat %v107 : f32 -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v112 = arith.subf %v105, %v111 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v113 = tt.splat %v107 : f32 -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v114 = arith.subf %v106, %v113 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v115 = math.exp %v112 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v116 = math.exp %v114 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v117 = arith.select %v103, %v115, %v18 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v118 = arith.select %v104, %v116, %v17 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v119 = arith.mulf %v96, %v118 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v120 = "tt.reduce"(%v119) <{axis = 0 : i32}> ({
^bb0(%v121: f32, %v122: f32):
%v123 = arith.addf %v121, %v122 : f32
tt.reduce.return %v123 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v124 = arith.muli %v74, %v16 : i64
%v125 = arith.muli %v34, %v15 : i32
%v126 = arith.extsi %v125 : i32 to i64
%v127 = arith.addi %v124, %v126 : i64
%v128 = tt.expand_dims %v60 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v129 = arith.muli %v128, %v14 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v130 = arith.extsi %v129 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> to tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v131 = tt.splat %v127 : i64 -> tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v132 = arith.addi %v131, %v130 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v133 = tt.expand_dims %v66 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v134 = arith.extsi %v133 : tensor<1x256xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> to tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v135 = tt.broadcast %v132 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v136 = tt.broadcast %v134 : tensor<1x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v137 = arith.addi %v135, %v136 : tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v138 = tt.expand_dims %v64 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v139 = tt.expand_dims %v70 {axis = 0 : i32} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v140 = tt.broadcast %v138 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v141 = tt.broadcast %v139 : tensor<1x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v142 = arith.andi %v140, %v141 : tensor<32x256xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v143 = tt.splat %v1 : !tt.ptr<f32> -> tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v144 = tt.addptr %v143, %v137 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<32x256xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v145 = tt.load %v144, %v142, %v13 : tensor<32x256x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v146 = tt.expand_dims %v117 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v147 = tt.broadcast %v146 : tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>> -> tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v148 = arith.mulf %v145, %v147 : tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
%v149 = "tt.reduce"(%v148) <{axis = 0 : i32}> ({
^bb0(%v150: f32, %v151: f32):
%v152 = arith.addf %v150, %v151 : f32
tt.reduce.return %v152 : f32
}) : (tensor<32x256xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v153 = arith.cmpf oeq, %v120, %v12 : f32
%v154 = tt.splat %v120 : f32 -> tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v155 = arith.divf %v149, %v154 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v156 = tt.splat %v153 : i1 -> tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v157 = arith.select %v156, %v11, %v155 : tensor<256xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>, tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v158 = arith.muli %v74, %v6 : i64
%v159 = arith.extsi %v34 : i32 to i64
%v160 = arith.muli %v159, %v7 : i64
%v161 = arith.addi %v158, %v160 : i64
%v162 = arith.extsi %v67 : tensor<256xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v163 = tt.splat %v161 : i64 -> tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v164 = arith.addi %v163, %v162 : tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v165 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v166 = tt.addptr %v165, %v164 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<256xi64, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v167 = arith.truncf %v157 : tensor<256xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> to tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>>
%v168 = ttg.convert_layout %v167 : tensor<256xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>}>> -> tensor<256xbf16, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v166, %v168, %v73 : tensor<256x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
