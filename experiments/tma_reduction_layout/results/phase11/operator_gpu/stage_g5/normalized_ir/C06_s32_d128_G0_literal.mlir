module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: i32, %v6: f32, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: i64 {tt.divisibility = 16 : i32}, %v10: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v11 = arith.constant 1023 : i32
%v12 = arith.constant dense<128> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v13 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v14 = arith.constant 0.000000e+00 : f32
%v15 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v16 = arith.constant dense<128> : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v17 = arith.constant dense<128> : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v18 = arith.constant 4096 : i32
%v19 = arith.constant 131072 : i64
%v20 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v21 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v22 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v23 = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v24 = arith.constant 1024 : i64
%v25 = arith.constant 32 : i32
%v26 = arith.constant 1024 : i32
%v27 = arith.constant 1 : i32
%v28 = arith.constant 2 : i32
%v29 = arith.constant 0 : i32
%v30 = tt.get_program_id x : i32
%v31 = tt.get_program_id y : i32
%v32 = tt.addptr %v10, %v5 : !tt.ptr<i32>, i32
%v33 = tt.load %v32 : !tt.ptr<i32>
%v34 = arith.cmpi sge, %v30, %v33 : i32
cf.cond_br %v34, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v35:2 = scf.while (%v36 = %v29, %v37 = %v5) : (i32, i32) -> (i32, i32) {
%v38 = arith.cmpi slt, %v36, %v37 : i32
scf.condition(%v38) %v36, %v37 : i32, i32
} do {
^bb0(%v36: i32, %v37: i32):
%v39 = arith.addi %v36, %v37 : i32
%v40 = arith.divsi %v39, %v28 : i32
%v41 = tt.addptr %v10, %v40 : !tt.ptr<i32>, i32
%v42 = tt.load %v41 : !tt.ptr<i32>
%v43 = arith.cmpi sle, %v42, %v30 : i32
%v44 = arith.select %v43, %v37, %v40 : i32
%v45 = scf.if %v43 -> (i32) {
%v46 = arith.addi %v40, %v27 : i32
scf.yield %v46 : i32
} else {
scf.yield %v36 : i32
}
scf.yield %v45, %v44 : i32, i32
}
%v47 = arith.subi %v35#0, %v27 : i32
%v48 = tt.addptr %v4, %v47 : !tt.ptr<i32>, i32
%v49 = tt.load %v48 : !tt.ptr<i32>
%v50 = arith.cmpi sle, %v49, %v29 : i32
cf.cond_br %v50, ^bb1, ^bb3
^bb3:
%v51 = arith.addi %v49, %v11 : i32
%v52 = arith.divsi %v51, %v26 : i32
%v53 = arith.muli %v52, %v25 : i32
%v54 = arith.addi %v49, %v53 : i32
%v55 = arith.subi %v54, %v27 : i32
%v56 = arith.divsi %v55, %v53 : i32
%v57 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v58 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v59 = tt.splat %v56 : i32 -> tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v60 = arith.cmpi slt, %v57, %v59 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = tt.splat %v56 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v62 = arith.cmpi slt, %v58, %v61 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v63 = arith.extsi %v30 : i32 to i64
%v64 = arith.muli %v63, %v24 : i64
%v65 = arith.muli %v31, %v25 : i32
%v66 = arith.extsi %v65 : i32 to i64
%v67 = arith.addi %v64, %v66 : i64
%v68 = tt.addptr %v2, %v67 : !tt.ptr<f32>, i64
%v69 = tt.splat %v68 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v70 = tt.addptr %v69, %v57 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = tt.load %v70, %v60, %v23 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = tt.splat %v68 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v73 = tt.addptr %v72, %v58 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v74 = tt.load %v73, %v62, %v22 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v75 = tt.addptr %v3, %v67 : !tt.ptr<f32>, i64
%v76 = tt.splat %v75 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v77 = tt.addptr %v76, %v57 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v78 = tt.load %v77, %v60, %v21 : tensor<32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v79 = tt.splat %v75 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v80 = tt.addptr %v79, %v58 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v81 = tt.load %v80, %v62, %v20 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v82 = arith.cmpf ogt, %v78, %v21 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v83 = arith.andi %v60, %v82 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v84 = arith.cmpf une, %v71, %v23 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v85 = arith.andi %v83, %v84 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v86 = arith.cmpf ogt, %v81, %v20 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v87 = arith.andi %v62, %v86 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v88 = arith.cmpf une, %v74, %v22 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v89 = arith.andi %v87, %v88 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v90 = arith.select %v85, %v71, %v23 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v91 = arith.select %v89, %v74, %v22 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v92 = "tt.reduce"(%v90) <{axis = 0 : i32}> ({
^bb0(%v93: f32, %v94: f32):
%v95 = arith.maxnumf %v93, %v94 : f32
tt.reduce.return %v95 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v96 = tt.splat %v92 : f32 -> tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v97 = arith.subf %v90, %v96 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v98 = math.exp %v97 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v99 = arith.select %v85, %v98, %v21 : tensor<32xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v100 = tt.splat %v92 : f32 -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v101 = arith.subf %v91, %v100 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v102 = math.exp %v101 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v103 = arith.select %v89, %v102, %v20 : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v104 = arith.mulf %v78, %v99 : tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v105 = "tt.reduce"(%v104) <{axis = 0 : i32}> ({
^bb0(%v106: f32, %v107: f32):
%v108 = arith.addf %v106, %v107 : f32
tt.reduce.return %v108 : f32
}) : (tensor<32xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v109 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v110 = arith.muli %v63, %v19 : i64
%v111 = arith.muli %v31, %v18 : i32
%v112 = arith.extsi %v111 : i32 to i64
%v113 = arith.addi %v110, %v112 : i64
%v114 = tt.expand_dims %v58 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v115 = arith.muli %v114, %v17 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v116 = arith.extsi %v115 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v117 = tt.splat %v113 : i64 -> tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v118 = arith.addi %v117, %v116 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = tt.expand_dims %v109 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v120 = arith.extsi %v119 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v121 = tt.broadcast %v118 : tensor<32x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v122 = tt.broadcast %v120 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v123 = arith.addi %v121, %v122 : tensor<32x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v124 = tt.expand_dims %v62 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v125 = arith.cmpi slt, %v119, %v16 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = tt.broadcast %v124 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = tt.broadcast %v125 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v128 = arith.andi %v126, %v127 : tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v129 = tt.splat %v1 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v130 = tt.addptr %v129, %v123 : tensor<32x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v131 = tt.load %v130, %v128, %v15 : tensor<32x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v132 = tt.expand_dims %v103 {axis = 1 : i32} : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v133 = tt.broadcast %v132 : tensor<32x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v134 = arith.mulf %v131, %v133 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v135 = "tt.reduce"(%v134) <{axis = 0 : i32}> ({
^bb0(%v136: f32, %v137: f32):
%v138 = arith.addf %v136, %v137 : f32
tt.reduce.return %v138 : f32
}) : (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v139 = arith.cmpf oeq, %v105, %v14 : f32
%v140 = tt.splat %v105 : f32 -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v141 = arith.divf %v135, %v140 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v142 = tt.splat %v139 : i1 -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v143 = arith.select %v142, %v13, %v141 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v144 = arith.truncf %v143 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v145 = ttg.convert_layout %v144 : tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v146 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v147 = arith.muli %v63, %v7 : i64
%v148 = arith.extsi %v31 : i32 to i64
%v149 = arith.muli %v148, %v8 : i64
%v150 = arith.addi %v147, %v149 : i64
%v151 = arith.extsi %v146 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v152 = tt.splat %v150 : i64 -> tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v153 = arith.addi %v152, %v151 : tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v154 = arith.cmpi slt, %v146, %v12 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v155 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v156 = tt.addptr %v155, %v153 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v156, %v145, %v154 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
