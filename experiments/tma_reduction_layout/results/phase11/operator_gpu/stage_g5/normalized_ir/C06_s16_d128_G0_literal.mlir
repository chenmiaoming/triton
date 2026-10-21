module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: i32, %v6: f32, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: i64 {tt.divisibility = 16 : i32}, %v10: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v11 = arith.constant 511 : i32
%v12 = arith.constant dense<128> : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v13 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v14 = arith.constant 0.000000e+00 : f32
%v15 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v16 = arith.constant dense<128> : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v17 = arith.constant dense<128> : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v18 = arith.constant 2048 : i32
%v19 = arith.constant 65536 : i64
%v20 = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v21 = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v22 = arith.constant dense<0xFF800000> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v23 = arith.constant dense<0xFF800000> : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v24 = arith.constant 16 : i32
%v25 = arith.constant 512 : i64
%v26 = arith.constant 32 : i32
%v27 = arith.constant 512 : i32
%v28 = arith.constant 1 : i32
%v29 = arith.constant 2 : i32
%v30 = arith.constant 0 : i32
%v31 = tt.get_program_id x : i32
%v32 = tt.get_program_id y : i32
%v33 = tt.addptr %v10, %v5 : !tt.ptr<i32>, i32
%v34 = tt.load %v33 : !tt.ptr<i32>
%v35 = arith.cmpi sge, %v31, %v34 : i32
cf.cond_br %v35, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v36:2 = scf.while (%v37 = %v30, %v38 = %v5) : (i32, i32) -> (i32, i32) {
%v39 = arith.cmpi slt, %v37, %v38 : i32
scf.condition(%v39) %v37, %v38 : i32, i32
} do {
^bb0(%v37: i32, %v38: i32):
%v40 = arith.addi %v37, %v38 : i32
%v41 = arith.divsi %v40, %v29 : i32
%v42 = tt.addptr %v10, %v41 : !tt.ptr<i32>, i32
%v43 = tt.load %v42 : !tt.ptr<i32>
%v44 = arith.cmpi sle, %v43, %v31 : i32
%v45 = arith.select %v44, %v38, %v41 : i32
%v46 = scf.if %v44 -> (i32) {
%v47 = arith.addi %v41, %v28 : i32
scf.yield %v47 : i32
} else {
scf.yield %v37 : i32
}
scf.yield %v46, %v45 : i32, i32
}
%v48 = arith.subi %v36#0, %v28 : i32
%v49 = tt.addptr %v4, %v48 : !tt.ptr<i32>, i32
%v50 = tt.load %v49 : !tt.ptr<i32>
%v51 = arith.cmpi sle, %v50, %v30 : i32
cf.cond_br %v51, ^bb1, ^bb3
^bb3:
%v52 = arith.addi %v50, %v11 : i32
%v53 = arith.divsi %v52, %v27 : i32
%v54 = arith.muli %v53, %v26 : i32
%v55 = arith.addi %v50, %v54 : i32
%v56 = arith.subi %v55, %v28 : i32
%v57 = arith.divsi %v56, %v54 : i32
%v58 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v59 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v60 = tt.splat %v57 : i32 -> tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = arith.cmpi slt, %v58, %v60 : tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = tt.splat %v57 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v63 = arith.cmpi slt, %v59, %v62 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v64 = arith.extsi %v31 : i32 to i64
%v65 = arith.muli %v64, %v25 : i64
%v66 = arith.muli %v32, %v24 : i32
%v67 = arith.extsi %v66 : i32 to i64
%v68 = arith.addi %v65, %v67 : i64
%v69 = tt.addptr %v2, %v68 : !tt.ptr<f32>, i64
%v70 = tt.splat %v69 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = tt.addptr %v70, %v58 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = tt.load %v71, %v61, %v23 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.splat %v69 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v74 = tt.addptr %v73, %v59 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v75 = tt.load %v74, %v63, %v22 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v76 = tt.addptr %v3, %v68 : !tt.ptr<f32>, i64
%v77 = tt.splat %v76 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v78 = tt.addptr %v77, %v58 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v79 = tt.load %v78, %v61, %v21 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v80 = tt.splat %v76 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v81 = tt.addptr %v80, %v59 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v82 = tt.load %v81, %v63, %v20 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v83 = arith.cmpf ogt, %v79, %v21 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v84 = arith.andi %v61, %v83 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v85 = arith.cmpf une, %v72, %v23 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v86 = arith.andi %v84, %v85 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v87 = arith.cmpf ogt, %v82, %v20 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v88 = arith.andi %v63, %v87 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v89 = arith.cmpf une, %v75, %v22 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v90 = arith.andi %v88, %v89 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v91 = arith.select %v86, %v72, %v23 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v92 = arith.select %v90, %v75, %v22 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v93 = "tt.reduce"(%v91) <{axis = 0 : i32}> ({
^bb0(%v94: f32, %v95: f32):
%v96 = arith.maxnumf %v94, %v95 : f32
tt.reduce.return %v96 : f32
}) : (tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v97 = tt.splat %v93 : f32 -> tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v98 = arith.subf %v91, %v97 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v99 = math.exp %v98 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v100 = arith.select %v86, %v99, %v21 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v101 = tt.splat %v93 : f32 -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v102 = arith.subf %v92, %v101 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v103 = math.exp %v102 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v104 = arith.select %v90, %v103, %v20 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v105 = arith.mulf %v79, %v100 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v106 = "tt.reduce"(%v105) <{axis = 0 : i32}> ({
^bb0(%v107: f32, %v108: f32):
%v109 = arith.addf %v107, %v108 : f32
tt.reduce.return %v109 : f32
}) : (tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v110 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v111 = arith.muli %v64, %v19 : i64
%v112 = arith.muli %v32, %v18 : i32
%v113 = arith.extsi %v112 : i32 to i64
%v114 = arith.addi %v111, %v113 : i64
%v115 = tt.expand_dims %v59 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v116 = arith.muli %v115, %v17 : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v117 = arith.extsi %v116 : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v118 = tt.splat %v114 : i64 -> tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = arith.addi %v118, %v117 : tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v120 = tt.expand_dims %v110 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v121 = arith.extsi %v120 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v122 = tt.broadcast %v119 : tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v123 = tt.broadcast %v121 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v124 = arith.addi %v122, %v123 : tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v125 = tt.expand_dims %v63 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v126 = arith.cmpi slt, %v120, %v16 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v127 = tt.broadcast %v125 : tensor<16x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v128 = tt.broadcast %v126 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v129 = arith.andi %v127, %v128 : tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v130 = tt.splat %v1 : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v131 = tt.addptr %v130, %v124 : tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v132 = tt.load %v131, %v129, %v15 : tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v133 = tt.expand_dims %v104 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v134 = tt.broadcast %v133 : tensor<16x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v135 = arith.mulf %v132, %v134 : tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v136 = "tt.reduce"(%v135) <{axis = 0 : i32}> ({
^bb0(%v137: f32, %v138: f32):
%v139 = arith.addf %v137, %v138 : f32
tt.reduce.return %v139 : f32
}) : (tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v140 = arith.cmpf oeq, %v106, %v14 : f32
%v141 = tt.splat %v106 : f32 -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v142 = arith.divf %v136, %v141 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v143 = tt.splat %v140 : i1 -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v144 = arith.select %v143, %v13, %v142 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v145 = arith.truncf %v144 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v146 = ttg.convert_layout %v145 : tensor<128xbf16, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v147 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v148 = arith.muli %v64, %v7 : i64
%v149 = arith.extsi %v32 : i32 to i64
%v150 = arith.muli %v149, %v8 : i64
%v151 = arith.addi %v148, %v150 : i64
%v152 = arith.extsi %v147 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v153 = tt.splat %v151 : i64 -> tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v154 = arith.addi %v153, %v152 : tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v155 = arith.cmpi slt, %v147, %v12 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v156 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v157 = tt.addptr %v156, %v154 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v157, %v146, %v155 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
