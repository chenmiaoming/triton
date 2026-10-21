module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v4: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %v5: i32, %v6: f32, %v7: i64 {tt.divisibility = 16 : i32}, %v8: i64 {tt.divisibility = 16 : i32}, %v9: i64 {tt.divisibility = 16 : i32}, %v10: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v11 = arith.constant 0 : i32
%v12 = arith.constant 2 : i32
%v13 = arith.constant 1 : i32
%v14 = arith.constant 512 : i32
%v15 = arith.constant 32 : i32
%v16 = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v17 = arith.constant dense<1> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v18 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v19 = arith.constant 512 : i64
%v20 = arith.constant 16 : i32
%v21 = arith.constant dense<0xFF800000> : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v22 = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v23 = arith.constant 65536 : i64
%v24 = arith.constant 2048 : i32
%v25 = arith.constant dense<128> : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v26 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v27 = arith.constant 0.000000e+00 : f32
%v28 = arith.constant dense<0.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v29 = arith.constant 511 : i32
%v30 = tt.get_program_id x : i32
%v31 = tt.get_program_id y : i32
%v32 = tt.addptr %v10, %v5 : !tt.ptr<i32>, i32
%v33 = tt.load %v32 : !tt.ptr<i32>
%v34 = arith.cmpi sge, %v30, %v33 : i32
cf.cond_br %v34, ^bb1, ^bb2
^bb1:
tt.return
^bb2:
%v35:2 = scf.while (%v36 = %v11, %v37 = %v5) : (i32, i32) -> (i32, i32) {
%v38 = arith.cmpi slt, %v36, %v37 : i32
scf.condition(%v38) %v36, %v37 : i32, i32
} do {
^bb0(%v36: i32, %v37: i32):
%v39 = arith.addi %v36, %v37 : i32
%v40 = arith.divsi %v39, %v12 : i32
%v41 = tt.addptr %v10, %v40 : !tt.ptr<i32>, i32
%v42 = tt.load %v41 : !tt.ptr<i32>
%v43 = arith.cmpi sle, %v42, %v30 : i32
%v44 = arith.select %v43, %v37, %v40 : i32
%v45 = scf.if %v43 -> (i32) {
%v46 = arith.addi %v40, %v13 : i32
scf.yield %v46 : i32
} else {
scf.yield %v36 : i32
}
scf.yield %v45, %v44 : i32, i32
}
%v47 = arith.subi %v35#0, %v13 : i32
%v48 = tt.addptr %v4, %v47 : !tt.ptr<i32>, i32
%v49 = tt.load %v48 : !tt.ptr<i32>
%v50 = arith.cmpi sle, %v49, %v11 : i32
cf.cond_br %v50, ^bb1, ^bb3
^bb3:
%v51 = arith.addi %v49, %v29 : i32
%v52 = arith.divsi %v51, %v14 : i32
%v53 = arith.muli %v52, %v15 : i32
%v54 = arith.addi %v49, %v53 : i32
%v55 = arith.subi %v54, %v13 : i32
%v56 = arith.divsi %v55, %v53 : i32
%v57 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v58 = tt.splat %v56 : i32 -> tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v59 = arith.cmpi slt, %v57, %v58 : tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v60 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v61 = arith.cmpi slt, %v60, %v16 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v62 = arith.select %v61, %v17, %v18 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v63 = arith.cmpi ne, %v62, %v18 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v64 = arith.extsi %v30 : i32 to i64
%v65 = arith.muli %v64, %v19 : i64
%v66 = arith.muli %v31, %v20 : i32
%v67 = arith.extsi %v66 : i32 to i64
%v68 = arith.addi %v65, %v67 : i64
%v69 = arith.extsi %v57 : tensor<16xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<16xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v70 = tt.splat %v68 : i64 -> tensor<16xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = arith.addi %v70, %v69 : tensor<16xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = tt.splat %v2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.addptr %v72, %v71 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.load %v73, %v59, %v21 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75 = tt.splat %v3 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v76 = tt.addptr %v75, %v71 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xi64, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v77 = tt.load %v76, %v59, %v22 : tensor<16x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v78 = arith.cmpf ogt, %v77, %v22 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v79 = arith.andi %v59, %v78 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v80 = arith.cmpf une, %v74, %v21 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v81 = arith.andi %v79, %v80 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v82 = arith.select %v81, %v74, %v21 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v83 = "tt.reduce"(%v82) <{axis = 0 : i32}> ({
^bb0(%v84: f32, %v85: f32):
%v86 = arith.maxnumf %v84, %v85 : f32
tt.reduce.return %v86 : f32
}) : (tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v87 = tt.splat %v83 : f32 -> tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v88 = arith.subf %v82, %v87 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v89 = math.exp %v88 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v90 = arith.select %v81, %v89, %v22 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v91 = arith.mulf %v77, %v90 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v92 = "tt.reduce"(%v91) <{axis = 0 : i32}> ({
^bb0(%v93: f32, %v94: f32):
%v95 = arith.addf %v93, %v94 : f32
tt.reduce.return %v95 : f32
}) : (tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
%v96 = arith.muli %v64, %v23 : i64
%v97 = arith.muli %v31, %v24 : i32
%v98 = arith.extsi %v97 : i32 to i64
%v99 = arith.addi %v96, %v98 : i64
%v100 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v101 = tt.expand_dims %v100 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v102 = arith.muli %v101, %v25 : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v103 = arith.extsi %v102 : tensor<16x1xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v104 = tt.splat %v99 : i64 -> tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v105 = arith.addi %v104, %v103 : tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v106 = tt.expand_dims %v60 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v107 = arith.extsi %v106 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v108 = tt.broadcast %v105 : tensor<16x1xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v109 = tt.broadcast %v107 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v110 = arith.addi %v108, %v109 : tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v111 = ttg.convert_layout %v59 : tensor<16xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v112 = tt.expand_dims %v111 {axis = 1 : i32} : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v113 = tt.expand_dims %v63 {axis = 0 : i32} : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v114 = tt.broadcast %v112 : tensor<16x1xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v115 = tt.broadcast %v113 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v116 = arith.andi %v114, %v115 : tensor<16x128xi1, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v117 = tt.splat %v1 : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v118 = tt.addptr %v117, %v110 : tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<16x128xi64, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v119 = tt.load %v118, %v116, %v26 : tensor<16x128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v120 = ttg.convert_layout %v90 : tensor<16xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v121 = tt.expand_dims %v120 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<16x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v122 = tt.broadcast %v121 : tensor<16x1xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v123 = arith.mulf %v119, %v122 : tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v124 = "tt.reduce"(%v123) <{axis = 0 : i32}> ({
^bb0(%v125: f32, %v126: f32):
%v127 = arith.addf %v125, %v126 : f32
tt.reduce.return %v127 : f32
}) : (tensor<16x128xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v128 = arith.cmpf oeq, %v92, %v27 : f32
%v129 = tt.splat %v92 : f32 -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v130 = arith.divf %v124, %v129 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v131 = tt.splat %v128 : i1 -> tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v132 = arith.select %v131, %v28, %v130 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v133 = arith.muli %v64, %v7 : i64
%v134 = arith.extsi %v31 : i32 to i64
%v135 = arith.muli %v134, %v8 : i64
%v136 = arith.addi %v133, %v135 : i64
%v137 = arith.extsi %v60 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v138 = tt.splat %v136 : i64 -> tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v139 = arith.addi %v138, %v137 : tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v140 = ttg.convert_layout %v63 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v141 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v142 = tt.addptr %v141, %v139 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v143 = ttg.convert_layout %v142 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v144 = ttg.convert_layout %v132 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v145 = arith.truncf %v144 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v143, %v145, %v140 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
