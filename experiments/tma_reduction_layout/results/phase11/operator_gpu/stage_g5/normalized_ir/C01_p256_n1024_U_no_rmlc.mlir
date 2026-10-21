module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v4: i32 {tt.divisibility = 16 : i32}, %v5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v6 = arith.constant 128 : i32
%v7 = arith.constant 0 : i32
%v8 = arith.constant 32 : i32
%v9 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v10 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v11 = tt.get_program_id x : i32
%v12 = arith.muli %v11, %v6 : i32
%v13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v14 = tt.splat %v12 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v15 = arith.addi %v14, %v13 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v17 = tt.splat %v4 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v18 = ttg.convert_layout %v15 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
%v19 = tt.expand_dims %v18 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>>
%v20 = ttg.convert_layout %v19 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v21 = tt.splat %v5 : i32 -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v22 = arith.cmpi slt, %v20, %v21 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v23 = tt.broadcast %v22 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v24 = tt.splat %v5 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v25 = tt.broadcast %v20 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v26 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v27 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v28:2 = scf.for %v29 = %v7 to %v4 step %v8 iter_args(%v30 = %v9, %v31 = %v9) -> (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) : i32 {
%v32 = tt.splat %v29 : i32 -> tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v33 = arith.addi %v32, %v16 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v34 = ttg.convert_layout %v33 : tensor<32xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>>
%v35 = tt.expand_dims %v34 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
%v36 = ttg.convert_layout %v35 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v37 = arith.cmpi slt, %v36, %v17 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v38 = tt.broadcast %v37 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v39 = ttg.convert_layout %v38 : tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v40 = arith.andi %v39, %v23 : tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v41 = arith.muli %v36, %v24 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v42 = tt.broadcast %v41 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v43 = ttg.convert_layout %v42 : tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v44 = arith.addi %v43, %v25 : tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v45 = tt.addptr %v26, %v44 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v46 = ttg.convert_layout %v45 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v47 = ttg.convert_layout %v40 : tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v48 = tt.load %v46, %v47, %v10 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v49 = ttg.convert_layout %v48 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v50 = arith.extf %v49 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> to tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v51 = arith.addf %v30, %v50 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v52 = tt.addptr %v27, %v44 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v53 = ttg.convert_layout %v52 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v54 = tt.load %v53, %v47, %v10 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v55 = ttg.convert_layout %v54 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v56 = arith.extf %v55 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>> to tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
%v57 = arith.addf %v31, %v56 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
scf.yield %v51, %v57 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>, tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>
}
%v58 = "tt.reduce"(%v28#0) <{axis = 0 : i32}> ({
^bb0(%v59: f32, %v60: f32):
%v61 = arith.addf %v59, %v60 : f32
tt.reduce.return %v61 : f32
}) : (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>>
%v62 = ttg.convert_layout %v58 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v63 = "tt.reduce"(%v28#1) <{axis = 0 : i32}> ({
^bb0(%v64: f32, %v65: f32):
%v66 = arith.addf %v64, %v65 : f32
tt.reduce.return %v66 : f32
}) : (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>>
%v67 = ttg.convert_layout %v63 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v68 = tt.splat %v5 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v69 = arith.cmpi slt, %v15, %v68 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v70 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v71 = tt.addptr %v70, %v15 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v72 = arith.truncf %v62 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v71, %v72, %v69 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v73 = tt.splat %v3 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v74 = tt.addptr %v73, %v15 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v75 = arith.truncf %v67 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v74, %v75, %v69 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
