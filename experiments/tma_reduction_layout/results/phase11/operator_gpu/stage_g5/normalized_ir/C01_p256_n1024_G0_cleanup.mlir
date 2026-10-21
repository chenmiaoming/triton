module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
tt.func public @ENTRY(%v0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %v4: i32 {tt.divisibility = 16 : i32}, %v5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%v6 = arith.constant 128 : i32
%v7 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v8 = arith.constant 0 : i32
%v9 = arith.constant 32 : i32
%v10 = arith.constant dense<0.000000e+00> : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v11 = tt.get_program_id x : i32
%v12 = arith.muli %v11, %v6 : i32
%v13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v14 = tt.splat %v12 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v15 = arith.addi %v14, %v13 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v16:2 = scf.for %v17 = %v8 to %v4 step %v9 iter_args(%v18 = %v7, %v19 = %v7) -> (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) : i32 {
%v20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v21 = tt.splat %v17 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v22 = arith.addi %v21, %v20 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v23 = tt.expand_dims %v22 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v24 = tt.splat %v4 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v25 = arith.cmpi slt, %v23, %v24 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v26 = tt.expand_dims %v15 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v27 = tt.splat %v5 : i32 -> tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v28 = arith.cmpi slt, %v26, %v27 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v29 = tt.broadcast %v25 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v30 = tt.broadcast %v28 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v31 = arith.andi %v29, %v30 : tensor<32x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v32 = tt.splat %v5 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v33 = arith.muli %v23, %v32 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v34 = tt.broadcast %v33 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v35 = tt.broadcast %v26 : tensor<1x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v36 = arith.addi %v34, %v35 : tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v37 = tt.splat %v0 : !tt.ptr<bf16> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v38 = tt.addptr %v37, %v36 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v39 = tt.load %v38, %v31, %v10 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v40 = arith.extf %v39 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v41 = arith.addf %v18, %v40 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v42 = tt.splat %v1 : !tt.ptr<bf16> -> tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v43 = tt.addptr %v42, %v36 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v44 = tt.load %v43, %v31, %v10 : tensor<32x128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v45 = arith.extf %v44 : tensor<32x128xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> to tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
%v46 = arith.addf %v19, %v45 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
scf.yield %v41, %v46 : tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
}
%v47 = "tt.reduce"(%v16#0) <{axis = 0 : i32}> ({
^bb0(%v48: f32, %v49: f32):
%v50 = arith.addf %v48, %v49 : f32
tt.reduce.return %v50 : f32
}) : (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v51 = "tt.reduce"(%v16#1) <{axis = 0 : i32}> ({
^bb0(%v52: f32, %v53: f32):
%v54 = arith.addf %v52, %v53 : f32
tt.reduce.return %v54 : f32
}) : (tensor<32x128xf32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v55 = tt.splat %v5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v56 = arith.cmpi slt, %v15, %v55 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v57 = ttg.convert_layout %v56 : tensor<128xi1, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v58 = tt.splat %v2 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v59 = tt.addptr %v58, %v15 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v60 = ttg.convert_layout %v59 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v61 = ttg.convert_layout %v47 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v62 = arith.truncf %v61 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v60, %v62, %v57 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v63 = tt.splat %v3 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v64 = tt.addptr %v63, %v15 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
%v65 = ttg.convert_layout %v64 : tensor<128x!tt.ptr<bf16>, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v66 = ttg.convert_layout %v51 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
%v67 = arith.truncf %v66 : tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> to tensor<128xbf16, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.store %v65, %v67, %v57 : tensor<128x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
tt.return
}
}
