#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @negative_zero_accumulator(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %6 = arith.muli %arg6, %c128_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = arith.addi %7, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %11 = tt.addptr %arg3, %10 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %12 = tt.load %11 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %13 = tt.reshape %12 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %14 = tt.trans %13 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %15 = tt.reshape %14 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %16 = ttg.convert_layout %15 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %17 = "tt.reduce"(%16) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %19 = arith.addf %arg8, %arg9 : f32
        tt.reduce.return %19 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %18 = arith.addf %arg7, %17 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %18 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %6 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %5 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 4], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @positive_zero_accumulator(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst_0) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %8 = arith.muli %arg6, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %10 = arith.addi %9, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %12 = tt.broadcast %11 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %13 = tt.addptr %arg3, %12 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %14 = tt.load %13 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %15 = tt.reshape %14 : tensor<32x128xf32, #blocked> -> tensor<32x1x32x4xf32, #blocked3>
      %16 = tt.trans %15 {order = array<i32: 0, 2, 1, 3>} : tensor<32x1x32x4xf32, #blocked3> -> tensor<32x32x1x4xf32, #blocked4>
      %17 = tt.reshape %16 efficient_layout : tensor<32x32x1x4xf32, #blocked4> -> tensor<32x32x4xf32, #ttg.slice<{dim = 2, parent = #blocked4}>>
      %18 = ttg.convert_layout %17 : tensor<32x32x4xf32, #ttg.slice<{dim = 2, parent = #blocked4}>> -> tensor<32x32x4xf32, #blocked2>
      %19 = "tt.reduce"(%18) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %21 = arith.addf %arg8, %arg9 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %20 = arith.addf %arg7, %19 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %20 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %8 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %8 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.addf %5, %cst : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %7 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unary_loop_update(%arg0: tensor<16x256x!tt.ptr<f32>, #blocked>, %arg1: tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %1 = tt.load %arg0 : tensor<16x256x!tt.ptr<f32>, #blocked>
      %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %4 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %4 : f32
      }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %3 = arith.negf %2 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %3 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    tt.store %arg1, %0 : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 4], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 4], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [1, 0, 0], [2, 0, 0], [4, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @row_major_register_grouping(%arg0: tensor<8x256x!tt.ptr<f32>, #blocked>, %arg1: tensor<8x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %cst_0) -> (tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>)  : i32 {
      %4 = tt.load %arg0 : tensor<8x256x!tt.ptr<f32>, #blocked>
      %5 = tt.reshape %4 : tensor<8x256xf32, #blocked> -> tensor<8x2x32x4xf32, #blocked2>
      %6 = tt.trans %5 {order = array<i32: 0, 2, 1, 3>} : tensor<8x2x32x4xf32, #blocked2> -> tensor<8x32x2x4xf32, #blocked3>
      %7 = tt.reshape %6 efficient_layout : tensor<8x32x2x4xf32, #blocked3> -> tensor<8x32x8xf32, #linear>
      %8 = ttg.convert_layout %7 : tensor<8x32x8xf32, #linear> -> tensor<8x32x8xf32, #blocked1>
      %9 = "tt.reduce"(%8) <{axis = 2 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %11 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %11 : f32
      }) : (tensor<8x32x8xf32, #blocked1>) -> tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      %10 = arith.addf %arg4, %9 : tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      scf.yield %10 : tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    }
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<8x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %2 = ttg.convert_layout %1 : tensor<8xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = arith.addf %2, %cst : tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.store %arg1, %3 : tensor<8x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4, 1, 1, 1], threadsPerWarp = [32, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [0, 3, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1, 1, 1], threadsPerWarp = [32, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [0, 3, 1, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_rows_when_reordering_registers(%arg0: tensor<256x2x!tt.ptr<f32>, #blocked>, %arg1: tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %cst_0) -> (tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>)  : i32 {
      %4 = tt.load %arg0 : tensor<256x2x!tt.ptr<f32>, #blocked>
      %5 = tt.reshape %4 : tensor<256x2xf32, #blocked> -> tensor<256x2x1x1xf32, #blocked2>
      %6 = tt.trans %5 {order = array<i32: 0, 2, 1, 3>} : tensor<256x2x1x1xf32, #blocked2> -> tensor<256x1x2x1xf32, #blocked3>
      %7 = tt.reshape %6 efficient_layout : tensor<256x1x2x1xf32, #blocked3> -> tensor<256x1x2xf32, #ttg.slice<{dim = 3, parent = #blocked3}>>
      %8 = ttg.convert_layout %7 : tensor<256x1x2xf32, #ttg.slice<{dim = 3, parent = #blocked3}>> -> tensor<256x1x2xf32, #blocked1>
      %9 = "tt.reduce"(%8) <{axis = 2 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %11 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %11 : f32
      }) : (tensor<256x1x2xf32, #blocked1>) -> tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      %10 = arith.addf %arg4, %9 : tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      scf.yield %10 : tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    }
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<256x1xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %2 = ttg.convert_layout %1 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = arith.addf %2, %cst : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.store %arg1, %3 : tensor<256x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_outside_accumulator_loop(%arg0: tensor<32x2x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32  : i32 {
      %0 = tt.load %arg0 : tensor<32x2x!tt.ptr<f32>, #blocked>
      %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
      ^bb0(%arg4: f32, %arg5: f32):
        %3 = arith.addf %arg4, %arg5 : f32
        tt.reduce.return %3 : f32
      }) : (tensor<32x2xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %2 = scf.for %arg4 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg5 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
        %3 = arith.addf %arg5, %1 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      }
      tt.store %arg1, %2 : tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 4, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @slice_layout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #ttg.slice<{dim = 2, parent = #blocked}>> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>)  : i32 {
      %5 = arith.muli %arg6, %c128_i32 : i32
      %6 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %7 = arith.addi %6, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<32x128xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %10 = tt.addptr %arg3, %9 : tensor<32x128x!tt.ptr<f32>, #ttg.slice<{dim = 2, parent = #blocked}>>, tensor<32x128xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
      %11 = tt.load %10 : tensor<32x128x!tt.ptr<f32>, #ttg.slice<{dim = 2, parent = #blocked}>>
      %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %14 = arith.addf %arg8, %arg9 : f32
        tt.reduce.return %14 : f32
      }) : (tensor<32x128xf32, #ttg.slice<{dim = 2, parent = #blocked}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      %13 = arith.addf %arg7, %12 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
      scf.yield %13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    }
    %4 = ttg.convert_layout %3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %4 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mma_layout(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #mma> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %5 = arith.muli %arg6, %c128_i32 : i32
      %6 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %7 = arith.addi %6, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #mma> -> tensor<32x128xi32, #mma>
      %10 = tt.addptr %arg3, %9 : tensor<32x128x!tt.ptr<f32>, #mma>, tensor<32x128xi32, #mma>
      %11 = tt.load %10 : tensor<32x128x!tt.ptr<f32>, #mma>
      %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %14 = arith.addf %arg8, %arg9 : f32
        tt.reduce.return %14 : f32
      }) : (tensor<32x128xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %13 = arith.addf %arg7, %12 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      scf.yield %13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    }
    %4 = ttg.convert_layout %3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32xf32, #blocked>
    tt.store %arg5, %4 : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @max_reduce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0xFF800000> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %6 = arith.muli %arg6, %c128_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = arith.addi %7, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %11 = tt.addptr %arg3, %10 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %12 = tt.load %11 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %13 = tt.reshape %12 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %14 = tt.trans %13 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %15 = tt.reshape %14 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %16 = ttg.convert_layout %15 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %17 = "tt.reduce"(%16) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %19 = arith.maximumf %arg8, %arg9 : f32
        tt.reduce.return %19 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %18 = arith.maximumf %arg7, %17 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %18 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %6 = arith.maximumf %arg6, %arg7 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %5 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @max_reduce_zero_int_accumulator(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst_0) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %8 = arith.muli %arg6, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %10 = arith.addi %9, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %12 = tt.broadcast %11 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %13 = tt.addptr %arg3, %12 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %14 = tt.load %13 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %15 = tt.reshape %14 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %16 = tt.trans %15 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %17 = tt.reshape %16 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %18 = ttg.convert_layout %17 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %19 = "tt.reduce"(%18) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %21 = arith.maximumf %arg8, %arg9 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %20 = arith.maximumf %arg7, %19 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %20 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %8 = arith.maximumf %arg6, %arg7 : f32
      tt.reduce.return %8 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.maximumf %5, %cst : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %7 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @min_reduce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0x7F800000> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %6 = arith.muli %arg6, %c128_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = arith.addi %7, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %11 = tt.addptr %arg3, %10 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %12 = tt.load %11 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %13 = tt.reshape %12 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %14 = tt.trans %13 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %15 = tt.reshape %14 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %16 = ttg.convert_layout %15 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %17 = "tt.reduce"(%16) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %19 = arith.minimumf %arg8, %arg9 : f32
        tt.reduce.return %19 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %18 = arith.minimumf %arg7, %17 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %18 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %6 = arith.minimumf %arg6, %arg7 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %5 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @min_reduce_zero_int_accumulator(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0x7F800000> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst_0) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %8 = arith.muli %arg6, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %10 = arith.addi %9, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %12 = tt.broadcast %11 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %13 = tt.addptr %arg3, %12 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %14 = tt.load %13 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %15 = tt.reshape %14 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %16 = tt.trans %15 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %17 = tt.reshape %16 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %18 = ttg.convert_layout %17 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %19 = "tt.reduce"(%18) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %21 = arith.minimumf %arg8, %arg9 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %20 = arith.minimumf %arg7, %19 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %20 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %8 = arith.minimumf %arg6, %arg7 : f32
      tt.reduce.return %8 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.minimumf %5, %cst : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %7 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mul_reduce(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %6 = arith.muli %arg6, %c128_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = arith.addi %7, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %11 = tt.addptr %arg3, %10 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %12 = tt.load %11 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %13 = tt.reshape %12 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %14 = tt.trans %13 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %15 = tt.reshape %14 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %16 = ttg.convert_layout %15 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %17 = "tt.reduce"(%16) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %19 = arith.mulf %arg8, %arg9 : f32
        tt.reduce.return %19 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %18 = arith.mulf %arg7, %17 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %18 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %6 = arith.mulf %arg6, %arg7 : f32
      tt.reduce.return %6 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %5 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [4, 1, 1, 1], order = [3, 1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [4, 0, 0], [8, 0, 0], [16, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [[1, 0, 0], [2, 0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mul_reduce_zero_int_accumulator(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst_0) -> (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>)  : i32 {
      %8 = arith.muli %arg6, %c128_i32 : i32
      %9 = tt.splat %8 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %10 = arith.addi %9, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %12 = tt.broadcast %11 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %13 = tt.addptr %arg3, %12 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %14 = tt.load %13 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %15 = tt.reshape %14 : tensor<32x128xf32, #blocked> -> tensor<32x2x32x2xf32, #blocked3>
      %16 = tt.trans %15 {order = array<i32: 0, 2, 1, 3>} : tensor<32x2x32x2xf32, #blocked3> -> tensor<32x32x2x2xf32, #blocked4>
      %17 = tt.reshape %16 efficient_layout : tensor<32x32x2x2xf32, #blocked4> -> tensor<32x32x4xf32, #linear>
      %18 = ttg.convert_layout %17 : tensor<32x32x4xf32, #linear> -> tensor<32x32x4xf32, #blocked2>
      %19 = "tt.reduce"(%18) <{axis = 2 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %21 = arith.mulf %arg8, %arg9 : f32
        tt.reduce.return %21 : f32
      }) : (tensor<32x32x4xf32, #blocked2>) -> tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      %20 = arith.mulf %arg7, %19 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
      scf.yield %20 : tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    }
    %4 = "tt.reduce"(%3) <{axis = 1 : i32}> ({
    ^bb0(%arg6: f32, %arg7: f32):
      %8 = arith.mulf %arg6, %arg7 : f32
      tt.reduce.return %8 : f32
    }) : (tensor<32x32xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %5 = ttg.convert_layout %4 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = arith.mulf %5, %cst : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = ttg.convert_layout %6 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %7 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @remains_unchanged(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg6 = %0 to %arg4 step %1 iter_args(%arg7 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %5 = arith.muli %arg6, %c128_i32 : i32
      %6 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %7 = arith.addi %6, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %10 = tt.addptr %arg3, %9 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %11 = tt.load %10 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %12 = arith.mulf %11, %11 : tensor<32x128xf32, #blocked>
      %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
      ^bb0(%arg8: f32, %arg9: f32):
        %15 = arith.maximumf %arg8, %arg9 : f32
        tt.reduce.return %15 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.maximumf %arg7, %13 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %4 = ttg.convert_layout %3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %4 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @optimize_view_layout(%arg0: tensor<8x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = tt.reshape %arg0 allow_reorder efficient_layout : tensor<8x128xf32, #blocked> -> tensor<64x16xf32, #blocked2>
    %1 = ttg.convert_layout %0 : tensor<64x16xf32, #blocked2> -> tensor<64x16xf32, #blocked1>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %3 = arith.maximumf %arg1, %arg2 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<64x16xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %2 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @optimize_view_layout_same_shape(%arg0: tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.reshape %arg0 allow_reorder efficient_layout : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked1>
    %1 = ttg.convert_layout %0 : tensor<64x16xf32, #blocked1> -> tensor<64x16xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %3 = arith.maximumf %arg1, %arg2 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %2 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func public @reduce_for_arg(%arg0: tensor<64x128xf32, #blocked>, %arg1: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = scf.for %arg2 = %c0_i32 to %c4096_i32 step %c128_i32 iter_args(%arg3 = %arg0) -> (tensor<64x128xf32, #blocked>)  : i32 {
      %1 = "tt.reduce"(%arg3) <{axis = 1 : i32}> ({
      ^bb0(%arg4: f32, %arg5: f32):
        %7 = arith.maxnumf %arg4, %arg5 : f32
        tt.reduce.return %7 : f32
      }) : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %2 = ttg.convert_layout %1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
      %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
      %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %5 = tt.addptr %4, %3 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      tt.store %5, %2 : tensor<64x!tt.ptr<f32>, #blocked1>
      %6 = arith.addf %arg3, %cst : tensor<64x128xf32, #blocked>
      scf.yield %6 : tensor<64x128xf32, #blocked>
    }
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_square_axis_0(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    tt.return %3 : tensor<64x64xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_square_axis_1(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    tt.return %3 : tensor<64x64xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_warp_broadcast(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<64x1xi32, #blocked> -> tensor<64x1xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked>
    tt.return %3 : tensor<64x1xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_3d_warp(%arg0: tensor<32x2x32xf32, #blocked>, %arg1: tensor<32x2x2xi32, #blocked>) -> tensor<32x2x2xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<32x2x32xf32, #blocked> -> tensor<32x2x32xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<32x2x2xi32, #blocked> -> tensor<32x2x2xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 2 : i32, efficient_layout} : (tensor<32x2x32xf32, #blocked1>, tensor<32x2x2xi32, #blocked1>) -> tensor<32x2x2xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<32x2x2xf32, #blocked1> -> tensor<32x2x2xf32, #blocked>
    tt.return %3 : tensor<32x2x2xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_3d_warp_thread_split(%arg0: tensor<32x4x16xf32, #blocked>, %arg1: tensor<32x4x2xi32, #blocked>) -> tensor<32x4x2xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<32x4x16xf32, #blocked> -> tensor<32x4x16xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<32x4x2xi32, #blocked> -> tensor<32x4x2xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 2 : i32, efficient_layout} : (tensor<32x4x16xf32, #blocked1>, tensor<32x4x2xi32, #blocked1>) -> tensor<32x4x2xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<32x4x2xf32, #blocked1> -> tensor<32x4x2xf32, #blocked>
    tt.return %3 : tensor<32x4x2xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_thread_broadcast(%arg0: tensor<16x64xf32, #blocked>, %arg1: tensor<16x1xi32, #blocked>) -> tensor<16x1xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<16x1xi32, #blocked> -> tensor<16x1xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 1 : i32, efficient_layout} : (tensor<16x64xf32, #blocked1>, tensor<16x1xi32, #blocked1>) -> tensor<16x1xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<16x1xf32, #blocked1> -> tensor<16x1xf32, #blocked>
    tt.return %3 : tensor<16x1xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @set_warp_shuffle_layout_large_source(%arg0: tensor<256x256xf32, #blocked>, %arg1: tensor<256x8xi32, #blocked>) -> tensor<256x8xf32, #blocked> {
    %0 = ttg.convert_layout %arg0 : tensor<256x256xf32, #blocked> -> tensor<256x256xf32, #blocked1>
    %1 = ttg.convert_layout %arg1 : tensor<256x8xi32, #blocked> -> tensor<256x8xi32, #blocked1>
    %2 = tt.gather %0[%1] {axis = 1 : i32, efficient_layout} : (tensor<256x256xf32, #blocked1>, tensor<256x8xi32, #blocked1>) -> tensor<256x8xf32, #blocked1>
    %3 = ttg.convert_layout %2 : tensor<256x8xf32, #blocked1> -> tensor<256x8xf32, #blocked>
    tt.return %3 : tensor<256x8xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @skip_optimize_on_1d_tensor(%arg0: tensor<256xf32, #blocked>, %arg1: tensor<8xi32, #blocked>) -> tensor<8xf32, #blocked> {
    %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<256xf32, #blocked>, tensor<8xi32, #blocked>) -> tensor<8xf32, #blocked>
    tt.return %0 : tensor<8xf32, #blocked>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 1, 32, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 2], threadsPerWarp = [1, 32, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 1, 2, 0]}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @loop_result_multiple_uses(%arg0: tensor<1x64x!tt.ptr<f32>, #blocked>, %arg1: tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %cst_0) -> (tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>)  : i32 {
      %5 = tt.load %arg0 : tensor<1x64x!tt.ptr<f32>, #blocked>
      %6 = tt.reshape %5 : tensor<1x64xf32, #blocked> -> tensor<1x1x32x2xf32, #blocked2>
      %7 = tt.trans %6 {order = array<i32: 0, 2, 1, 3>} : tensor<1x1x32x2xf32, #blocked2> -> tensor<1x32x1x2xf32, #blocked3>
      %8 = tt.reshape %7 efficient_layout : tensor<1x32x1x2xf32, #blocked3> -> tensor<1x32x2xf32, #ttg.slice<{dim = 2, parent = #blocked3}>>
      %9 = ttg.convert_layout %8 : tensor<1x32x2xf32, #ttg.slice<{dim = 2, parent = #blocked3}>> -> tensor<1x32x2xf32, #blocked1>
      %10 = "tt.reduce"(%9) <{axis = 2 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %12 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %12 : f32
      }) : (tensor<1x32x2xf32, #blocked1>) -> tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      %11 = arith.addf %arg4, %10 : tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
      scf.yield %11 : tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    }
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %5 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %5 : f32
    }) : (tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked1}>>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %2 = ttg.convert_layout %1 : tensor<1xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = arith.addf %2, %cst : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = arith.addf %3, %3 : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.store %arg1, %4 : tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @in_loop_accumulator_use(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg3: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}, %arg6: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id y : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = scf.for %arg7 = %0 to %arg4 step %1 iter_args(%arg8 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %5 = arith.muli %arg7, %c128_i32 : i32
      %6 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %7 = arith.addi %6, %2 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %8 = tt.expand_dims %7 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %9 = tt.broadcast %8 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %10 = tt.addptr %arg3, %9 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %11 = tt.load %10 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %12 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg9: f32, %arg10: f32):
        %15 = arith.addf %arg9, %arg10 : f32
        tt.reduce.return %15 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = ttg.convert_layout %arg8 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
      tt.store %arg6, %13 : tensor<32x!tt.ptr<f32>, #blocked1>
      %14 = arith.addf %arg8, %12 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %14 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %4 = ttg.convert_layout %3 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %arg5, %4 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 2], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_instruction_cost(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked1>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked1> to tensor<1x32x64xf32, #blocked1>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %4 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<1x32x64xf32, #blocked1>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = ttg.convert_layout %2 : tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %3 : tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_sum_instruction_cost(%arg0: !tt.tensordesc<1x32x128xbf16>, %arg1: i32) -> tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x128xbf16> -> tensor<1x32x128xbf16, #blocked1>
    %1 = arith.extf %0 : tensor<1x32x128xbf16, #blocked1> to tensor<1x32x128xf32, #blocked1>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %4 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<1x32x128xf32, #blocked1>) -> tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = ttg.convert_layout %2 : tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %3 : tensor<1x128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_multi_load_use(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> (tensor<1x32x64xbf16, #blocked>, tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked> to tensor<1x32x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<1x32x64xf32, #blocked>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %0, %2 : tensor<1x32x64xbf16, #blocked>, tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_multi_ext_use(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> (tensor<1x32x64xf32, #blocked>, tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked> to tensor<1x32x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<1x32x64xf32, #blocked>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1, %2 : tensor<1x32x64xf32, #blocked>, tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_innermost(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked> to tensor<1x32x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 2 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<1x32x64xf32, #blocked>) -> tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
    tt.return %2 : tensor<1x32xf32, #ttg.slice<{dim = 2, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_custom_combiner(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked> to tensor<1x32x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.addf %arg2, %arg3 : f32
      %4 = arith.maximumf %3, %arg3 : f32
      tt.reduce.return %4 : f32
    }) : (tensor<1x32x64xf32, #blocked>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %2 : tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @descriptor_reduce_non_nvidia(%arg0: !tt.tensordesc<1x32x64xbf16>, %arg1: i32) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x32x64xbf16> -> tensor<1x32x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x32x64xbf16, #blocked> to tensor<1x32x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<1x32x64xf32, #blocked>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %2 : tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_load_cost_fallback(%arg0: !tt.tensordesc<1x1024x64xbf16>, %arg1: i32) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.descriptor_load %arg0[%arg1, %arg1, %arg1] : !tt.tensordesc<1x1024x64xbf16> -> tensor<1x1024x64xbf16, #blocked>
    %1 = arith.extf %0 : tensor<1x1024x64xbf16, #blocked> to tensor<1x1024x64xf32, #blocked>
    %2 = "tt.reduce"(%1) <{axis = 1 : i32}> ({
    ^bb0(%arg2: f32, %arg3: f32):
      %3 = arith.maximumf %arg2, %arg3 : f32
      tt.reduce.return %3 : f32
    }) : (tensor<1x1024x64xf32, #blocked>) -> tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %2 : tensor<1x64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

