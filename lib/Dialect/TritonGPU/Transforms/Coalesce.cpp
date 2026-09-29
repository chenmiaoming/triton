#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOALESCE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// Descriptor load/stores don't need to consider L1 coalescing but the
// destination layout will affect the shared memory load/store generated. So we
// still want to allow vectorization for the src/destination layout up to
// 16bytes.
static Attribute pickDescriptorLoadStoreLayout(int numWarps, int threadsPerWarp,
                                               RankedTensorType type) {
  auto shapePerCTA = triton::gpu::getShapePerCTA(type);
  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;
  int numElemsPerThread = std::max(numElems / numThreads, 1);

  int maxVectorSize = 128 / type.getElementTypeBitWidth();

  int vectorSize = std::min(numElemsPerThread, maxVectorSize);

  // EXPERIMENT ONLY: allow explicit override of descriptor load contiguous sizePerThread
  const char *expEnv = std::getenv("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT");
  if (expEnv && std::strlen(expEnv) > 0 && std::strcmp(expEnv, "default") != 0) {
    char *end = nullptr;
    long forcedVec = std::strtol(expEnv, &end, 10);
    if (*end != '\0' || forcedVec < 1) {
      llvm::report_fatal_error(
          llvm::Twine("candidate invalid: invalid TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT value '") + expEnv + "'");
    }
    // Validation:
    // 1. Must not exceed 128 bits / element bitwidth
    if (forcedVec > maxVectorSize) {
      llvm::report_fatal_error(
          llvm::Twine("candidate invalid: forced vector size ") + llvm::Twine(forcedVec) +
          " exceeds maxVectorSize " + llvm::Twine(maxVectorSize));
    }
    // 2. Must not exceed numElemsPerThread
    if (forcedVec > numElemsPerThread) {
      llvm::report_fatal_error(
          llvm::Twine("candidate invalid: forced vector size ") + llvm::Twine(forcedVec) +
          " exceeds numElemsPerThread " + llvm::Twine(numElemsPerThread));
    }
    // 3. Contiguous dimension size must be divisible by forcedVec
    int64_t contiguousDim = shapePerCTA.back();
    if (contiguousDim % forcedVec != 0) {
      llvm::report_fatal_error(
          llvm::Twine("candidate invalid: contiguous dimension ") + llvm::Twine(contiguousDim) +
          " not divisible by forced vector size " + llvm::Twine(forcedVec));
    }
    // 4. Must be a power of 2
    if ((forcedVec & (forcedVec - 1)) != 0) {
      llvm::report_fatal_error(
          llvm::Twine("candidate invalid: forced vector size ") + llvm::Twine(forcedVec) +
          " is not a power of 2");
    }
    vectorSize = forcedVec;
  }
  SmallVector<unsigned> sizePerThread(type.getRank(), 1);
  sizePerThread.back() = vectorSize;

  SmallVector<unsigned> order =
      getMatrixOrder(type.getRank(), /*rowMajor*/ true);
  auto cgaLayout = triton::gpu::getCGALayout(type.getEncoding());

  Attribute layout = triton::gpu::BlockedEncodingAttr::get(
      type.getContext(), type.getShape(), sizePerThread, order, numWarps,
      threadsPerWarp, cgaLayout);
  return layout;
}

static void pickDescriptorLoadStoreLayout(
    ModuleOp moduleOp, llvm::MapVector<Operation *, Attribute> &layoutMap) {
  int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
  moduleOp.walk([&](Operation *op) {
    int numWarps = lookupNumWarps(op);
    if (auto load = dyn_cast<DescriptorOpInterface>(op)) {
      if (load->getNumResults() == 1)
        layoutMap[op] = pickDescriptorLoadStoreLayout(
            numWarps, threadsPerWarp,
            cast<RankedTensorType>(load->getResult(0).getType()));
    }
    if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
      layoutMap[op] = pickDescriptorLoadStoreLayout(numWarps, threadsPerWarp,
                                                    store.getSrc().getType());
    }
  });
}

struct CoalescePass : public impl::TritonGPUCoalesceBase<CoalescePass> {
  static Type getNewType(Type type, Attribute encoding) {
    RankedTensorType tensorType = cast<RankedTensorType>(type);
    return tensorType.cloneWithEncoding(encoding);
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    llvm::MapVector<Operation *, Attribute> layoutMap;
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(moduleOp);
    moduleOp.walk([&](Operation *curr) {
      Value ptr = getMemAccessPtr(curr);
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` load/store
      bool isPtrTensor = false;
      if (auto tensorType = dyn_cast<RankedTensorType>(ptr.getType()))
        isPtrTensor = isa<PointerType>(tensorType.getElementType());
      if (!isPtrTensor)
        return;
      int numWarps = lookupNumWarps(curr);

      auto tensorType = cast<RankedTensorType>(ptr.getType());
      CGAEncodingAttr cgaLayout = getCGALayout(tensorType.getEncoding());
      SmallVector<int64_t> shapePerCTA = getShapePerCTA(tensorType);
      auto layout =
          buildCoalescedEncoding(axisInfoAnalysis, curr, numWarps,
                                 threadsPerWarp, cgaLayout, shapePerCTA);
      layoutMap[curr] = layout;
    });

    // Also pick a layout for descriptor load/store ops.
    pickDescriptorLoadStoreLayout(moduleOp, layoutMap);

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    for (auto &kv : layoutMap) {
      convertDistributedOpEncoding(kv.second, kv.first);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
