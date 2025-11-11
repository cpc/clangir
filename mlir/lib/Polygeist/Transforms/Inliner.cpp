//===- Inliner.cpp ---*- C++ -*-===//
// File ported from Polygeist
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Inliner.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"

#include "mlir/Polygeist/Transforms/Passes.h"

namespace {
#define GEN_PASS_DEF_INLINER
#include "mlir/Polygeist/Transforms/Passes.h.inc"
} // namespace

struct AlwaysInlinerInterface : public mlir::InlinerInterface {
  using InlinerInterface::InlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All operations within standard ops can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ArrayRef<mlir::Value> valuesToRepl) const {
    // Only "std.return" needs to be handled here.
    auto returnOp = mlir::cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      auto valueToRepl = valuesToRepl[it.index()];
      valueToRepl.replaceAllUsesWith(it.value());
    }
  }
};

namespace {
struct InlinerPass : public impl::InlinerBase<InlinerPass> {

  void runOnOperation() override {

    auto mod = getOperation();
    auto context = mod.getContext();
    auto opbuilder = mlir::OpBuilder(mod);
    mlir::SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(mod);

    std::function<void(mlir::func::CallOp)> callInliner =
        [&](mlir::func::CallOp caller) {
          // Build the inliner interface.
          AlwaysInlinerInterface interface(context);

          auto callable = caller.getCallableForCallee();
          mlir::CallableOpInterface callableOp;
          if (mlir::SymbolRefAttr symRef =
                  mlir::dyn_cast<mlir::SymbolRefAttr>(callable)) {
            if (!mlir::isa<mlir::FlatSymbolRefAttr>(symRef)) {
              return;
            }
            auto *symbolOp =
                symbolTable.lookupNearestSymbolFrom(getOperation(), symRef);
            callableOp =
                mlir::dyn_cast_or_null<mlir::CallableOpInterface>(symbolOp);
          } else {
            return;
          }
          mlir::Region *targetRegion = callableOp.getCallableRegion();
          if (!targetRegion) {
            return;
          }
          if (targetRegion->empty()) {
            return;
          }
          llvm::SmallVector<mlir::func::CallOp> ops;
          callableOp.walk(
              [&](mlir::func::CallOp caller) { ops.push_back(caller); });
          for (auto op : ops) {
            callInliner(op);
          }
          mlir::OpBuilder b(caller);
          auto allocScope = b.create<mlir::memref::AllocaScopeOp>(
              caller.getLoc(), caller.getResultTypes());
          allocScope.getRegion().push_back(new mlir::Block());
          b.setInsertionPointToStart(&allocScope.getRegion().front());
          auto exOp = b.create<mlir::scf::ExecuteRegionOp>(
              caller.getLoc(), caller.getResultTypes());
          mlir::Block *blk = new mlir::Block();
          exOp.getRegion().push_back(blk);
          caller->moveBefore(blk, blk->begin());
          caller.replaceAllUsesWith(allocScope.getResults());
          b.setInsertionPointToEnd(blk);
          b.create<mlir::scf::YieldOp>(caller.getLoc(), caller.getResults());
          auto inlinerConfig = mlir::InlinerConfig();
          if (inlineCall(interface, inlinerConfig.getCloneCallback(), caller,
                         callableOp, targetRegion,
                         /*shouldCloneInlinedRegion=*/true)
                  .succeeded()) {
            caller.erase();
          }
          b.setInsertionPointToEnd(&allocScope.getRegion().front());
          b.create<mlir::memref::AllocaScopeReturnOp>(allocScope.getLoc(),
                                                      exOp.getResults());
        };
    llvm::SmallVector<mlir::func::CallOp> ops;
    auto kernelFuncs = mod.getRegion().front().getOps<mlir::func::FuncOp>();
    for (auto kernelFunc : kernelFuncs) {
      bool isKernel =
          kernelFunc->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());
      if (isKernel) {
        kernelFunc.walk(
            [&](mlir::func::CallOp caller) { ops.push_back(caller); });
        for (auto op : ops) {
          callInliner(op);
        }
      }
    }
    llvm::SmallVector<mlir::func::FuncOp> opsToErase;
    auto kernelFuncsAfterInlining =
        mod.getRegion().front().getOps<mlir::func::FuncOp>();
    for (auto kernelFunc : kernelFuncsAfterInlining) {

      bool isKernel =
          kernelFunc->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());
      if (!isKernel) {
        opsToErase.push_back(kernelFunc);
      }
    }
    for (auto op : opsToErase) {
      op.erase();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::polygeist::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
