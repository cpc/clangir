//===- Passes.td ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_MLIR_PASSES_TD
#define POLYGEIST_MLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def Inliner : Pass<"pocl-inliner", "mlir::ModuleOp"> {
	let summary = "TODO";
	let constructor = "mlir::polygeist::createInlinerPass()";
	let dependentDialects = [
	"mlir::scf::SCFDialect",
	"mlir::memref::MemRefDialect",
	"mlir::func::FuncDialect",
	"mlir::arith::ArithDialect",
	"mlir::gpu::GPUDialect",
	];
}

def Mem2Reg : Pass<"pocl-mem2reg", "mlir::ModuleOp"> {
	let summary = "TODO";
	let constructor = "mlir::polygeist::createMem2RegPass()";
	let dependentDialects = [
	"mlir::memref::MemRefDialect",
	"mlir::func::FuncDialect",
	];
}

def DistributeBarriers : Pass<"pocl-distribute-barriers", "mlir::ModuleOp"> {
  let summary = "remove scf.barrier";
  let constructor = "mlir::polygeist::createDistributeBarriersPass()";
  let dependentDialects = [
	"mlir::memref::MemRefDialect",
	"mlir::func::FuncDialect",
	"mlir::polygeist::PolygeistDialect",
	];
  let options = [
  Option<"method", "method", "std::string", /*default=*/"\"distribute\"", "Method of doing distribution">
  ];
}

def AffineCFG : Pass<"pocl-affine-cfg", "mlir::ModuleOp"> {
	let summary = "Replace scf.if and similar with affine.if. Replace memref load/store with affine equivalent.";
	let constructor = "mlir::polygeist::replaceAffineCFGPass()";
	let dependentDialects = [
	"mlir::affine::AffineDialect",
	"mlir::func::FuncDialect",
	"mlir::scf::SCFDialect",
	"mlir::memref::MemRefDialect"
	];
}

def RaiseToAffine : Pass<"pocl-raise-to-affine", "mlir::ModuleOp"> {
	let summary = "Replace scf.parallel and scf.for with affine equivalents.";
	let constructor = "mlir::polygeist::createRaiseToAffinePass()";
	let dependentDialects = [
	"mlir::affine::AffineDialect",
	"mlir::func::FuncDialect",
	"mlir::scf::SCFDialect"
	];
}

def DetectReduction : Pass<"pocl-detect-reduction", "mlir::ModuleOp"> {
	let summary = "TODO";
	let constructor = "mlir::polygeist::createDetectReductionPass()";
	let dependentDialects = [
	"mlir::affine::AffineDialect",
	"mlir::func::FuncDialect",
	"mlir::scf::SCFDialect"
	];
  let statistics = [
    Statistic<"numReductions", "num-reductions", "Number of reductions detected">
  ];
}

#endif
