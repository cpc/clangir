//===- Passes.h ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_MLIR_PASSES_H
#define POLYGEIST_MLIR_PASSES_H

#include "mlir/Polygeist/Dialect/Dialect.h"

#include <memory>

namespace mlir
{
class Pass;
}

namespace mlir
{
namespace polygeist
{

#define GEN_PASS_DECL_DISTRIBUTEBARRIERS
#include "mlir/Polygeist/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createInlinerPass ();
std::unique_ptr<mlir::Pass> createMem2RegPass ();
std::unique_ptr<mlir::Pass> createDistributeBarriersPass (
  mlir::polygeist::DistributeBarriersOptions method);
std::unique_ptr<mlir::Pass> createDistributeBarriersPass ();
std::unique_ptr<mlir::Pass> replaceAffineCFGPass ();
std::unique_ptr<mlir::Pass> createRaiseToAffinePass ();
std::unique_ptr<mlir::Pass> createDetectReductionPass ();

#define GEN_PASS_REGISTRATION
#include "mlir/Polygeist/Transforms/Passes.h.inc"

}
}

#endif
