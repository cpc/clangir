
//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Polygeist/Dialect/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Polygeist/Dialect/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Polygeist/Dialect/PolygeistOps.cpp.inc"
      >();
}

#include "mlir/Polygeist/Dialect/PolygeistOpsDialect.cpp.inc"
