//===- PolygeistOps.td------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_OPS
#define POLYGEIST_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def CacheLoad
    : Polygeist_Op<"cacheload"> {

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];
  let summary = "load from cross barrier cache";
  let description = [{}];
}
//===----------------------------------------------------------------------===//
// SubIndexOp
//===----------------------------------------------------------------------===//

def SubIndexOp : Polygeist_Op<"subindex", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source, Index : $index);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}
// HasParent<"ParallelOp">,
def BarrierOp
    : Polygeist_Op<"barrier",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins Variadic<Index>:$indices);
  let summary = "barrier for parallel loops";
  let description = [{}];
  let hasCanonicalizer = true;
}


#endif
