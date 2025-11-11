//===- Dialect.td ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_DIALECT
#define POLYGEIST_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BFV dialect definition.
//===----------------------------------------------------------------------===//

def PolygeistDialect : Dialect {
  let name = "polygeist";
  let description = [{}];
  let cppNamespace = "::mlir::polygeist";
}

//===----------------------------------------------------------------------===//
// Base BFV operation definition.
//===----------------------------------------------------------------------===//

class Polygeist_Op<string mnemonic, list<Trait> traits = []>
    : Op<PolygeistDialect, mnemonic, traits>;

#endif // POLYGEIST_DIALECT
