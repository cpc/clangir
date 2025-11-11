//===- PolygeistOps.h ---*- C++ -*-===//
// File ported from Polygeist
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLYGEIST_POLYGEISTOPS_H
#define POLYGEIST_POLYGEISTOPS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Polygeist/Dialect/Ops.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::polygeist;

bool isValidIndex (mlir::Value val);

void fully2ComposeAffineMapAndOperands (
  mlir::PatternRewriter &rewriter,
  mlir::AffineMap *map,
  llvm::SmallVectorImpl<mlir::Value> *operands,
  mlir::DominanceInfo &DI);

bool collectEffects (
  mlir::Operation *op,
  llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
  bool ignoreBarriers);

bool getEffectsBefore (Operation *op,
                       SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                       bool stopAtBarrier);
bool getEffectsAfter (Operation *op,
                      SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                      bool stopAtBarrier);
bool isReadOnly (mlir::Operation *);
bool isReadNone (mlir::Operation *);

bool mayReadFrom (mlir::Operation *, mlir::Value);
bool mayWriteTo (mlir::Operation *, mlir::Value, bool ignoreBarrier = false);
bool mayAlias (mlir::MemoryEffects::EffectInstance a,
               mlir::MemoryEffects::EffectInstance b);

bool mayAlias (mlir::MemoryEffects::EffectInstance a, mlir::Value b);
Value getBase (Value v);
bool isStackAlloca (Value v);
bool isCaptured (Value v, Operation *potentialUser, bool *seenuse);

template <bool NotTopLevel = false>
class BarrierElim final
  : public mlir::OpRewritePattern<mlir::polygeist::BarrierOp>
{
public:
  using mlir::OpRewritePattern<mlir::polygeist::BarrierOp>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite (mlir::polygeist::BarrierOp barrier,
                   mlir::PatternRewriter &rewriter) const override
  {
    using namespace mlir;
    using namespace polygeist;
    // Remove if it only sync's constant indices.
    if (llvm::all_of (barrier.getOperands (), [] (mlir::Value v) {
          IntegerAttr constValue;
          return matchPattern (v, m_Constant (&constValue));
        }))
      {
        rewriter.eraseOp (barrier);
        return success ();
      }

    Operation *op = barrier;
    if (NotTopLevel
        && isa<mlir::scf::ParallelOp, mlir::affine::AffineParallelOp> (
          barrier->getParentOp ()))
      return failure ();

    {
      SmallVector<mlir::MemoryEffects::EffectInstance> beforeEffects;
      getEffectsBefore (op, beforeEffects, /*stopAtBarrier*/ true);

      SmallVector<mlir::MemoryEffects::EffectInstance> afterEffects;
      getEffectsAfter (op, afterEffects, /*stopAtBarrier*/ false);

      bool conflict = false;
      for (auto before : beforeEffects)
        for (auto after : afterEffects)
          {
            if (mayAlias (before, after))
              {
                // Read, read is okay
                if (isa<mlir::MemoryEffects::Read> (before.getEffect ())
                    && isa<mlir::MemoryEffects::Read> (after.getEffect ()))
                  {
                    continue;
                  }

                // Write, write is not okay because may be different offsets
                // and the later must subsume other conflicts are invalid.
                conflict = true;
                break;
              }
          }

      if (!conflict)
        {
          rewriter.eraseOp (barrier);
          return success ();
        }
    }

    {
      SmallVector<mlir::MemoryEffects::EffectInstance> beforeEffects;
      getEffectsBefore (op, beforeEffects, /*stopAtBarrier*/ false);

      SmallVector<mlir::MemoryEffects::EffectInstance> afterEffects;
      getEffectsAfter (op, afterEffects, /*stopAtBarrier*/ true);

      bool conflict = false;
      for (auto before : beforeEffects)
        for (auto after : afterEffects)
          {
            if (mayAlias (before, after))
              {
                // Read, read is okay
                if (isa<mlir::MemoryEffects::Read> (before.getEffect ())
                    && isa<mlir::MemoryEffects::Read> (after.getEffect ()))
                  {
                    continue;
                  }
                // Write, write is not okay because may be different offsets
                // and the later must subsume other conflicts are invalid.
                conflict = true;
                break;
              }
          }

      if (!conflict)
        {
          rewriter.eraseOp (barrier);
          return success ();
        }
    }

    return failure ();
  }
};

struct ValueOrInt
{
  bool isValue;
  mlir::Value v_val;
  int64_t i_val;
  ValueOrInt (mlir::Value v) { initValue (v); }
  void
  initValue (mlir::Value v)
  {
    using namespace mlir;
    if (v)
      {
        IntegerAttr iattr;
        if (matchPattern (v, m_Constant (&iattr)))
          {
            i_val = iattr.getValue ().getSExtValue ();
            v_val = nullptr;
            isValue = false;
            return;
          }
      }
    isValue = true;
    v_val = v;
  }

  ValueOrInt (size_t i) : isValue (false), v_val (), i_val (i) {}

  bool
  operator>= (int64_t v)
  {
    if (isValue)
      return false;
    return i_val >= v;
  }
  bool
  operator> (int64_t v)
  {
    if (isValue)
      return false;
    return i_val > v;
  }
  bool
  operator== (int64_t v)
  {
    if (isValue)
      return false;
    return i_val == v;
  }
  bool
  operator< (int64_t v)
  {
    if (isValue)
      return false;
    return i_val < v;
  }
  bool
  operator<= (int64_t v)
  {
    if (isValue)
      return false;
    return i_val <= v;
  }
  bool
  operator>= (llvm::APInt v)
  {
    if (isValue)
      return false;
    return i_val >= v.getSExtValue ();
  }
  bool
  operator> (llvm::APInt v)
  {
    if (isValue)
      return false;
    return i_val > v.getSExtValue ();
  }
  bool
  operator== (llvm::APInt v)
  {
    if (isValue)
      return false;
    return i_val == v.getSExtValue ();
  }
  bool
  operator< (llvm::APInt v)
  {
    if (isValue)
      return false;
    return i_val < v.getSExtValue ();
  }
  bool
  operator<= (llvm::APInt v)
  {
    if (isValue)
      return false;
    return i_val <= v.getSExtValue ();
  }
};

enum class Cmp
{
  EQ,
  LT,
  LE,
  GT,
  GE
};

bool valueCmp (Cmp cmp,
               mlir::AffineExpr expr,
               size_t numDim,
               mlir::ValueRange operands,
               ValueOrInt val);

bool valueCmp (Cmp cmp, mlir::Value bval, ValueOrInt val);

#endif
