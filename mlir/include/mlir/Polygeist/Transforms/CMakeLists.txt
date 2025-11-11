
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PolygeistTransforms)
add_mlir_dialect_tablegen_target(MLIRPolygeistTransformsPassIncGen)
