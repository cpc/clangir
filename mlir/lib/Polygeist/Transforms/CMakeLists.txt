

add_mlir_dialect_library(MLIRPolygeistTransforms
 Mem2Reg.cpp
 Inliner.cpp
 DistributeBarriers.cpp
 DetectReduction.cpp
 AffineCFG.cpp
 RaiseToAffine.cpp

 DEPENDS
 MLIRPolygeistTransformsPassIncGen

 LINK_LIBS PUBLIC
 MLIRAnalysis
 MLIRFunctionInterfaces
 MLIRLoopLikeInterface
 MLIRMemorySlotInterfaces
 MLIRPass
 MLIRRuntimeVerifiableOpInterface
 MLIRSideEffectInterfaces
 MLIRSupport
 MLIRTransformUtils
 MLIRUBDialect
 MLIRPolygeist
)
