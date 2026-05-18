// Educational EmitC-style pseudo lowering.
module {
  func.func @run_model() {
    %ctx = emitc.call_opaque "npuGetDefaultContext"() : () -> !emitc.opaque<"NpuContext*">
    %module = emitc.call_opaque "npuLoadModule"(%ctx) : (!emitc.opaque<"NpuContext*">) -> !emitc.opaque<"NpuModule*">
    %fence = emitc.call_opaque "npuSubmit"(%ctx, %module) : (!emitc.opaque<"NpuContext*">, !emitc.opaque<"NpuModule*">) -> !emitc.opaque<"NpuFence*">
    emitc.call_opaque "npuWait"(%fence) : (!emitc.opaque<"NpuFence*">) -> ()
    return
  }
}
