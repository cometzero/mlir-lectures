// Educational EmitC-style pseudo lowering.
module {
  func.func @run_model() {
    %ctx = emitc.call_opaque "npuGetDefaultContext"() : () -> !emitc.ptr<!emitc.opaque<"NpuContext">>
    %module = emitc.call_opaque "npuLoadModule"(%ctx) : (!emitc.ptr<!emitc.opaque<"NpuContext">>) -> !emitc.ptr<!emitc.opaque<"NpuModule">>
    %fence = emitc.call_opaque "npuSubmit"(%ctx, %module) : (!emitc.ptr<!emitc.opaque<"NpuContext">>, !emitc.ptr<!emitc.opaque<"NpuModule">>) -> !emitc.ptr<!emitc.opaque<"NpuFence">>
    emitc.call_opaque "npuWait"(%fence) : (!emitc.ptr<!emitc.opaque<"NpuFence">>) -> ()
    return
  }
}
