# Lecture 18 Quick Reference - Custom NPU Dialect

## ODS fields
- `let arguments`: operands and attributes.
- `let results`: SSA result values.
- `let assemblyFormat`: custom textual form.
- `let hasVerifier = 1`: C++ verifier hook.
- `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>`: read/write effects.

## Attribute vs operand
| Field | Prefer | Reason |
|---|---|---|
| native tile shape | Attribute | compile-time target fact |
| source/destination buffer | Operand | runtime SSA value |
| memory space | Type/Attribute | storage contract |
| DMA size if dynamic | Operand | runtime value |
| DMA size if static | Attribute | static command field |
| barrier scope | Attribute | compile-time ordering policy |

## Verifier checklist
- Tile shape supported by hardware.
- Layout matches engine dataflow.
- DType pair and accumulator type are supported.
- DMA direction maps legal memory spaces.
- Barrier operands are tokens and non-empty.
