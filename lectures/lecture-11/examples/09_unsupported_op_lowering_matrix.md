# Unsupported Op Lowering Matrix

| Input op/pattern | Problem | Preferred lowering | Guard condition | Fallback |
|---|---|---|---|---|
| exact GELU / erf | no transcendental unit | LUT or polynomial GELU | calibrated input range, error <= budget | CPU runtime |
| LayerNorm rsqrt | exact rsqrt cost | LUT + Newton or vector rsqrt | epsilon policy fixed, dtype f16/i8 | CPU/DSP |
| dynamic gather | irregular memory | static index folding or coalesced gather | index tensor static/coalesced | CPU runtime |
| unsupported reduction axis | SRAM/reduction tree mismatch | transpose + reduce or split reduction | axis static, layout transform profitable | reject/fallback |
| non-multiple K matmul | MMA tile mismatch | edge tile path or pad K | padding cost acceptable | reject diagnostic |
