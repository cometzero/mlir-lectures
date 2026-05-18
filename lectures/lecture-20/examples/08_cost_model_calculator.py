#!/usr/bin/env python3
"""Tiny educational cost model for Lecture 20 Capstone."""
from dataclasses import dataclass

@dataclass
class Tile:
    m: int = 16
    n: int = 16
    k: int = 32
    bytes_a: int = 1
    bytes_b: int = 1
    bytes_acc: int = 4
    double_buffer: bool = True

    def sram_bytes(self) -> int:
        a = self.m * self.k * self.bytes_a
        b = self.k * self.n * self.bytes_b
        acc = self.m * self.n * self.bytes_acc
        db = 2 if self.double_buffer else 1
        return db * (a + b) + acc

    def macs(self) -> int:
        return self.m * self.n * self.k

    def arithmetic_intensity(self) -> float:
        return self.macs() / max(1, self.sram_bytes())

if __name__ == "__main__":
    t = Tile()
    print(f"Native tile: {t.m}x{t.n}x{t.k}")
    print(f"SRAM bytes per tile: {t.sram_bytes()}")
    print(f"MACs per tile: {t.macs()}")
    print(f"Arithmetic intensity: {t.arithmetic_intensity():.2f} MAC/byte")
