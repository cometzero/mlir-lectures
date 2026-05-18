# Lab Environment

## 최소 요구사항

- Python 3.10+
- Git
- 선택: LLVM/MLIR tools (`mlir-opt`, `mlir-translate`, `FileCheck`)

자료 검증만 하려면 Python만 있으면 됩니다.

```bash
python3 scripts/verify_materials.py
```

## MLIR 도구 확인

```bash
mlir-opt --version
mlir-translate --version
FileCheck --version
```

## LLVM/MLIR를 source build 하는 예시

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja   -DLLVM_ENABLE_PROJECTS=mlir   -DLLVM_BUILD_EXAMPLES=ON   -DLLVM_TARGETS_TO_BUILD=host   -DCMAKE_BUILD_TYPE=Release   -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build build --target mlir-opt mlir-translate FileCheck -j$(nproc)
export PATH="$PWD/build/bin:$PATH"
```

## Lecture 실행 helper

예제 목록만 확인:

```bash
scripts/run_lesson.sh 01
```

MLIR parse/verify를 가능한 범위에서 시도:

```bash
scripts/run_lesson.sh 04 --try-mlir
```

엄격 모드(실패 시 non-zero exit):

```bash
scripts/run_lesson.sh 04 --try-mlir --strict
```

## 중요한 주의사항

이 코스에는 교육용 pseudo NPU dialect, skeleton C++/ODS, target-specific attribute가 많이 포함됩니다. 따라서 vanilla MLIR toolchain만으로 모든 `.mlir`이 성공해야 하는 것은 아닙니다.

실패를 보면 다음 질문으로 이어가세요.

- 어떤 dialect/type/op가 등록되어야 하는가?
- verifier가 어느 조건을 reject해야 하는가?
- 이 IR은 high-level graph, kernel schedule, runtime ABI 중 어느 abstraction인가?
- FileCheck는 presence, absence, order, diagnostic 중 무엇을 검증해야 하는가?
