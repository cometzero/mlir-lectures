# Lab Environment

## 최소 요구사항

- Python 3.10+
- Git
- 선택: LLVM/MLIR tools (`mlir-opt`, `mlir-translate`, `FileCheck`)

자료 구조만 검증하려면 Python만 있으면 됩니다.

```bash
python3 scripts/verify_materials.py
```

## MLIR 도구 확인

```bash
mlir-opt --version
mlir-translate --version
FileCheck --version
```

Ubuntu 패키지를 사용하면 실행 파일 이름이 `mlir-opt-18`처럼 version suffix를 가질 수 있습니다. 그 경우는 LLVM bin directory를 `PATH`에 넣어 root helper가 unversioned tool name을 찾게 합니다.

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build clang lld llvm-18-tools mlir-18-tools
export PATH="/usr/lib/llvm-18/bin:$PATH"
mlir-opt --version
FileCheck --version
```

## LLVM/MLIR source build 예시

강의용으로는 MLIR tools만 있으면 되므로 shallow clone과 host target build를 권장합니다. 전체 LLVM tree와 build artifact는 수 GB 이상이 필요합니다.

```bash
git clone --depth 1 --filter=blob:none https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_BUILD_DOCS=OFF
cmake --build build --target mlir-opt mlir-translate FileCheck -j2
export PATH="$PWD/build/bin:$PATH"
```

메모리와 디스크가 충분한 개발 장비에서는 `-j$(nproc)`로 병렬도를 높일 수 있습니다. 작은 VM에서는 `-j2` 정도가 안전합니다.

## Lecture 실행 helper

예제 목록만 확인:

```bash
scripts/run_lesson.sh 01
```

stock MLIR toolchain으로 실행 가능한 parse/verify와 Python helper를 실행:

```bash
scripts/run_lesson.sh 04 --try-mlir
```

엄격 모드(실패 시 non-zero exit):

```bash
scripts/run_lesson.sh 04 --try-mlir --strict
```

전체 lecture smoke test:

```bash
scripts/run_all_lessons.sh
```

`MLIR_OPT`와 `PYTHON`을 명시해서 다른 toolchain을 사용할 수도 있습니다.

```bash
MLIR_OPT=/path/to/llvm-project/build/bin/mlir-opt \
PYTHON=python3 \
scripts/run_all_lessons.sh
```

## Stock-safe 실행 정책

이 코스에는 다음과 같은 교육용 artifact가 섞여 있습니다.

- pseudo NPU dialect/op/type/attribute (`npu`, `npu_graph`, `npu_kernel`, `npu_rt` 등)
- 아직 구현하지 않은 custom NPU pass (`--npu-*`, `npu-opt`)
- version-sensitive Transform/PDL/Quantization syntax sketch
- StableHLO처럼 upstream LLVM/MLIR 기본 빌드에 포함되지 않는 외부 dialect 예제

따라서 vanilla `mlir-opt`로 모든 `.mlir` 파일이 통과해야 하는 것은 아닙니다. `scripts/run_lesson.sh --try-mlir`는 stock toolchain에서 실행 가능한 파일만 `CHECK`하고, custom backend가 필요한 파일은 `SKIP`으로 표시합니다. 이 SKIP은 오류가 아니라 강의에서 “어떤 dialect/pass/verifier를 직접 구현해야 하는가?”를 토론하기 위한 경계 표시입니다.

실패를 보면 다음 질문으로 이어가세요.

- 어떤 dialect/type/op가 등록되어야 하는가?
- verifier가 어느 조건을 reject해야 하는가?
- 이 IR은 high-level graph, kernel schedule, runtime ABI 중 어느 abstraction인가?
- FileCheck는 presence, absence, order, diagnostic 중 무엇을 검증해야 하는가?
