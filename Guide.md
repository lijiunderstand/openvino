# Guide: OpenVINO GPU Plugin Custom Op Development (BevPoolV2 Example)

This guide summarizes how to develop a custom operator in the OpenVINO Intel GPU plugin, using BevPoolV2 as a concrete end-to-end example.

## 1. Scope

Goal:
- Explain the development path from Core op to GPU execution.
- Explain how to validate correctness and performance.
- Capture the current repository status and practical lessons.

Non-goals:
- This is not a generic ONNX tutorial.
- This does not cover every historical BevPoolV2 path; it focuses on the current architecture.

## 2. Current Repository Change Summary (as of 2026-03-21)

### 2.1 Landed major changes

1. BevPoolV2 Core op and GPU integration are available.
2. GPU path for BevPoolV2 is switched to ocl_v2 as the main path.
3. Legacy BevPoolV2 OCL/kernel_selector path has been removed.
4. ONNX custom domain is unified to org.openvinotoolkit.
5. Guide/GSG and helper scripts for export/compare/benchmark workflow were added.

Representative recent commits:
- 1f1744b2f1 [core] Add BevPoolV2 op with attributes and shape/eval support
- ca638ec58d [gpu] Add BevPoolV2 primitive, translator, and graph OCL impl
- 0d76a8843c [gpu][kernel] Add BevPoolV2 kernel selector, OpenCL kernel, and JIT constants
- b4051ca78a intel_gpu: switch bevpool_v2 registry to ocl_v2 static-only
- b888ca7d7b remove bevpool_v2 legacy ocl impl and register hooks
- 8f644fdac1 drop bevpool_v2 kernel_selector path and update guide
- 96fa77e02b onnx: use org.openvinotoolkit domain for bevpool_v2 only

### 2.2 Current working-tree updates

Current local updates include:
- ocl_v2 host-side stage selection improvements (ref/opt4/opt8), capability guard, env-force switches.
- test enhancements for thresholds and larger-shape coverage.
- Performance/parity report script and reference-bin directory integration.

Files currently relevant to these local updates:
- src/plugins/intel_gpu/src/graph/impls/ocl_v2/bevpool_v2.cpp
- src/plugins/intel_gpu/src/graph/impls/ocl_v2/bevpool_v2_opt.cl
- src/plugins/intel_gpu/tests/unit/test_cases/bevpool_v2_gpu_test.cpp
- src/tests/functional/plugin/shared/src/single_op/bevpool_v2.cpp
- scripts/bevpool_performance_and_parity_report.py

## 3. Recommended Architecture for a New GPU Custom Op

For a new operator in OpenVINO GPU plugin, follow this layering:

1. Core op layer
- src/core/include/openvino/op/<op>.hpp
- src/core/src/op/<op>.cpp

2. Frontend translation layer (if importing from ONNX/TF/etc.)
- src/frontends/onnx/frontend/src/op/<domain>/<op>.cpp

3. GPU primitive + plugin translator layer
- src/plugins/intel_gpu/include/intel_gpu/primitives/<op>.hpp
- src/plugins/intel_gpu/src/plugin/ops/<op>.cpp

4. GPU implementation layer (preferred: ocl_v2)
- src/plugins/intel_gpu/src/graph/impls/ocl_v2/<op>.hpp
- src/plugins/intel_gpu/src/graph/impls/ocl_v2/<op>.cpp
- src/plugins/intel_gpu/src/graph/impls/ocl_v2/<op>.cl

5. Registry layer
- src/plugins/intel_gpu/src/graph/registry/<op>_impls.cpp
- src/plugins/intel_gpu/src/graph/registry/registry.hpp

6. Tests and validation
- GPU unit tests
- plugin shared single-op tests
- plugin functional tests
- benchmark/parity scripts

## 4. BevPoolV2 End-to-End Dataflow (Current)

1. ONNX model uses org.openvinotoolkit::BevPoolV2.
2. Frontend translator maps ONNX op to ov::op::v15::BevPoolV2.
3. GPU plugin translator maps Core op to cldnn::bevpool_v2 primitive.
4. Registry picks ocl_v2 implementation manager.
5. ocl_v2 host code builds stage/JIT/dispatch and runs OpenCL kernel.
6. Tests and benchmark compare CPU/GPU/ref-opt behavior.

## 5. Step-by-Step Implementation Checklist (Using BevPoolV2 Pattern)

### Step 1: Define Core op

Required:
- attributes
- validate_and_infer_types()
- clone_with_new_inputs()
- evaluate() when feasible (helps debug/reference)

Guideline:
- Keep shape/type validation strict and fail fast.

### Step 2: Add frontend translation contract

For ONNX custom ops:
- Keep exporter and frontend domain/op name aligned.
- Current BevPoolV2 contract: domain org.openvinotoolkit.

### Step 3: Add GPU primitive and plugin translator

Required:
- Primitive should carry all runtime attributes needed by GPU impl.
- Plugin op translator should map Core op attributes exactly.

### Step 4: Implement ocl_v2 host and kernel

Host side responsibilities:
- JIT constants for all dimensions and bounds.
- argument descriptors for all inputs/outputs.
- dispatch function (GWS/LWS).
- stage selection strategy (ref/opt) with safe fallback.

Kernel side responsibilities:
- deterministic indexing and bounds checks.
- numeric stability and explicit accumulation policy.
- avoid hidden assumptions about input layout/rank.

### Step 5: Register implementation correctly

Required:
- Add a dedicated registry file for the op.
- Ensure REGISTER_IMPLS(op_name) wiring is complete.

Important:
- Keep one authoritative path during migration.
- For BevPoolV2, do not reintroduce legacy OCL/kernel_selector path.

### Step 6: Add test coverage before optimization

Minimum test set:
- Unit tests for deterministic tiny cases.
- Functional single-op tests for realistic shapes and dtypes.
- Index type coverage (i32/i64 and u32 where needed).
- Error thresholds by dtype and stress shape.

### Step 7: Add benchmark and parity workflow

Required outputs:
- latency / throughput for CPU and GPU variants
- max_abs / mean_abs / max_rel / mean_rel against reference bins

Use script-based report generation to keep results reproducible.

## 6. Build and Validation Commands

Run from repository root:

```bash
cd /home/lijie/intel/intel_gpu/openvino

cmake -S . -B build_bevpool_tests -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
cmake --build build_bevpool_tests --target ov_gpu_unit_tests ov_gpu_func_tests -j"$(nproc)"

./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter='*bevpool_v2*'
./bin/intel64/Release/ov_gpu_func_tests --gtest_filter='*BevPoolV2*'
```

Benchmark smoke (named shape format):

```bash
./bin/intel64/Release/benchmark_app \
  -m ./bevpool_v2_custom.onnx \
  -d GPU \
  -shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7314,3]" \
  -niter 100 --nireq 1
```

Performance/parity report generation:

```bash
python scripts/bevpool_performance_and_parity_report.py \
  --repo-root . \
  --model ./bevpool_v2_custom.onnx \
  --ref-dir /home/lijie/intel/intel_gpu/openvino/scripts/bevpool_compare_inputs \
  --shape "feat[1,54,96,80],depth[1,90,54,96],indices[466560],intervals[7314,3]" \
  --benchmark-bin ./bin/intel64/Release/benchmark_app \
  --compare-script ./compare_bevpool_ref.py \
  --niter 100 \
  --nireq 1 \
  --topk 10 \
  --output ./bevpool_performance_parity_report.md
```

By default, the script prepends local build paths for parity subprocesses:

- `PYTHONPATH += ./bin/intel64/Release/python`
- `LD_LIBRARY_PATH += ./bin/intel64/Release`

You can override these defaults when needed:

- `--ov-python-dir <path_to_local_python_pkg>`
- `--ov-lib-dir <path_to_local_runtime_libs>`

## 7. Common Failure Modes and Fixes

1. Kernel cache miss / stage not found
- Check KernelGenerator template naming and CL template mapping.
- Ensure stage is actually added in constructor and selected in execution order.

2. Runtime build issues after deleting old paths
- Re-run CMake configure to refresh stale source lists:
  - cmake -S . -B build_bevpool_tests

3. Wrong benchmark shape format
- Use named input format expected by benchmark_app:
  - feat[...],depth[...],indices[...],intervals[...]

4. Parity step fails
- Verify reference directory contains all required files:
  - meta.txt
  - camera_features.bin
  - camera_depth_weights.bin
  - indices.bin
  - intervals.bin
  - bev_ref_output.bin
- For `scripts/bevpool_performance_and_parity_report.py`, local Python/runtime paths are injected automatically for parity; if your build layout differs, pass `--ov-python-dir` and `--ov-lib-dir` explicitly.
- For standalone `compare_bevpool_ref.py` runs, ensure Python OpenVINO package and runtime libraries are from the same local build:
  - export PYTHONPATH=./bin/intel64/Release/python:$PYTHONPATH
  - export LD_LIBRARY_PATH=./bin/intel64/Release:$LD_LIBRARY_PATH
- `compare_bevpool_ref.py` uses input-name mapping for feat/depth/indices/intervals; if names are customized, keep aliases aligned in the comparator.

## 8. Recommended Acceptance Criteria

A BevPoolV2-like custom op is considered ready when:

1. Build passes in Release with tests enabled.
2. Unit tests and functional tests pass for the op.
3. benchmark_app runs on CPU and GPU with the same model+shape.
4. Parity metrics are recorded and within agreed thresholds.
5. Documentation is reproducible (commands/paths/env pinned).

## 9. Practical Engineering Rules from This Case

1. Keep one active implementation path during migration.
2. Prioritize correctness + deterministic tests before optimization.
3. Add optimization branches with explicit guard and fallback.
4. Keep frontend domain contract stable and single-sourced.
5. Update docs and scripts in the same iteration as code changes.
