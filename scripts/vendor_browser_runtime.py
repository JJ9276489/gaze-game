#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASKS_VERSION = "0.10.34"
ORT_VERSION = "1.24.3"

ASSETS = [
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/vision_bundle.mjs",
        "web/vendor/mediapipe/tasks-vision/vision_bundle.mjs",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_internal.js",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_internal.wasm",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_module_internal.js",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_module_internal.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_module_internal.wasm",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_module_internal.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_nosimd_internal.js",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{TASKS_VERSION}/wasm/vision_wasm_nosimd_internal.wasm",
        "web/vendor/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm",
    ),
    (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "web/vendor/mediapipe/models/face_landmarker.task",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort.wasm.min.mjs",
        "web/vendor/onnxruntime/ort.wasm.min.mjs",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.asyncify.wasm",
        "web/vendor/onnxruntime/ort-wasm-simd-threaded.asyncify.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.jsep.wasm",
        "web/vendor/onnxruntime/ort-wasm-simd-threaded.jsep.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.jspi.wasm",
        "web/vendor/onnxruntime/ort-wasm-simd-threaded.jspi.wasm",
    ),
    (
        f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist/ort-wasm-simd-threaded.wasm",
        "web/vendor/onnxruntime/ort-wasm-simd-threaded.wasm",
    ),
]


def remote_content_length(url: str) -> int | None:
    result = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "--fail", "--head", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in reversed(result.stdout.splitlines()):
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def download_asset(url: str, relative_path: str, force: bool = False) -> None:
    output_path = PROJECT_ROOT / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not shutil.which("curl"):
        raise SystemExit("curl is required to vendor browser runtime assets")

    temp_path = output_path.with_name(f"{output_path.name}.part")
    expected_size = remote_content_length(url)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        existing_size = output_path.stat().st_size
        if expected_size is None or existing_size == expected_size:
            print(f"Already vendored {relative_path}")
            return
        if existing_size < expected_size and not temp_path.exists():
            output_path.replace(temp_path)
        else:
            output_path.unlink()

    print(f"Vendoring {relative_path}")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "600",
            "-C",
            "-",
            "-o",
            str(temp_path),
            url,
        ],
        check=True,
    )
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vendor browser runtime assets under web/vendor/.")
    parser.add_argument("--force", action="store_true", help="Download assets even if they already exist.")
    args = parser.parse_args()

    for url, relative_path in ASSETS:
        download_asset(url, relative_path, force=args.force)


if __name__ == "__main__":
    main()
