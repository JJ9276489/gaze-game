# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).resolve()
model_dir = project_root / "models"


def include_mediapipe_task_module(name):
    return not (
        ".benchmark" in name
        or ".test" in name
        or name.startswith("mediapipe.tasks.python.genai.")
        or name.startswith("mediapipe.tasks.python.metadata")
    )


datas = []
for model_name in ("face_landmarker.task", "vision_gaze_spatial_geom.pt"):
    model_path = model_dir / model_name
    if model_path.exists():
        datas.append((str(model_path), "models"))

datas += collect_data_files(
    "mediapipe.tasks",
    includes=["c/libmediapipe.dylib", "metadata/*.fbs"],
)
mediapipe_task_hiddenimports = collect_submodules(
    "mediapipe.tasks",
    filter=include_mediapipe_task_module,
)

a = Analysis(
    ["gaze_game.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "mediapipe",
        "mediapipe.tasks",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "websockets.asyncio.client",
        *mediapipe_task_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Gaze Game",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=str(project_root / "packaging" / "macos" / "entitlements.plist"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Gaze Game",
)

app = BUNDLE(
    coll,
    name="Gaze Game.app",
    icon=None,
    bundle_identifier="com.jeraldyuan.gazegame",
    info_plist={
        "CFBundleDisplayName": "Gaze Game",
        "CFBundleName": "Gaze Game",
        "LSMinimumSystemVersion": "13.0",
        "NSCameraUsageDescription": (
            "Gaze Game uses your camera locally to estimate gaze. "
            "Video, eye crops, and face landmarks never leave your Mac."
        ),
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "Copyright © 2026 Jerald Yuan",
    },
)
