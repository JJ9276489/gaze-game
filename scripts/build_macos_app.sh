#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

mkdir -p models
for model_name in face_landmarker.task vision_gaze_spatial_geom.pt; do
  if [[ ! -f "models/${model_name}" && -f "../eye-cursor/models/${model_name}" ]]; then
    cp "../eye-cursor/models/${model_name}" "models/${model_name}"
  fi
done

python -m PyInstaller --clean --noconfirm gaze_game.spec

APP_PATH="dist/Gaze Game.app"
ZIP_PATH="dist/Gaze-Game-alpha-macos-arm64.zip"

if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$APP_PATH"
fi

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - \
    --entitlements packaging/macos/entitlements.plist \
    "$APP_PATH"
fi

rm -f "$ZIP_PATH"
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "Built $APP_PATH"
echo "Packaged $ZIP_PATH"
