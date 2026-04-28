from dataclasses import dataclass
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "config" / "browser_models.json"


@dataclass(frozen=True)
class BrowserModelSpec:
    key: str
    label: str
    checkpoint: Path
    fallback_checkpoint: Path | None
    output: Path

    def resolve_checkpoint(self) -> Path:
        if self.checkpoint.exists():
            return self.checkpoint
        if self.fallback_checkpoint is not None and self.fallback_checkpoint.exists():
            return self.fallback_checkpoint
        candidates = [self.checkpoint]
        if self.fallback_checkpoint is not None:
            candidates.append(self.fallback_checkpoint)
        joined = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No checkpoint found for {self.key}: {joined}")


def _project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_browser_model_specs() -> list[BrowserModelSpec]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    specs = []
    for item in manifest.get("models", []):
        specs.append(
            BrowserModelSpec(
                key=str(item["key"]),
                label=str(item["label"]),
                checkpoint=_project_path(item["checkpoint"]) or PROJECT_ROOT,
                fallback_checkpoint=_project_path(item.get("fallback_checkpoint")),
                output=_project_path(item["output"]) or PROJECT_ROOT,
            )
        )
    if not specs:
        raise ValueError(f"No browser models configured in {MANIFEST_PATH}")
    return specs


def browser_model_spec(key: str) -> BrowserModelSpec:
    for spec in load_browser_model_specs():
        if spec.key == key:
            return spec
    available = ", ".join(spec.key for spec in load_browser_model_specs())
    raise KeyError(f"Unknown browser model {key!r}. Available: {available}")


def selected_browser_model_specs(model: str | None, all_models: bool) -> list[BrowserModelSpec]:
    specs = load_browser_model_specs()
    if all_models or model is None:
        return specs
    return [browser_model_spec(model)]
