from pathlib import Path, PureWindowsPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value):
    """Resolve bundled paths while preserving explicitly configured external paths."""
    if not value:
        return value
    normalized = str(value).replace("\\", "/")
    legacy = "C:/OKO_PORUCHNI/"
    if normalized.lower().startswith(legacy.lower()):
        return _bundle_path(normalized[len(legacy):])
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return str(value)
    return _bundle_path(normalized)


def _bundle_path(relative):
    path = PROJECT_ROOT / relative
    if not path.exists() and path.parent == PROJECT_ROOT / "data" / "videos":
        name = path.name
        alternative = ("ВК" + name[2:] if name.startswith("BK") else
                       "BK" + name[2:] if name.startswith("ВК") else name)
        candidate = path.with_name(alternative)
        if candidate.exists():
            path = candidate
    return str(path)
