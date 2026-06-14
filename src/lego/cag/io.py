from __future__ import annotations

import importlib
import pickle
from pathlib import Path

from .template import CAGTemplate


LEGACY_MODULE_REMAP: dict[str, str] = {
    "src.lego.cag.template": "lego.cag.template",
    "src.lego.cag.node_schema": "lego.cag.node_schema",
    "src.lego.cag.instance": "lego.cag.instance",
    "src.lego.data.operator_context": "lego.data.operator_context",
}


class _RemappingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        target_module = LEGACY_MODULE_REMAP.get(module, module)
        imported = importlib.import_module(target_module)
        return getattr(imported, name)


def save_cag_template(template: CAGTemplate, path: str | Path) -> None:
    template_path = Path(path)
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open('wb') as handle:
        pickle.dump(template, handle)


def load_cag_template(path: str | Path) -> CAGTemplate:
    template_path = Path(path)
    with template_path.open('rb') as handle:
        return _RemappingUnpickler(handle).load()
