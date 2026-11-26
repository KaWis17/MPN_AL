"""Label schema: which cell classes exist and how annotation names map onto them.

A schema file lives in ``configs/labels/``. Class ids start at 1; 0 is background. ``aliases``
let several annotation names share one class (e.g. all megakaryocyte subtypes while the task
is still "megakaryocyte or not"). ``ignore`` names mark regions excluded from the loss, and
``roi`` names mark the area that was annotated exhaustively.

The schema travels inside every head checkpoint, so predictions are self-describing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_PALETTE = ((230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230))


@dataclass(frozen=True)
class LabelClass:
    id: int
    name: str
    color: tuple[int, int, int]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelSchema:
    name: str
    classes: tuple[LabelClass, ...]
    ignore: tuple[str, ...] = ()
    roi: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [c.id for c in self.classes]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(f"label schema {self.name!r}: class ids must be 1..K in order, got {ids}")
        names = [n for c in self.classes for n in (c.name, *c.aliases)] + list(self.ignore) + list(self.roi)
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"label schema {self.name!r}: names used more than once: {repeated}")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def by_id(self, class_id: int) -> LabelClass:
        return self.classes[class_id - 1]

    def role_of(self, annotation_name: str | None) -> tuple[str, LabelClass | None]:
        """Classify an annotation name: ``("class", c)``, ``("ignore", None)``, ``("roi", None)`` or ``("unknown", None)``."""
        if annotation_name is not None:
            for label in self.classes:
                if annotation_name == label.name or annotation_name in label.aliases:
                    return "class", label
            if annotation_name in self.ignore:
                return "ignore", None
            if annotation_name in self.roi:
                return "roi", None
        return "unknown", None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": [
                {"id": c.id, "name": c.name, "color": list(c.color), "aliases": list(c.aliases)} for c in self.classes
            ],
            "ignore": list(self.ignore),
            "roi": list(self.roi),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "LabelSchema":
        classes = []
        for index, entry in enumerate(document.get("classes", []), start=1):
            color = entry.get("color") or _PALETTE[(index - 1) % len(_PALETTE)]
            classes.append(
                LabelClass(
                    id=int(entry.get("id", index)),
                    name=str(entry["name"]),
                    color=tuple(int(v) for v in color),
                    aliases=tuple(entry.get("aliases") or ()),
                )
            )
        if not classes:
            raise ValueError("a label schema needs at least one class")
        return cls(
            name=str(document.get("name", "labels")),
            classes=tuple(classes),
            ignore=tuple(document.get("ignore") or ()),
            roi=tuple(document.get("roi") or ()),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelSchema":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    @classmethod
    def single_class(cls, name: str = "cell", color: tuple[int, int, int] = (0, 200, 255)) -> "LabelSchema":
        """Class-agnostic schema: every instance is a ``name``."""
        return cls(name=f"{name}_only", classes=(LabelClass(1, name, color),))
