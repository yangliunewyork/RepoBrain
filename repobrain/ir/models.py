"""Language-agnostic intermediate representation (IR) for source code.

Every language parser (Java today, others later) must produce these
dataclasses. Downstream analysis and documentation generation only ever
see this IR, never raw source text or a language-specific AST.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ParameterInfo:
    name: str
    type: str


@dataclass
class ImportInfo:
    path: str
    is_static: bool = False
    is_wildcard: bool = False


@dataclass
class MethodCall:
    """One method-invocation expression found in a method body, in source
    order. `receiver` is the simple identifier the call was made on
    (a field, parameter, or local variable name), `"this"` for an explicit
    self-call, or `None` for an implicit self-call or an unresolvable
    receiver expression (e.g. a chained call like `a.b().c()`)."""

    method: str
    receiver: Optional[str] = None


@dataclass
class MethodInfo:
    name: str
    return_type: str
    parameters: list[ParameterInfo] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    #: Simple annotation names (no `@`, no arguments), e.g. "Test",
    #: "GetMapping". Framework/annotation-aware analysis (entry-point
    #: detection, layer classification) reads this instead of guessing
    #: from the method/class name.
    annotations: list[str] = field(default_factory=list)
    #: annotation name -> its single string argument, where recognizable,
    #: e.g. {"PostMapping": "/products"}. Only populated for annotations
    #: with a bare or `value=`/`path=` string argument; used to label
    #: sequence-diagram entry points with an actual HTTP route.
    annotation_args: dict[str, str] = field(default_factory=dict)
    doc_comment: Optional[str] = None
    is_constructor: bool = False
    start_line: int = 0
    end_line: int = 0
    calls: list[MethodCall] = field(default_factory=list)
    referenced_types: list[str] = field(default_factory=list)

    @property
    def signature(self) -> str:
        params = ", ".join(f"{p.type} {p.name}" for p in self.parameters)
        mods = " ".join(self.modifiers)
        prefix = f"{mods} ".lstrip() if mods else ""
        if self.is_constructor:
            return f"{prefix}{self.name}({params})"
        return f"{prefix}{self.return_type} {self.name}({params})"


@dataclass
class FieldInfo:
    name: str
    type: str
    modifiers: list[str] = field(default_factory=list)
    doc_comment: Optional[str] = None


@dataclass
class ClassInfo:
    name: str
    kind: str  # "class" | "interface" | "enum" | "record" | "annotation"
    qualified_name: str
    modifiers: list[str] = field(default_factory=list)
    #: Simple annotation names (no `@`, no arguments), e.g. "RestController".
    annotations: list[str] = field(default_factory=list)
    #: annotation name -> its single string argument, e.g. a class-level
    #: {"RequestMapping": "/products"} base path — combined with a
    #: method's own `annotation_args` to build a full route label.
    annotation_args: dict[str, str] = field(default_factory=dict)
    extends: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    type_parameters: list[str] = field(default_factory=list)
    fields: list[FieldInfo] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)
    inner_classes: list["ClassInfo"] = field(default_factory=list)
    doc_comment: Optional[str] = None
    start_line: int = 0
    end_line: int = 0

    def iter_all(self):
        """Yield this class and every nested class recursively."""
        yield self
        for inner in self.inner_classes:
            yield from inner.iter_all()


@dataclass
class FileIR:
    path: str  # repo-relative, POSIX separators
    language: str
    content_hash: str
    package: Optional[str] = None
    imports: list[ImportInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    def iter_classes(self):
        for c in self.classes:
            yield from c.iter_all()


@dataclass
class RepoIR:
    repo_root: str
    generated_at: str
    files: dict[str, FileIR] = field(default_factory=dict)  # keyed by path

    def iter_classes(self):
        for f in self.files.values():
            yield from f.iter_classes()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "RepoIR":
        files = {}
        for path, fdata in data.get("files", {}).items():
            classes = [_class_from_dict(c) for c in fdata.get("classes", [])]
            imports = [ImportInfo(**i) for i in fdata.get("imports", [])]
            files[path] = FileIR(
                path=fdata["path"],
                language=fdata["language"],
                content_hash=fdata["content_hash"],
                package=fdata.get("package"),
                imports=imports,
                classes=classes,
                parse_errors=fdata.get("parse_errors", []),
            )
        return RepoIR(
            repo_root=data["repo_root"],
            generated_at=data["generated_at"],
            files=files,
        )


def _class_from_dict(data: dict) -> ClassInfo:
    fields_ = [FieldInfo(**f) for f in data.get("fields", [])]
    methods = [
        MethodInfo(
            name=m["name"],
            return_type=m["return_type"],
            parameters=[ParameterInfo(**p) for p in m.get("parameters", [])],
            modifiers=m.get("modifiers", []),
            annotations=m.get("annotations", []),
            annotation_args=m.get("annotation_args", {}),
            doc_comment=m.get("doc_comment"),
            is_constructor=m.get("is_constructor", False),
            start_line=m.get("start_line", 0),
            end_line=m.get("end_line", 0),
            calls=[MethodCall(**c) for c in m.get("calls", [])],
            referenced_types=m.get("referenced_types", []),
        )
        for m in data.get("methods", [])
    ]
    inner = [_class_from_dict(c) for c in data.get("inner_classes", [])]
    return ClassInfo(
        name=data["name"],
        kind=data["kind"],
        qualified_name=data["qualified_name"],
        modifiers=data.get("modifiers", []),
        annotations=data.get("annotations", []),
        annotation_args=data.get("annotation_args", {}),
        extends=data.get("extends", []),
        implements=data.get("implements", []),
        type_parameters=data.get("type_parameters", []),
        fields=fields_,
        methods=methods,
        inner_classes=inner,
        doc_comment=data.get("doc_comment"),
        start_line=data.get("start_line", 0),
        end_line=data.get("end_line", 0),
    )
