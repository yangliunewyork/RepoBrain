"""Java parser built on tree-sitter.

Walks the concrete syntax tree directly (rather than relying purely on
tree-sitter queries) so that nested classes, Javadoc association, and
Java's handful of declaration shapes (class/interface/enum/record/
annotation) can all be handled with plain, easy-to-follow control flow.
"""
from __future__ import annotations

import hashlib
import re

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from repobrain.ir.models import (
    ClassInfo,
    FieldInfo,
    FileIR,
    ImportInfo,
    MethodCall,
    MethodInfo,
    ParameterInfo,
)
from repobrain.logging_setup import get_logger
from repobrain.parsing.base import LanguageParser

logger = get_logger("parsing.java")

_JAVA_LANGUAGE = Language(tsjava.language())

_TYPE_DECLARATION_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "annotation",
}

_MODIFIER_KEYWORDS = {
    "public", "private", "protected", "static", "final", "abstract",
    "synchronized", "native", "transient", "volatile", "strictfp", "default",
    "sealed", "non-sealed",
}


def _text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _doc_comment_for(node: Node, source: bytes) -> str | None:
    """Javadoc is a `block_comment` immediately preceding a declaration."""
    prev = node.prev_sibling
    if prev is not None and prev.type == "block_comment":
        text = _text(prev, source)
        if text.startswith("/**"):
            return text
    return None


def _find_modifiers_node(node: Node) -> Node | None:
    mods_node = node.child_by_field_name("modifiers")
    if mods_node is not None:
        return mods_node
    for child in node.children:
        if child.type == "modifiers":
            return child
    return None


def _modifiers_of(node: Node, source: bytes) -> list[str]:
    mods_node = _find_modifiers_node(node)
    if mods_node is None:
        return []
    result = []
    for child in mods_node.children:
        text = _text(child, source)
        if text in _MODIFIER_KEYWORDS:
            result.append(text)
    return result


def _annotations_of(node: Node, source: bytes) -> list[str]:
    """Simple annotation names (no `@`, no arguments) applied to a class
    or method declaration, e.g. `@RestController` -> "RestController".
    Annotations sit inside the same `modifiers` node as keywords like
    `public`, as either `marker_annotation` (`@Foo`) or `annotation`
    (`@Foo(...)`), each with an `identifier` (or `scoped_identifier` for
    a fully-qualified annotation type) child holding the name.
    """
    mods_node = _find_modifiers_node(node)
    if mods_node is None:
        return []
    result = []
    for child in mods_node.children:
        if child.type in ("marker_annotation", "annotation"):
            name_node = next((c for c in child.children if c.type in ("identifier", "scoped_identifier")), None)
            if name_node is not None:
                result.append(_text(name_node, source).rsplit(".", 1)[-1])
    return result


def _string_literal_text(node: Node | None, source: bytes) -> str | None:
    """Extracts the literal text of a `string_literal` node (its
    `string_fragment` child/children), or None if `node` isn't one —
    used to pull a route path like "/products" out of an annotation
    argument without the surrounding quotes."""
    if node is None or node.type != "string_literal":
        return None
    fragments = [c for c in node.children if c.type == "string_fragment"]
    if not fragments:
        return None
    return "".join(_text(f, source) for f in fragments)


def _annotation_args_of(node: Node, source: bytes) -> dict[str, str]:
    """Best-effort single string argument per annotation, e.g.
    `@PostMapping("/products")` -> {"PostMapping": "/products"} or
    `@RequestMapping(value = "/x", ...)` -> {"RequestMapping": "/x"}.
    Used only to label sequence-diagram entry points with an actual
    route path — annotations with no string argument (or a more complex
    one this doesn't recognize) simply have no entry here, which callers
    must treat as "no path available", not an error.
    """
    mods_node = _find_modifiers_node(node)
    if mods_node is None:
        return {}
    result: dict[str, str] = {}
    for child in mods_node.children:
        if child.type != "annotation":
            continue  # marker_annotation (@Foo, no parens) has no args by definition
        name_node = next((c for c in child.children if c.type in ("identifier", "scoped_identifier")), None)
        args_node = next((c for c in child.children if c.type == "annotation_argument_list"), None)
        if name_node is None or args_node is None:
            continue
        name = _text(name_node, source).rsplit(".", 1)[-1]

        value = None
        for arg in args_node.children:
            if arg.type == "string_literal":
                value = _string_literal_text(arg, source)
                break
            if arg.type == "element_value_pair":
                key_node = arg.child_by_field_name("key")
                value_node = arg.child_by_field_name("value")
                if key_node is not None and _text(key_node, source) in ("value", "path"):
                    value = _string_literal_text(value_node, source)
                    if value is not None:
                        break
        if value is not None:
            result[name] = value
    return result


def _type_list_texts(list_node: Node | None, source: bytes) -> list[str]:
    if list_node is None:
        return []
    inner = None
    for child in list_node.children:
        if child.type == "type_list":
            inner = child
            break
    types = inner.children if inner is not None else list_node.children
    return [_text(t, source) for t in types if t.type.endswith(("type_identifier", "generic_type")) or "type" in t.type]


def _single_type_text(field_node: Node | None, source: bytes) -> str | None:
    """For `superclass` (`extends Foo`): pull the type out, skipping the `extends` keyword."""
    if field_node is None:
        return None
    for child in field_node.children:
        if child.type not in ("extends", "implements"):
            return _text(child, source)
    return None


def _parse_parameters(params_node: Node | None, source: bytes) -> list[ParameterInfo]:
    if params_node is None:
        return []
    result = []
    for p in params_node.children:
        if p.type == "formal_parameter":
            type_node = p.child_by_field_name("type")
            name_node = p.child_by_field_name("name")
            result.append(ParameterInfo(name=_text(name_node, source), type=_text(type_node, source)))
        elif p.type == "spread_parameter":
            type_text = None
            name_text = "args"
            for child in p.children:
                if child.type == "variable_declarator":
                    name_text = _text(child.child_by_field_name("name") or child, source)
                elif child.type not in ("...",):
                    type_text = _text(child, source) if type_text is None else type_text
            result.append(ParameterInfo(name=name_text, type=f"{type_text or 'Object'}..."))
    return result


_SIMPLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _has_resolvable_receiver(object_node: Node | None, source: bytes) -> bool:
    """True for an implicit receiver (no object at all) or a bare
    identifier/`this`; false for a chained expression like `a.b()`,
    whose runtime receiver can't be read off the syntax."""
    if object_node is None:
        return True
    text = _text(object_node, source)
    return text == "this" or bool(_SIMPLE_IDENTIFIER_RE.match(text))


def _collect_body_expressions(
    node: Node, calls: list[MethodCall], type_names: set[str], seen_calls: set[tuple[str | None, str]], source: bytes
) -> None:
    """Walks a method body in source order, recording each distinct call
    (receiver, method name) the first time it's seen — good enough
    fidelity for a sequence diagram without tracking full control flow —
    and every `new X()` type name for the dependency graph's "creates"
    edges. `receiver=None` means an implicit self-call (bare `helper()`,
    no object at all) — calls on a receiver that isn't a bare identifier
    or `this` (e.g. a chained expression like `a.b().c()`) are dropped
    entirely rather than mislabeled as a self-call, since which object
    `c()` actually runs on can't be read off the syntax alone.
    """
    if node.type == "method_invocation":
        name_node = node.child_by_field_name("name")
        object_node = node.child_by_field_name("object")
        if name_node is not None and _has_resolvable_receiver(object_node, source):
            method_name = _text(name_node, source)
            receiver = _text(object_node, source) if object_node is not None else None
            key = (receiver, method_name)
            if key not in seen_calls:
                seen_calls.add(key)
                calls.append(MethodCall(receiver=receiver, method=method_name))
    elif node.type == "object_creation_expression":
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            type_names.add(_text(type_node, source).split("<")[0])
    for child in node.children:
        _collect_body_expressions(child, calls, type_names, seen_calls, source)


class JavaParser(LanguageParser):
    language_name = "java"
    file_extensions = (".java",)

    def __init__(self) -> None:
        self._parser = Parser(_JAVA_LANGUAGE)

    def parse(self, rel_path: str, source: bytes) -> FileIR:
        content_hash = hashlib.sha256(source).hexdigest()
        file_ir = FileIR(path=rel_path, language=self.language_name, content_hash=content_hash)
        try:
            tree = self._parser.parse(source)
            root = tree.root_node
            if root.has_error:
                file_ir.parse_errors.append("syntax error encountered; partial IR extracted")

            for child in root.children:
                if child.type == "package_declaration":
                    file_ir.package = self._parse_package(child, source)
                elif child.type == "import_declaration":
                    file_ir.imports.append(self._parse_import(child, source))
                elif child.type in _TYPE_DECLARATION_KINDS:
                    file_ir.classes.append(
                        self._parse_type_declaration(child, source, file_ir.package, "")
                    )
        except Exception as exc:  # defensive: one bad file must not abort the run
            logger.warning("Failed to parse %s: %s", rel_path, exc)
            file_ir.parse_errors.append(str(exc))
        return file_ir

    def _parse_package(self, node: Node, source: bytes) -> str:
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                return _text(child, source)
        return ""

    def _parse_import(self, node: Node, source: bytes) -> ImportInfo:
        is_static = any(c.type == "static" for c in node.children)
        is_wildcard = any(c.type == "asterisk" for c in node.children)
        path_node = None
        for child in node.children:
            if child.type in ("scoped_identifier", "identifier"):
                path_node = child
                break
        path = _text(path_node, source)
        if is_wildcard:
            path = f"{path}.*"
        return ImportInfo(path=path, is_static=is_static, is_wildcard=is_wildcard)

    def _parse_type_declaration(self, node: Node, source: bytes, package: str | None, outer_qualified: str) -> ClassInfo:
        kind = _TYPE_DECLARATION_KINDS[node.type]
        name_node = node.child_by_field_name("name")
        name = _text(name_node, source)
        qualified = f"{outer_qualified}.{name}" if outer_qualified else (f"{package}.{name}" if package else name)

        modifiers = _modifiers_of(node, source)
        annotations = _annotations_of(node, source)
        annotation_args = _annotation_args_of(node, source)
        type_params = self._parse_type_parameters(node, source)
        extends, implements = self._parse_supertypes(node, kind, source)

        class_info = ClassInfo(
            name=name,
            kind=kind,
            qualified_name=qualified,
            modifiers=modifiers,
            annotations=annotations,
            annotation_args=annotation_args,
            extends=extends,
            implements=implements,
            type_parameters=type_params,
            doc_comment=_doc_comment_for(node, source),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
        )

        if kind == "record":
            class_info.fields.extend(self._parse_record_components(node, source))

        body_node = node.child_by_field_name("body")
        if body_node is not None:
            self._parse_body(body_node, class_info, source, package, qualified)

        return class_info

    def _parse_type_parameters(self, node: Node, source: bytes) -> list[str]:
        tp_node = node.child_by_field_name("type_parameters")
        if tp_node is None:
            return []
        return [_text(c, source) for c in tp_node.children if c.type == "type_parameter"]

    def _parse_supertypes(self, node: Node, kind: str, source: bytes) -> tuple[list[str], list[str]]:
        extends: list[str] = []
        implements: list[str] = []

        superclass_node = node.child_by_field_name("superclass")
        if superclass_node is not None:
            single = _single_type_text(superclass_node, source)
            if single:
                extends.append(single)

        extends_interfaces_node = next((c for c in node.children if c.type == "extends_interfaces"), None)
        if extends_interfaces_node is not None:
            extends.extend(_type_list_texts(extends_interfaces_node, source))

        interfaces_node = node.child_by_field_name("interfaces")
        if interfaces_node is not None:
            implements.extend(_type_list_texts(interfaces_node, source))

        return extends, implements

    def _parse_record_components(self, node: Node, source: bytes) -> list[FieldInfo]:
        params_node = node.child_by_field_name("parameters")
        fields = []
        if params_node is not None:
            for p in params_node.children:
                if p.type == "formal_parameter":
                    type_node = p.child_by_field_name("type")
                    name_node = p.child_by_field_name("name")
                    fields.append(FieldInfo(name=_text(name_node, source), type=_text(type_node, source), modifiers=["final"]))
        return fields

    def _parse_body(self, body_node: Node, class_info: ClassInfo, source: bytes, package: str | None, qualified: str) -> None:
        seen_field_decls: set[int] = set()
        children = list(body_node.children)
        # Members declared after the `;` in an enum body (methods, extra
        # fields) are nested one level deeper, inside `enum_body_declarations`.
        for child in body_node.children:
            if child.type == "enum_body_declarations":
                children.extend(child.children)

        for child in children:
            if child.type == "field_declaration":
                if child.id in seen_field_decls:
                    continue
                seen_field_decls.add(child.id)
                type_node = child.child_by_field_name("type")
                modifiers = _modifiers_of(child, source)
                doc = _doc_comment_for(child, source)
                for decl_child in child.children:
                    if decl_child.type == "variable_declarator":
                        name_node = decl_child.child_by_field_name("name")
                        class_info.fields.append(
                            FieldInfo(name=_text(name_node, source), type=_text(type_node, source), modifiers=modifiers, doc_comment=doc)
                        )
            elif child.type in ("method_declaration", "constructor_declaration"):
                class_info.methods.append(self._parse_method(child, source, class_info.name))
            elif child.type == "enum_constant":
                class_info.fields.append(
                    FieldInfo(name=_text(child.child_by_field_name("name") or child, source), type=class_info.name, modifiers=["static", "final"])
                )
            elif child.type in _TYPE_DECLARATION_KINDS:
                class_info.inner_classes.append(self._parse_type_declaration(child, source, package, qualified))

    def _parse_method(self, node: Node, source: bytes, owner_class_name: str) -> MethodInfo:
        is_constructor = node.type == "constructor_declaration"
        name_node = node.child_by_field_name("name")
        name = _text(name_node, source) if name_node is not None else owner_class_name

        type_node = node.child_by_field_name("type")
        return_type = _text(type_node, source) if type_node is not None else ("void" if not is_constructor else "")

        params = _parse_parameters(node.child_by_field_name("parameters"), source)
        modifiers = _modifiers_of(node, source)
        annotations = _annotations_of(node, source)
        annotation_args = _annotation_args_of(node, source)
        doc = _doc_comment_for(node, source)

        calls: list[MethodCall] = []
        referenced_types: set[str] = set()
        body_node = node.child_by_field_name("body")
        if body_node is not None:
            _collect_body_expressions(body_node, calls, referenced_types, set(), source)

        return MethodInfo(
            name=name,
            return_type=return_type,
            parameters=params,
            modifiers=modifiers,
            annotations=annotations,
            annotation_args=annotation_args,
            doc_comment=doc,
            is_constructor=is_constructor,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            calls=calls,
            referenced_types=sorted(referenced_types),
        )
