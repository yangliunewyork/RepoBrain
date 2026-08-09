from repobrain.parsing.java_parser import JavaParser


def _parse(source: str, path: str = "Test.java"):
    return JavaParser().parse(path, source.encode("utf-8"))


def test_parses_package_and_imports():
    ir = _parse(
        """
        package com.example.model;

        import java.util.Objects;
        import static java.util.Collections.emptyList;
        import java.util.*;
        """
    )
    assert ir.package == "com.example.model"
    paths = {(i.path, i.is_static, i.is_wildcard) for i in ir.imports}
    assert ("java.util.Objects", False, False) in paths
    assert ("java.util.Collections.emptyList", True, False) in paths
    assert ("java.util.*", False, True) in paths


def test_parses_class_with_fields_methods_and_javadoc():
    ir = _parse(
        """
        package com.example;

        /**
         * A widget.
         */
        public class Widget {
            private final String name;

            /**
             * Returns the name.
             */
            public String getName() {
                return name;
            }
        }
        """
    )
    assert len(ir.classes) == 1
    cls = ir.classes[0]
    assert cls.kind == "class"
    assert cls.qualified_name == "com.example.Widget"
    assert "public" in cls.modifiers
    assert cls.doc_comment is not None and "A widget." in cls.doc_comment

    assert [f"{f.type} {f.name}" for f in cls.fields] == ["String name"]
    assert cls.fields[0].modifiers == ["private", "final"]

    assert len(cls.methods) == 1
    method = cls.methods[0]
    assert method.signature == "public String getName()"
    assert method.doc_comment is not None and "Returns the name." in method.doc_comment
    assert not method.is_constructor


def test_parses_constructor_and_calls():
    ir = _parse(
        """
        public class Widget {
            public Widget(String name) {
                this.name = name;
                helper();
            }
            private void helper() {}
        }
        """
    )
    cls = ir.classes[0]
    ctor = next(m for m in cls.methods if m.is_constructor)
    assert ctor.name == "Widget"
    assert ctor.parameters[0].type == "String"
    assert ctor.parameters[0].name == "name"
    assert any(c.method == "helper" and c.receiver is None for c in ctor.calls)


def test_parses_extends_and_implements():
    ir = _parse(
        """
        public class InMemoryWidgetRepository extends BaseRepo implements WidgetRepository, Closeable {
        }
        """
    )
    cls = ir.classes[0]
    assert cls.extends == ["BaseRepo"]
    assert cls.implements == ["WidgetRepository", "Closeable"]


def test_parses_interface_extends_multiple_interfaces():
    ir = _parse(
        """
        public interface Combined extends Runnable, AutoCloseable {
            void run();
        }
        """
    )
    cls = ir.classes[0]
    assert cls.kind == "interface"
    assert set(cls.extends) == {"Runnable", "AutoCloseable"}
    assert cls.methods[0].signature == "void run()"


def test_parses_enum_with_constants_and_implements():
    ir = _parse(
        """
        public enum Color implements Paintable {
            RED, GREEN, BLUE;

            void paint() {}
        }
        """
    )
    cls = ir.classes[0]
    assert cls.kind == "enum"
    assert cls.implements == ["Paintable"]
    constant_names = {f.name for f in cls.fields}
    assert constant_names == {"RED", "GREEN", "BLUE"}
    assert any(m.name == "paint" for m in cls.methods)


def test_parses_record_components_as_fields():
    ir = _parse("public record Point(int x, int y) implements Shape {}")
    cls = ir.classes[0]
    assert cls.kind == "record"
    assert cls.implements == ["Shape"]
    assert [(f.type, f.name) for f in cls.fields] == [("int", "x"), ("int", "y")]


def test_parses_nested_class():
    ir = _parse(
        """
        public class Outer {
            static class Inner {
                void run() {}
            }
        }
        """
    )
    outer = ir.classes[0]
    assert len(outer.inner_classes) == 1
    inner = outer.inner_classes[0]
    assert inner.name == "Inner"
    assert inner.qualified_name == "Outer.Inner"
    all_classes = list(outer.iter_all())
    assert {c.name for c in all_classes} == {"Outer", "Inner"}


def test_parses_generic_type_parameters_and_varargs():
    ir = _parse(
        """
        public class Box<T> {
            void addAll(T... items) {}
        }
        """
    )
    cls = ir.classes[0]
    assert cls.type_parameters == ["T"]
    method = cls.methods[0]
    assert method.parameters[0].type == "T..."
    assert method.parameters[0].name == "items"


def test_object_creation_recorded_as_referenced_type():
    ir = _parse(
        """
        public class Factory {
            Widget create() {
                return new Widget();
            }
        }
        """
    )
    method = ir.classes[0].methods[0]
    assert "Widget" in method.referenced_types


def test_malformed_source_does_not_raise():
    ir = _parse("public class Broken { public void foo( {")
    assert isinstance(ir.parse_errors, list)
    assert ir.parse_errors  # syntax error should be recorded, not raised


def test_annotation_arg_extracted_from_bare_string_literal():
    ir = _parse(
        """
        public class WidgetController {
            @PostMapping("/products")
            public void create() {}
        }
        """
    )
    method = ir.classes[0].methods[0]
    assert method.annotation_args == {"PostMapping": "/products"}


def test_annotation_arg_extracted_from_value_named_pair():
    ir = _parse(
        """
        public class WidgetController {
            @RequestMapping(value = "/x", method = RequestMethod.POST)
            public void other() {}
        }
        """
    )
    method = ir.classes[0].methods[0]
    assert method.annotation_args == {"RequestMapping": "/x"}


def test_annotation_arg_extracted_from_path_named_pair():
    ir = _parse(
        """
        public class WidgetController {
            @Path(path = "/legacy")
            public void jaxrs() {}
        }
        """
    )
    method = ir.classes[0].methods[0]
    assert method.annotation_args == {"Path": "/legacy"}


def test_marker_annotation_has_no_args():
    ir = _parse(
        """
        public class WidgetController {
            @GetMapping
            public void bare() {}
        }
        """
    )
    method = ir.classes[0].methods[0]
    assert method.annotations == ["GetMapping"]
    assert method.annotation_args == {}


def test_class_level_annotation_args_captured_separately_from_method():
    ir = _parse(
        """
        @RequestMapping("/products")
        public class WidgetController {
            @PostMapping("/create")
            public void create() {}
        }
        """
    )
    cls = ir.classes[0]
    assert cls.annotation_args == {"RequestMapping": "/products"}
    assert cls.methods[0].annotation_args == {"PostMapping": "/create"}
