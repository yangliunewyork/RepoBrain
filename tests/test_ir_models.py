from repobrain.ir.models import (
    ClassInfo,
    FieldInfo,
    FileIR,
    ImportInfo,
    MethodCall,
    MethodInfo,
    ParameterInfo,
    RepoIR,
)


def _sample_repo_ir() -> RepoIR:
    method = MethodInfo(
        name="getName",
        return_type="String",
        parameters=[ParameterInfo(name="x", type="int")],
        modifiers=["public"],
        doc_comment="/** doc */",
        calls=[MethodCall(receiver=None, method="helper")],
        referenced_types=["Widget"],
    )
    inner = ClassInfo(name="Inner", kind="class", qualified_name="Outer.Inner")
    cls = ClassInfo(
        name="Outer",
        kind="class",
        qualified_name="com.example.Outer",
        modifiers=["public"],
        extends=["Base"],
        implements=["Runnable"],
        fields=[FieldInfo(name="name", type="String", modifiers=["private"])],
        methods=[method],
        inner_classes=[inner],
    )
    file_ir = FileIR(
        path="com/example/Outer.java",
        language="java",
        content_hash="abc123",
        package="com.example",
        imports=[ImportInfo(path="java.util.List")],
        classes=[cls],
    )
    return RepoIR(repo_root="/repo", generated_at="2026-01-01T00:00:00Z", files={file_ir.path: file_ir})


def test_iter_classes_flattens_nested_classes():
    file_ir = _sample_repo_ir().files["com/example/Outer.java"]
    names = {c.name for c in file_ir.iter_classes()}
    assert names == {"Outer", "Inner"}


def test_method_signature_rendering():
    method = MethodInfo(name="add", return_type="int", parameters=[ParameterInfo(name="a", type="int"), ParameterInfo(name="b", type="int")], modifiers=["public", "static"])
    assert method.signature == "public static int add(int a, int b)"


def test_constructor_signature_omits_return_type():
    method = MethodInfo(name="Widget", return_type="", is_constructor=True, modifiers=["public"], parameters=[ParameterInfo(name="name", type="String")])
    assert method.signature == "public Widget(String name)"


def test_repo_ir_round_trips_through_dict():
    original = _sample_repo_ir()
    restored = RepoIR.from_dict(original.to_dict())

    assert restored.repo_root == original.repo_root
    assert set(restored.files.keys()) == set(original.files.keys())

    orig_file = original.files["com/example/Outer.java"]
    restored_file = restored.files["com/example/Outer.java"]
    assert restored_file.package == orig_file.package
    assert restored_file.content_hash == orig_file.content_hash
    assert [i.path for i in restored_file.imports] == [i.path for i in orig_file.imports]

    orig_cls, restored_cls = orig_file.classes[0], restored_file.classes[0]
    assert restored_cls.qualified_name == orig_cls.qualified_name
    assert restored_cls.extends == orig_cls.extends
    assert restored_cls.implements == orig_cls.implements
    assert [f.name for f in restored_cls.fields] == [f.name for f in orig_cls.fields]
    assert [m.signature for m in restored_cls.methods] == [m.signature for m in orig_cls.methods]
    assert restored_cls.methods[0].calls == orig_cls.methods[0].calls
    assert [c.name for c in restored_cls.inner_classes] == [c.name for c in orig_cls.inner_classes]
