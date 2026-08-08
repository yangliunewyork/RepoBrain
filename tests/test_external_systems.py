from datetime import datetime, timezone

from repobrain.analysis.external_systems import classify_external_systems
from repobrain.ir.models import RepoIR
from repobrain.parsing.java_parser import JavaParser


def _repo_ir(sources: dict[str, str]) -> RepoIR:
    parser = JavaParser()
    files = {path: parser.parse(path, src.encode("utf-8")) for path, src in sources.items()}
    return RepoIR(repo_root="/repo", generated_at=datetime.now(timezone.utc).isoformat(), files=files)


def test_detects_relational_database_import():
    repo_ir = _repo_ir({"A.java": "import javax.persistence.Entity;\npublic class A {}"})
    result = classify_external_systems(repo_ir)
    assert result == {"Relational Database": ["javax.persistence"]}


def test_detects_messaging_import():
    repo_ir = _repo_ir({"A.java": "import org.apache.kafka.clients.producer.KafkaProducer;\npublic class A {}"})
    result = classify_external_systems(repo_ir)
    assert result == {"Messaging / Streaming": ["org.apache.kafka"]}


def test_detects_http_client_import():
    repo_ir = _repo_ir({"A.java": "import okhttp3.OkHttpClient;\npublic class A {}"})
    result = classify_external_systems(repo_ir)
    assert result == {"HTTP / REST Client": ["okhttp3"]}


def test_detects_cloud_sdk_import():
    repo_ir = _repo_ir({"A.java": "import com.amazonaws.services.s3.AmazonS3;\npublic class A {}"})
    result = classify_external_systems(repo_ir)
    assert result == {"Cloud Provider SDK": ["com.amazonaws"]}


def test_no_matches_returns_empty_dict():
    repo_ir = _repo_ir({"A.java": "import java.util.List;\npublic class A {}"})
    assert classify_external_systems(repo_ir) == {}


def test_multiple_categories_across_files():
    repo_ir = _repo_ir(
        {
            "A.java": "import javax.persistence.Entity;\npublic class A {}",
            "B.java": "import org.apache.kafka.clients.producer.KafkaProducer;\npublic class B {}",
        }
    )
    result = classify_external_systems(repo_ir)
    assert set(result.keys()) == {"Relational Database", "Messaging / Streaming"}


def test_duplicate_imports_across_files_are_deduplicated():
    repo_ir = _repo_ir(
        {
            "A.java": "import javax.persistence.Entity;\npublic class A {}",
            "B.java": "import javax.persistence.Table;\npublic class B {}",
        }
    )
    result = classify_external_systems(repo_ir)
    assert result == {"Relational Database": ["javax.persistence"]}
