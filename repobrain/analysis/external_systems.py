"""Best-effort classification of external systems/integrations a
codebase talks to, inferred from its import statements.

Like `layers.py`'s annotation-based classification, this trades perfect
precision for a signal that's still meaningfully better than leaving an
8B model to guess "this probably talks to a database" from vibes: these
are real import package roots RepoBrain found in the source, not an
inference. A project using something not covered below simply gets no
category for it — see [[Extending RepoBrain]] to add more prefixes.
"""
from __future__ import annotations

from repobrain.ir.models import RepoIR

#: category -> recognized import package prefixes for that category.
#: Deliberately covers commonly-seen JVM ecosystem libraries rather than
#: attempting to be exhaustive.
_CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "Relational Database": (
        "java.sql", "javax.sql", "javax.persistence", "jakarta.persistence",
        "org.hibernate", "org.postgresql", "com.mysql", "com.microsoft.sqlserver",
        "oracle.jdbc", "com.zaxxer.hikari", "org.springframework.jdbc",
        "org.springframework.data.jpa", "org.flywaydb", "org.liquibase",
    ),
    "NoSQL / Cache": (
        "com.mongodb", "org.springframework.data.mongodb", "redis.clients",
        "org.springframework.data.redis", "io.lettuce", "com.datastax.oss",
        "com.couchbase",
    ),
    "Messaging / Streaming": (
        "org.apache.kafka", "org.springframework.kafka", "javax.jms", "jakarta.jms",
        "com.rabbitmq", "org.springframework.amqp",
    ),
    "HTTP / REST Client": (
        "okhttp3", "retrofit2", "org.apache.http", "org.apache.hc", "java.net.http",
        "feign", "org.springframework.web.client", "org.springframework.web.reactive.function.client",
    ),
    "Cloud Provider SDK": (
        "com.amazonaws", "software.amazon.awssdk", "com.google.cloud",
        "com.microsoft.azure", "com.azure",
    ),
    "Email / Notification": (
        "javax.mail", "jakarta.mail", "org.springframework.mail",
    ),
}


def classify_external_systems(repo_ir: RepoIR) -> dict[str, list[str]]:
    """Category -> sorted list of the recognized prefixes actually
    matched in this repo's imports. Categories with no matches are
    omitted entirely, so an empty dict means none of the known
    categories were detected (not that RepoBrain looked and found
    nothing at all — it only recognizes what's listed above)."""
    found: dict[str, set[str]] = {}
    for file_ir in repo_ir.files.values():
        for imp in file_ir.imports:
            for category, prefixes in _CATEGORY_PREFIXES.items():
                matched = next((p for p in prefixes if imp.path.startswith(p)), None)
                if matched:
                    found.setdefault(category, set()).add(matched)
    return {category: sorted(prefixes) for category, prefixes in found.items()}
