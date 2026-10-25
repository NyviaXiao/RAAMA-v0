"""Verified content-addressed file and S3-compatible object storage adapters."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Protocol

from adaptive_mas.industrial_rag.models import validate_sha256


class ObjectStoreError(RuntimeError):
    """Object bytes were unavailable or failed integrity verification."""


class ObjectStore(Protocol):
    def put(self, payload: bytes) -> str: ...
    def get(self, content_sha256: str) -> bytes: ...


class LocalContentAddressedStore:
    def __init__(self, root: str | Path, *, max_object_bytes: int = 100 * 1024 * 1024) -> None:
        self.root = Path(root).resolve()
        self.max_object_bytes = max_object_bytes
        if max_object_bytes < 1:
            raise ValueError("max_object_bytes must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        validate_sha256(digest)
        return self.root / digest[:2] / digest[2:]

    @staticmethod
    def _existing_digest(path: Path) -> str | None:
        for attempt in range(5):
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                return None
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.01)
        return None

    def put(self, payload: bytes) -> str:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("object payload must be nonempty bytes")
        if len(payload) > self.max_object_bytes:
            raise ValueError("object payload exceeds configured byte limit")
        digest = hashlib.sha256(payload).hexdigest()
        target = self._path(digest)
        if target.exists():
            if self._existing_digest(target) != digest:
                raise ObjectStoreError("existing content-addressed object failed integrity verification")
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".pending", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(payload)
                output.flush()
            # Windows rename is atomic and fails if another writer already
            # published this digest. POSIX hard-link creation has the same
            # no-overwrite property; both publish only the closed temp file.
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
                temporary.unlink(missing_ok=True)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            if self._existing_digest(target) != digest:
                raise ObjectStoreError("concurrent content-addressed object failed integrity verification")
            return digest
        except OSError:
            if target.exists():
                actual_digest = self._existing_digest(target)
                temporary.unlink(missing_ok=True)
                if actual_digest == digest:
                    return digest
                raise ObjectStoreError("concurrent content-addressed object failed integrity verification")
            temporary.unlink(missing_ok=True)
            raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return digest

    def get(self, content_sha256: str) -> bytes:
        path = self._path(content_sha256)
        payload = path.read_bytes()
        if len(payload) > self.max_object_bytes or hashlib.sha256(payload).hexdigest() != content_sha256:
            raise ObjectStoreError("content-addressed object failed integrity verification")
        return payload


class S3CompatibleObjectStore:
    """S3 adapter requiring a caller-supplied compatible client and bucket."""

    def __init__(self, client: object, bucket: str, *, prefix: str = "raama/objects", max_object_bytes: int = 100 * 1024 * 1024) -> None:
        if not bucket.strip():
            raise ValueError("bucket is required")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.max_object_bytes = max_object_bytes

    @classmethod
    def from_environment(cls, bucket: str, *, endpoint_url: str | None = None, prefix: str = "raama/objects") -> "S3CompatibleObjectStore":
        try:
            import boto3
        except ImportError as error:
            raise RuntimeError("S3-compatible storage unavailable: install the pinned 'boto3' optional extra") from error
        client = boto3.client("s3", endpoint_url=endpoint_url)
        return cls(client, bucket, prefix=prefix)

    def _key(self, digest: str) -> str:
        validate_sha256(digest)
        return f"{self.prefix}/{digest[:2]}/{digest}"

    def put(self, payload: bytes) -> str:
        if not isinstance(payload, bytes) or not payload or len(payload) > self.max_object_bytes:
            raise ValueError("object payload is empty or exceeds configured byte limit")
        digest = hashlib.sha256(payload).hexdigest()
        key = self._key(digest)
        self.client.put_object(  # type: ignore[attr-defined]
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            Metadata={"sha256": digest},
        )
        return digest

    def get(self, content_sha256: str) -> bytes:
        response = self.client.get_object(  # type: ignore[attr-defined]
            Bucket=self.bucket, Key=self._key(content_sha256)
        )
        payload = response["Body"].read(self.max_object_bytes + 1)
        if len(payload) > self.max_object_bytes or hashlib.sha256(payload).hexdigest() != content_sha256:
            raise ObjectStoreError("S3-compatible object failed size or SHA-256 verification")
        return payload
