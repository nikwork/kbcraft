#!/usr/bin/env python3
"""
Export kbcraft FAISS index artifacts to S3 — config-driven, boto3-powered.

Every connection parameter (endpoint, region, credentials, bucket, and the
multipart transfer tuning) is resolved from ``configs/storage.yaml`` through
:class:`kbcraft.config.ConfigFactory` — nothing is hardcoded here. ``boto3``
performs the actual bucket / upload / list operations. Env vars still win over
the yaml (see storage.yaml for the mapping), so MinIO, LocalStack, or a real
cloud bucket are all selected purely via config.

The three artifacts a ``kbcraft index`` run produces are treated as one unit:

    <name>.faiss        # the FAISS index
    <name>_chunks.json  # chunk text + metadata sidecar
    <name>_meta.json    # model / dim / total_chunks sidecar

Object keys are laid out under the configured exports prefix, e.g.
``s3://<bucket>/exports/<name>.faiss``.

Subcommands:
    ensure-bucket   Create the configured bucket if it does not exist (idempotent).
    upload          Upload the three artifacts for --name from --dir.
    verify          Assert all three artifacts are present and non-empty in S3.

Each subcommand exits 0 on success, non-zero on failure, so shell drivers can
wrap them in their own pass/fail checks.

Examples:
    python scripts/s3_export.py ensure-bucket
    python scripts/s3_export.py upload --dir ./vectordb --name test_index_openai
    python scripts/s3_export.py verify --name test_index_openai
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``import kbcraft`` work when run as a bare script (not via -m), by adding
# the project's src/ to the path. A real install (poetry) makes this a no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kbcraft.config import ConfigFactory, S3StorageConfig  # noqa: E402

ARTIFACT_SUFFIXES = (".faiss", "_chunks.json", "_meta.json")


def _load_s3_config(configs_dir: Path) -> S3StorageConfig:
    """Resolve the S3 storage config (env → yaml → defaults) via the factory."""
    return ConfigFactory(configs_dir).load_storage().s3


def _make_client(s3: S3StorageConfig):
    """Build a boto3 S3 client from an :class:`S3StorageConfig`.

    Credentials resolution mirrors boto3's own precedence: an explicit profile
    wins; otherwise static access keys (plus optional session token) are used;
    otherwise boto3 falls back to its default credential chain (IAM role,
    ``~/.aws/credentials``, env vars).
    """
    import boto3
    from botocore.config import Config

    session_kwargs = {}
    if s3.region:
        session_kwargs["region_name"] = s3.region
    if s3.profile:
        session_kwargs["profile_name"] = s3.profile
    session = boto3.Session(**session_kwargs)

    # botocore >= 1.36 defaults to adding data-integrity checksums on every
    # request ("when_supported"). Several S3-compatible stores (MinIO, Beget,
    # etc.) reject those with XAmzContentSHA256Mismatch, so only send checksums
    # when the API actually requires them. Harmless against real AWS S3.
    client_kwargs = {
        "config": Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
    }
    if s3.endpoint_url:
        client_kwargs["endpoint_url"] = s3.endpoint_url
    # Only pass static keys when no profile is configured; a profile carries its
    # own credentials and mixing the two raises.
    if not s3.profile and s3.access_key_id:
        client_kwargs["aws_access_key_id"] = s3.access_key_id
        client_kwargs["aws_secret_access_key"] = s3.secret_access_key
        if s3.session_token:
            client_kwargs["aws_session_token"] = s3.session_token

    return session.client("s3", **client_kwargs)


def _transfer_config(s3: S3StorageConfig):
    """Map the config's transfer block onto a boto3 ``TransferConfig``."""
    from boto3.s3.transfer import TransferConfig

    t = s3.transfer
    return TransferConfig(
        multipart_threshold=t.multipart_threshold,
        multipart_chunksize=t.multipart_chunksize,
        max_concurrency=t.max_concurrency,
        max_bandwidth=t.max_bandwidth,
        use_threads=t.use_threads,
    )


def _key_for(prefix: str, filename: str) -> str:
    """Join the exports prefix and a filename into an S3 object key."""
    prefix = (prefix or "").strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def _artifact_names(name: str) -> list[str]:
    return [f"{name}{suffix}" for suffix in ARTIFACT_SUFFIXES]


def _require_bucket(s3: S3StorageConfig) -> str:
    if not s3.bucket:
        print(
            "error: no S3 bucket configured. Set S3_BUCKET (env/.env) or "
            "backends.s3.bucket in configs/storage.yaml.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return s3.bucket


# ── subcommands ────────────────────────────────────────────────────────────────


def cmd_ensure_bucket(args) -> int:
    from botocore.exceptions import ClientError

    s3cfg = _load_s3_config(args.configs_dir)
    bucket = _require_bucket(s3cfg)
    client = _make_client(s3cfg)

    try:
        client.head_bucket(Bucket=bucket)
        print(f"     bucket exists: {bucket}")
        return 0
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchBucket", "NotFound"):
            client.create_bucket(Bucket=bucket)
            print(f"     created bucket: {bucket}")
            return 0
        print(f"error: cannot access bucket {bucket!r}: {exc}", file=sys.stderr)
        return 1


def cmd_upload(args) -> int:
    s3cfg = _load_s3_config(args.configs_dir)
    bucket = _require_bucket(s3cfg)
    client = _make_client(s3cfg)
    transfer = _transfer_config(s3cfg)
    prefix = args.prefix if args.prefix is not None else s3cfg_exports_prefix(args.configs_dir)

    src_dir = Path(args.dir)
    failures = 0
    for filename in _artifact_names(args.name):
        src = src_dir / filename
        if not src.is_file():
            print(f"error: missing artifact {src}", file=sys.stderr)
            failures += 1
            continue
        key = _key_for(prefix, filename)
        try:
            client.upload_file(str(src), bucket, key, Config=transfer)
            print(f"     uploaded s3://{bucket}/{key}")
        except Exception as exc:  # noqa: BLE001 — surface any boto3/transfer error
            print(f"error: upload failed for {filename}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def cmd_verify(args) -> int:
    s3cfg = _load_s3_config(args.configs_dir)
    bucket = _require_bucket(s3cfg)
    client = _make_client(s3cfg)
    prefix = args.prefix if args.prefix is not None else s3cfg_exports_prefix(args.configs_dir)

    normalized_prefix = (prefix or "").strip("/")
    list_kwargs = {"Bucket": bucket}
    if normalized_prefix:
        list_kwargs["Prefix"] = f"{normalized_prefix}/"
    resp = client.list_objects_v2(**list_kwargs)
    sizes = {obj["Key"]: obj["Size"] for obj in resp.get("Contents", [])}

    failures = 0
    for filename in _artifact_names(args.name):
        key = _key_for(prefix, filename)
        present = key in sizes and sizes[key] > 0
        if not present:
            print(f"error: missing or empty s3 object: {key}", file=sys.stderr)
            failures += 1

    for key, size in sorted(sizes.items()):
        print(f"     {size:>10}  {key}")
    return 1 if failures else 0


def s3cfg_exports_prefix(configs_dir: Path) -> str:
    """Default object-key prefix — the configured exports path (e.g. ``exports``)."""
    return ConfigFactory(configs_dir).load_storage().local.paths.exports


# ── entrypoint ───────────────────────────────────────────────────────────────


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--configs-dir",
        metavar="DIR",
        type=Path,
        default=Path(os.environ.get("KBCRAFT_CONFIGS_DIR", _PROJECT_ROOT / "configs")),
        help="Directory holding storage.yaml. Default: ./configs",
    )
    p.add_argument(
        "--prefix",
        default=None,
        help="S3 key prefix. Default: the configured exports path.",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="s3_export.py",
        description="Export kbcraft FAISS index artifacts to S3 using storage config + boto3.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ensure = sub.add_parser("ensure-bucket", help="Create the configured bucket if missing.")
    _add_common(p_ensure)
    p_ensure.set_defaults(func=cmd_ensure_bucket)

    p_upload = sub.add_parser("upload", help="Upload the three artifacts for --name.")
    _add_common(p_upload)
    p_upload.add_argument("--dir", required=True, help="Directory holding the index artifacts.")
    p_upload.add_argument("--name", required=True, help="Index base name (no extension).")
    p_upload.set_defaults(func=cmd_upload)

    p_verify = sub.add_parser("verify", help="Assert the three artifacts exist in S3.")
    _add_common(p_verify)
    p_verify.add_argument("--name", required=True, help="Index base name (no extension).")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
