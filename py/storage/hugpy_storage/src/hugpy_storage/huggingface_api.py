"""Hugging Face snapshot audit / repair helpers (operator CLI utilities).

The model list these iterate comes from the installed catalog source
(``get_all_configs`` used to be the engine's discovery); ``huggingface_hub``
and ``requests`` are imported on first use.
"""
import os
from dataclasses import dataclass
from typing import Optional
from abstract_essentials import eatAll, is_dir, parse_url
from hugpy_platform.constants import MODELS_HOME
from hugpy_storage.catalog_source import catalog_rows


def get_all_configs(*_a, **_k) -> dict:
    """Every catalog row, keyed by model_key (was the engine's discovery)."""
    return catalog_rows()

@dataclass
class HFAuditRow:
    file: str
    expected: int
    local: int
    missing: int
    complete_pct: float
    status: str


def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def get_huggingface_module_link(repo_id: str) -> str:
    repo_id = eatAll(repo_id, "/")
    return f"{MODELS_HOME}/{repo_id}"


def get_huggingface_module_dir(repo_id: str) -> str:
    return os.path.join(MODELS_HOME, repo_id)


def audit_hf_snapshot(
    repo_id: str,
    repo_dir: str,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> list[HFAuditRow]:
    from huggingface_hub import HfApi
    api = HfApi()

    incomplete: list[HFAuditRow] = []

    for item in api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        recursive=True,
    ):
        if not type(item).__name__ == "RepoFile":
            continue

        rel_path = item.path
        expected = int(item.size or 0)
        local_path = repo_dir / rel_path

        local = local_path.stat().st_size if local_path.exists() else 0
        missing = max(expected - local, 0)

        if local == expected:
            continue

        if local == 0:
            status = "missing"
        elif local < expected:
            status = "partial"
        else:
            status = "size_mismatch_extra"

        incomplete.append(
            HFAuditRow(
                file=rel_path,
                expected=expected,
                local=local,
                missing=missing,
                complete_pct=0.0 if expected == 0 else min(local / expected * 100, 100.0),
                status=status,
            )
        )

    incomplete.sort(key=lambda row: row.missing, reverse=True)
    return incomplete


def print_hf_audit(incomplete: list[HFAuditRow]) -> None:
    total_missing = sum(row.missing for row in incomplete)
    total_expected = sum(row.expected for row in incomplete)
    total_local = sum(min(row.local, row.expected) for row in incomplete)

    print(f"Incomplete files: {len(incomplete)}")
    print(f"Expected among incomplete: {human_bytes(total_expected)}")
    print(f"Present among incomplete:  {human_bytes(total_local)}")
    print(f"Missing among incomplete:  {human_bytes(total_missing)}")
    print()

    for row in incomplete:
        print(
            f"{row.status:18} "
            f"{row.complete_pct:7.2f}%  "
            f"missing {human_bytes(row.missing):>10} / {human_bytes(row.expected):>10}  "
            f"{row.file}"
        )


def download_missing_only(
    repo_id: str,
    repo_dir: str,
    incomplete: list[HFAuditRow],
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> None:
    files_to_download = [
        row.file
        for row in incomplete
        if row.status in {"missing", "partial", "size_mismatch_extra"}
    ]

    if not files_to_download:
        print(f"{repo_id}: already complete")
        return

    print(f"{repo_id}: downloading {len(files_to_download)} missing/incomplete files")
    print(f"Missing bytes: {human_bytes(sum(row.missing for row in incomplete))}")

    __import__('huggingface_hub').snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=repo_dir,
        allow_patterns=files_to_download,
        force_download=False,
    )
def download_repo(url):
    parsed = parse_url(url)
    repo_id=parsed.get('path')
    repo = repo_id.split('/')[0]
    repo_dir = os.path.join(MODELS_HOME,repo)
    os.makedirs(repo_dir,exist_ok=True)
    __import__('huggingface_hub').snapshot_download(
        repo_id=repo_id,
        local_dir=repo_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    get_all_configs()
    return repo_dir


def confirm_all_links():
    variables = get_all_configs()

    for name, values in variables.items():
        repo_id = values.get("hub_id")
        if not repo_id:
            print(f"{name}: missing hub_id")
            continue

        module_link = get_huggingface_module_link(repo_id)
        module_dir = get_huggingface_module_dir(repo_id)

        if not is_dir(module_dir):
            print(f"Missing local dir: {module_dir}")

        import requests
        resp = requests.get(module_link, timeout=20)
        if resp.status_code != 200:
            input(f"Bad Hugging Face link: {module_link} [{resp.status_code}]")

    return variables


def audit_hfs(download_missing=False, repo_type=None, revision=None):
    variables = confirm_all_links()

    for name, values in variables.items():
        repo_id = values.get("hub_id")
        if not repo_id:
            print(f"{name}: skipped, no hub_id")
            continue

        repo_dir = os.path.join(MODELS_HOME, repo_id)
        os.makedirs(repo_dir, exist_ok=True)

        print()
        print("=" * 80)
        print(f"{name}: {repo_id}")
        print(repo_dir)

        incomplete = audit_hf_snapshot(
            repo_id=repo_id,
            repo_dir=repo_dir,
            repo_type=repo_type,
            revision=revision,
        )

        print_hf_audit(incomplete)

        total_missing = sum(row.missing for row in incomplete)

        if download_missing and incomplete:
            download_missing_only(
                repo_id=repo_id,
                repo_dir=repo_dir,
                incomplete=incomplete,
                repo_type=repo_type,
                revision=revision,
            )

            # Optional re-audit after download.
            incomplete_after = audit_hf_snapshot(
                repo_id=repo_id,
                repo_dir=repo_dir,
                repo_type=repo_type,
                revision=revision,
            )

            if incomplete_after:
                print("Still incomplete after download:")
                print_hf_audit(incomplete_after)
            else:
                print("Complete after missing-file download.")

        else:
            input(f"Missing total for {repo_id}: {human_bytes(total_missing)}")

            
