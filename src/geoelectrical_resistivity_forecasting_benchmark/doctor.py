from __future__ import annotations

import configparser
import ctypes.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import PROJECT_ROOT


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required_for: str


def _command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def run_checks() -> list[Check]:
    in_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in os.uname().release.lower()
    uv_version = _command_output(["uv", "--version"])
    gpu = _command_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    oracle_config = Path.home() / ".adder" / "default.conf"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "geo-bench"
    ingestion_python = cache_root / "venvs" / "ingestion" / "bin" / "python"
    ingestion_packages = (
        _command_output(
            [
                str(ingestion_python),
                "-c",
                "import addereq,oracledb; print('addereq+oracledb installed')",
            ]
        )
        if ingestion_python.is_file()
        else None
    )
    configured_client: Path | None = None
    if oracle_config.is_file():
        parser = configparser.ConfigParser()
        parser.read(oracle_config, encoding="utf-8")
        if parser.has_section("cn"):
            value = parser.get("cn", "ORACLE_CLIENT_LIB_DIR", fallback="").strip()
            if value:
                configured_client = Path(value)
    client_available = bool(ctypes.util.find_library("clntsh")) or bool(
        configured_client and configured_client.is_dir() and list(configured_client.glob("libclntsh.so*"))
    )
    return [
        Check("wsl", in_wsl, os.environ.get("WSL_DISTRO_NAME", os.uname().release), "all commands"),
        Check("python", sys.version_info >= (3, 12), sys.version.split()[0], "core"),
        Check("uv", uv_version is not None, uv_version or "not found", "core"),
        Check("project_mount", PROJECT_ROOT.exists(), str(PROJECT_ROOT), "core"),
        Check("gpu", gpu is not None, gpu or "not available", "foundation/deep models"),
        Check(
            "ingestion_packages",
            ingestion_packages is not None,
            ingestion_packages or f"not available in {ingestion_python.parent.parent}",
            "G1 Oracle ingestion",
        ),
        Check(
            "oracle_client",
            client_available,
            str(configured_client) if configured_client else "libclntsh.so not found",
            "G1 Oracle ingestion",
        ),
        Check(
            "oracle_config",
            oracle_config.is_file(),
            str(oracle_config),
            "G1 Oracle ingestion",
        ),
    ]


def serialise_checks(checks: list[Check]) -> list[dict[str, object]]:
    return [asdict(check) for check in checks]
