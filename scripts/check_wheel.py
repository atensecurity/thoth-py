"""Validate the published SDK's integration extras before release."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import sys
from zipfile import BadZipFile, ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

PROVIDERS = {
    "anthropic": {"anthropic"},
    "autogen": {"pyautogen"},
    "claude": {"claude-agent-sdk"},
    "langchain": {"langchain-core"},
    "langgraph": {"langchain-core", "langgraph"},
    "openai": {"openai"},
}


def check_wheel(path: Path, version: str | None = None) -> None:
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1 or "thoth/__init__.py" not in archive.namelist():
            raise ValueError("Expected one SDK metadata file and thoth package")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    if canonicalize_name(metadata.get("Name", "")) != "atensec-thoth":
        raise ValueError("Wheel must contain atensec-thoth")
    actual_version = Version(metadata.get("Version", ""))
    if version is not None and actual_version != Version(version):
        raise ValueError(f"Wheel version {actual_version} does not match {version}")
    extras = {canonicalize_name(extra) for extra in metadata.get_all("Provides-Extra", [])}
    if not set(PROVIDERS) <= extras:
        raise ValueError("Wheel is missing supported integration extras")
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    providers = set().union(*PROVIDERS.values())
    for python_version in ["3.12", "3.13", "3.14"]:
        # Marker.evaluate merges these overrides into its default environment.
        env = {"python_version": python_version, "python_full_version": python_version + ".0"}
        for extra, expected in {"": set(), **PROVIDERS}.items():
            active = {canonicalize_name(req.name) for req in requirements if req.marker is None or req.marker.evaluate({**env, "extra": extra})} & providers
            if active != expected:
                raise ValueError(f"{extra or 'base'} on Python {python_version}: expected provider requirements {sorted(expected)}, got {sorted(active)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    try:
        check_wheel(args.wheel, args.version)
    except (OSError, BadZipFile, ValueError) as exc:
        print(f"Wheel contract failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wheel integration contract passed: {args.wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
