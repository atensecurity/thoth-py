"""Install a built wheel into a clean consumer environment and optionally test it."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

from check_wheel import PROVIDERS, check_wheel

SDK_ROOT = Path(__file__).resolve().parents[1]
TEST_REQUIREMENTS = ["pytest==9.0.2", "pytest-asyncio==1.3.0", "respx==0.22.0", "moto[sqs]==5.1.22"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--extras", default="")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    extras = sorted(set(filter(None, args.extras.split(","))))
    if any(extra not in PROVIDERS for extra in extras):
        parser.error("Unknown integration extra")
    wheel = args.wheel.resolve()
    check_wheel(wheel)
    required = sorted(set().union(*(PROVIDERS[e] for e in extras)))
    with tempfile.TemporaryDirectory(prefix="thoth-installed-package-") as temporary:
        root = Path(temporary)
        env_dir = root / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = dict(os.environ)
        for key in ["PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS"]:
            env.pop(key, None)
        requirement = str(wheel) + (f"[{','.join(extras)}]" if extras else "")
        subprocess.run([str(python), "-m", "pip", "install", requirement], cwd=root, env=env, check=True)
        # -I prevents the checkout/current directory from satisfying the import.
        verify = """
import importlib.metadata as md
import json
from pathlib import Path
import sys
import thoth
assert Path(thoth.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), thoth.__file__
required, all_providers = json.loads(sys.argv[1]), json.loads(sys.argv[2])
versions = {name: md.version(name) for name in required}
if not required:
    for name in all_providers:
        try:
            md.distribution(name)
        except md.PackageNotFoundError:
            continue
        raise AssertionError(f"Base installation unexpectedly included {name}")
print(json.dumps({"python": sys.version, "sdk": md.version("atensec-thoth"), "providers": versions, "import": thoth.__file__}))
"""
        subprocess.run([str(python), "-I", "-c", verify, json.dumps(required), json.dumps(sorted(set().union(*PROVIDERS.values())))], cwd=root, env=env, check=True)
        subprocess.run([str(python), "-m", "pip", "check"], cwd=root, env=env, check=True)
        if args.run_tests:
            subprocess.run([str(python), "-m", "pip", "install", *TEST_REQUIREMENTS], cwd=root, env=env, check=True)
            subprocess.run([str(python), "-m", "pip", "check"], cwd=root, env=env, check=True)
            shutil.copytree(SDK_ROOT / "tests", root / "tests")
            if "claude" in extras:
                env["THOTH_REQUIRE_CLAUDE_AGENT_SDK"] = "1"
            subprocess.run([str(python), "-m", "pytest", "--asyncio-mode=auto", "tests", "-q", "-ra", "--tb=short"], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
