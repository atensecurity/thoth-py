"""Regress wheel metadata that lets installation succeed without integrations."""

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from packaging.markers import default_environment

from check_wheel import check_wheel


class WheelContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.wheel = Path(self.temp.name) / "sdk.whl"
        self.metadata = """Metadata-Version: 2.3
Name: atensec-thoth
Version: 0.5.21
Provides-Extra: anthropic
Provides-Extra: autogen
Provides-Extra: claude
Provides-Extra: langchain
Provides-Extra: langgraph
Provides-Extra: openai
Requires-Dist: anyio>=4
Requires-Dist: anthropic>=0.40; extra == "anthropic"
Requires-Dist: pyautogen>=0.2; extra == "autogen"
Requires-Dist: claude-agent-sdk>=0.1.66; extra == "claude"
Requires-Dist: langchain-core>=0.2; extra == "langchain" or extra == "langgraph"
Requires-Dist: langgraph>=0.2; extra == "langgraph"
Requires-Dist: openai>=1.50; extra == "openai"
"""

    def write(self, metadata, package=True):
        with ZipFile(self.wheel, "w") as z:
            z.writestr("atensec_thoth-0.5.21.dist-info/METADATA", metadata)
            if package:
                z.writestr("thoth/__init__.py", "")

    def test_complete_optional_provider_contract(self):
        self.write(self.metadata)
        check_wheel(self.wheel, "0.5.21")

    def test_advertised_extras_without_dependencies_rejected(self):
        self.write("\n".join(line for line in self.metadata.splitlines() if not line.startswith("Requires-Dist:")))
        with self.assertRaisesRegex(ValueError, "expected provider requirements"):
            check_wheel(self.wheel)

    def test_each_missing_provider_rejected(self):
        for provider in ["anthropic", "pyautogen", "claude-agent-sdk", "langchain-core", "langgraph", "openai"]:
            with self.subTest(provider=provider):
                self.write("\n".join(line for line in self.metadata.splitlines() if not line.startswith(f"Requires-Dist: {provider}>")))
                with self.assertRaises(ValueError):
                    check_wheel(self.wheel)

    def test_base_install_must_not_pull_optional_providers(self):
        self.write(self.metadata.replace('; extra == "claude"', ""))
        with self.assertRaisesRegex(ValueError, "base on Python"):
            check_wheel(self.wheel)

    def test_wrong_extra_or_python_marker_rejected(self):
        for marker in ['extra == "openai"', 'extra == "claude" and python_version < "3.13"', 'extra == "claude" and python_full_version != "3.14.0"']:
            with self.subTest(marker=marker):
                self.write(self.metadata.replace('extra == "claude"', marker))
                with self.assertRaises(ValueError):
                    check_wheel(self.wheel)

    def test_host_environment_markers_remain_available(self):
        platform = default_environment()["sys_platform"]
        self.write(self.metadata.replace('extra == "claude"', f'extra == "claude" and sys_platform == "{platform}"'))
        check_wheel(self.wheel)

    def test_name_version_extras_and_package_identity(self):
        for metadata in [
            self.metadata.replace("Name: atensec-thoth", "Name: unrelated"),
            self.metadata.replace("Version: 0.5.21", "Version: 0.5.20"),
            self.metadata.replace("Provides-Extra: claude", "Provides-Extra: wrong"),
        ]:
            with self.subTest(metadata=metadata):
                self.write(metadata)
                with self.assertRaises(ValueError):
                    check_wheel(self.wheel, "0.5.21")
        self.write(self.metadata, package=False)
        with self.assertRaises(ValueError):
            check_wheel(self.wheel)


if __name__ == "__main__":
    unittest.main()
