import os
import unittest
from unittest.mock import patch

from zerogpu_runtime import configure_zerogpu_cgroup


class ZeroGPUCgroupTests(unittest.TestCase):
    def configure(self, content, *, exists=False, enabled="true"):
        env = {
            "SPACES_ZERO_GPU": enabled,
            "ZEROGPU_PROC_SELF_CGROUP_PATH": "/etc/host-cgroup",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("zerogpu_runtime.Path.exists", return_value=exists):
                with patch("zerogpu_runtime.Path.read_text", return_value=content) as read:
                    configure_zerogpu_cgroup()
                    return os.environ["ZEROGPU_PROC_SELF_CGROUP_PATH"], read.call_count

    def test_non_zerogpu_is_unchanged(self):
        self.assertEqual(self.configure("", enabled="false"), ("/etc/host-cgroup", 0))

    def test_existing_host_mount_is_preserved(self):
        self.assertEqual(self.configure("", exists=True), ("/etc/host-cgroup", 0))

    def test_missing_host_mount_with_v1_devices(self):
        self.assertEqual(
            self.configure("5:memory:/container\n6:devices:/kubepods/pod/container\n"),
            ("/proc/self/cgroup", 1),
        )

    def test_missing_host_mount_with_v2(self):
        self.assertEqual(
            self.configure("0::/kubepods.slice/pod/container\n"),
            ("/proc/self/cgroup", 1),
        )

    def test_unusable_procfs_is_rejected(self):
        for content in ("0::/\n", "6:devices:/\n", "", "garbage", "0::relative", "0::/../container"):
            with self.subTest(content=content):
                with self.assertRaisesRegex(RuntimeError, "Restart the Hugging Face Space"):
                    self.configure(content)

    def test_unreadable_procfs_is_actionable(self):
        with patch.dict(os.environ, {"SPACES_ZERO_GPU": "true"}, clear=True):
            with patch("zerogpu_runtime.Path.exists", return_value=False):
                with patch("zerogpu_runtime.Path.read_text", side_effect=FileNotFoundError):
                    with self.assertRaisesRegex(RuntimeError, "unreadable"):
                        configure_zerogpu_cgroup()


if __name__ == "__main__":
    unittest.main()
