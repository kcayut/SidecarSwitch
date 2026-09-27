"""Unit tests for BetterDisplayCLI integration and resilience."""

import unittest
from unittest.mock import MagicMock, patch

from core.betterdisplay import BetterDisplayCLI


class TestBetterDisplayCLI(unittest.TestCase):
    def test_disconnect_virtual_display_uses_the_exact_configured_name(self):
        with patch.object(BetterDisplayCLI, "_resolve_cli_path", return_value="/mock/betterdisplaycli"), \
             patch.object(BetterDisplayCLI, "is_available", return_value=True), \
             patch.object(BetterDisplayCLI, "run_cmd", return_value=(0, "", "")) as run:
            self.assertTrue(BetterDisplayCLI(probe=False).disconnect_virtual_display("SidecarSwitchVirtual"))
            run.assert_called_once_with(["set", "-name=SidecarSwitchVirtual", "-connected=off"], timeout=8.0)

    def test_stop_mirroring_targets_the_display_and_reports_failure(self):
        with patch.object(BetterDisplayCLI, "_resolve_cli_path", return_value="/mock/betterdisplaycli"), \
             patch.object(BetterDisplayCLI, "is_available", return_value=True), \
             patch.object(BetterDisplayCLI, "run_cmd", return_value=(0, "", "")) as run:
            cli = BetterDisplayCLI(probe=False)
            uuid = "11111111-2222-3333-4444-555555555555"
            self.assertTrue(cli.stop_mirroring(uuid))
            run.assert_called_once_with(["set", f"-uuid={uuid}", "-mirror=off"], timeout=8.0)
            run.return_value = (1, "", "rejected")
            self.assertFalse(cli.stop_mirroring("Monitor"))
            self.assertEqual(run.call_args.args[0], ["set", "-name=Monitor", "-mirror=off"])

    def test_resilience_when_probe_times_out(self) -> None:
        """Verify that when the CLI binary exists, is_available remains True even if probe times out."""
        with patch("os.path.isfile", return_value=True), \
             patch("os.access", return_value=True), \
             patch.object(BetterDisplayCLI, "_resolve_cli_path", return_value="/opt/homebrew/bin/betterdisplaycli"), \
             patch.object(BetterDisplayCLI, "run_cmd", return_value=(-1, "", "Command timed out")):
            cli = BetterDisplayCLI()
            self.assertTrue(cli.is_available())
            self.assertTrue(cli.capabilities.cli_available)
            self.assertFalse(cli.capabilities.sidecar_supported)
            self.assertFalse(cli.capabilities.virtual_creation_supported)

    def test_unavailable_when_binary_not_found(self) -> None:
        """Verify is_available is False when binary path is None or does not exist."""
        with patch.object(BetterDisplayCLI, "_resolve_cli_path", return_value=None):
            cli = BetterDisplayCLI()
            self.assertFalse(cli.is_available())
            self.assertFalse(cli.capabilities.cli_available)

    def test_resolve_cli_path_custom_and_app_bundle(self) -> None:
        """Verify resolve_cli_path resolves direct binaries and app bundle paths."""
        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            self.assertEqual(
                BetterDisplayCLI.resolve_cli_path("/custom/bin/betterdisplaycli"),
                "/custom/bin/betterdisplaycli"
            )
            self.assertEqual(
                BetterDisplayCLI.resolve_cli_path("/Applications/BetterDisplay.app"),
                "/Applications/BetterDisplay.app/Contents/MacOS/BetterDisplay"
            )

        with patch("os.path.isfile", return_value=False), patch("os.path.isdir", return_value=False):
            self.assertIsNone(BetterDisplayCLI.resolve_cli_path("/nonexistent/betterdisplaycli"))


if __name__ == "__main__":
    unittest.main()
