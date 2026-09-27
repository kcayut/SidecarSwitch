"""Unit tests for SidecarSwitch Detector and Topology Generation."""

import unittest
from unittest.mock import MagicMock, patch

from core.betterdisplay import BetterDisplayCLI
from core.config import Config
from core.detector import DisplayDetector
from core.models import DisplayInfo, IpadConfig


class TestSidecarSwitchDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config(
            ipad=IpadConfig(name="Cayut iPad", sidecar_uuid="UUID-1234", usb_serial="USB-SERIAL-999"),
            ignore_list=["Dummy", "Virtual"],
            virtual_display_name="SidecarSwitchVirtual",
        )
        self.mock_bd_cli = MagicMock(spec=BetterDisplayCLI)
        self.detector = DisplayDetector(self.config, self.mock_bd_cli)

    def test_display_ignore_filtering(self) -> None:
        """Verify that virtual screens and ignore list matchers are ignored."""
        d1 = DisplayInfo(display_id=1, name="ASUS PG279Q")
        d2 = DisplayInfo(display_id=2, name="SidecarSwitchVirtual")
        d3 = DisplayInfo(display_id=3, name="HDMI Dummy Plug")

        self.assertFalse(self.detector.is_display_ignored(d1))
        self.assertTrue(self.detector.is_display_ignored(d2))
        self.assertTrue(self.detector.is_display_ignored(d3))

    def test_deterministic_topology_signature(self) -> None:
        """Verify deterministic tuple comparison without Python hash() dependency."""
        d1 = DisplayInfo(display_id=10, name="ASUS Monitor")
        d2 = DisplayInfo(display_id=2, name="Dell Monitor")

        sig1 = (tuple(sorted([d1.display_id, d2.display_id])), True)
        sig2 = (tuple(sorted([d2.display_id, d1.display_id])), True)

        self.assertEqual(sig1, ( (2, 10), True ))
        self.assertEqual(sig1, sig2)

    def test_usb_parsing_with_configured_serial(self) -> None:
        """Verify exact USB serial matching for configured iPad."""
        mock_ioreg = """
+-o AppleT8132USBXHCI@00000000 <class AppleT8132USBXHCI>
+-o iPad@01100000 <class IOUSBHostDevice>
    {
      "idVendor" = 1452
      "idProduct" = 4780
      "kUSBSerialNumberString" = "USB-SERIAL-999"
      "kUSBProductString" = "iPad"
    }
"""
        with patch("subprocess.check_output", return_value=mock_ioreg):
            devices = self.detector.parse_usb_devices()
            self.assertEqual(len(devices), 1)
            self.assertEqual(devices[0]["serial"], "USB-SERIAL-999")
            self.assertEqual(devices[0]["vendor_id"], 1452)

    def test_vendor_2198_virtual_display_detection(self) -> None:
        """Verify that displays with BetterDisplay vendor 2198 are recognized as virtual."""
        import ctypes

        def mock_get_online_displays(max_d, d_ids, count_ref):
            d_ids[0] = 100
            ctypes.cast(count_ref, ctypes.POINTER(ctypes.c_uint32))[0] = 1
            return 0

        # Setup mock CoreGraphics
        mock_cg = MagicMock()
        mock_cg.CGGetOnlineDisplayList.side_effect = mock_get_online_displays
        mock_cg.CGDisplayIsActive.return_value = 1
        mock_cg.CGDisplayMirrorsDisplay.return_value = 0
        mock_cg.CGDisplayIsMain.return_value = 1
        mock_cg.CGDisplayIsBuiltin.return_value = 0
        mock_cg.CGDisplayPixelsWide.return_value = 1920
        mock_cg.CGDisplayPixelsHigh.return_value = 1080
        mock_cg.CGDisplayVendorNumber.return_value = 2198
        mock_cg.CGDisplayModelNumber.return_value = 5094

        self.detector._cg = mock_cg
        self.mock_bd_cli.get_display_identifiers.return_value = []

        displays = self.detector.get_online_displays()
        self.assertEqual(len(displays), 1)
        self.assertTrue(displays[0].is_virtual)
        self.assertEqual(displays[0].name, "SidecarSwitchVirtual")

    def test_ghost_headless_display_ignored(self) -> None:
        """Zero-size and unidentified headless placeholders remain excluded."""
        import ctypes

        def mock_get_online_displays(max_d, d_ids, count_ref):
            d_ids[0] = 1
            ctypes.cast(count_ref, ctypes.POINTER(ctypes.c_uint32))[0] = 1
            return 0

        mock_cg = MagicMock()
        mock_cg.CGGetOnlineDisplayList.side_effect = mock_get_online_displays
        # Inactive or 0x0
        mock_cg.CGDisplayIsActive.return_value = 0
        mock_cg.CGDisplayMirrorsDisplay.return_value = 0
        mock_cg.CGDisplayIsMain.return_value = 1
        mock_cg.CGDisplayIsBuiltin.return_value = 0
        mock_cg.CGDisplayPixelsWide.return_value = 0
        mock_cg.CGDisplayPixelsHigh.return_value = 0
        mock_cg.CGDisplayVendorNumber.return_value = 0
        mock_cg.CGDisplayModelNumber.return_value = 0

        self.detector._cg = mock_cg
        self.mock_bd_cli.get_display_identifiers.return_value = []

        for width, height in ((0, 0), (1, 1), (1920, 1080)):
            with self.subTest(size=(width, height)):
                mock_cg.CGDisplayPixelsWide.return_value = width
                mock_cg.CGDisplayPixelsHigh.return_value = height
                self.assertEqual(self.detector.get_online_displays(), [])

    def test_online_physical_display_stays_visible_without_admitting_placeholders(self) -> None:
        import ctypes

        def get_online(max_d, d_ids, count_ref):
            d_ids[0] = 3
            ctypes.cast(count_ref, ctypes.POINTER(ctypes.c_uint32))[0] = 1
            return 0

        cg = MagicMock()
        cg.CGGetOnlineDisplayList.side_effect = get_online
        cg.CGDisplayIsMain.return_value = 0
        cg.CGDisplayIsBuiltin.return_value = 0
        cg.CGDisplayVendorNumber.return_value = 0x0D2D
        cg.CGDisplayModelNumber.return_value = 7218
        self.detector._cg = cg
        self.mock_bd_cli.get_display_identifiers.return_value = [{"displayID": 3, "name": "CH7218"}]

        for active, source, width, height, expected in (
            (False, 4, 1920, 1080, True),  # Hardware mirror target.
            (True, 4, 1920, 1080, True),   # Software mirror target.
            (True, 0, 1920, 1080, True),   # Independent physical output.
            (False, 0, 1920, 1080, True),  # Inactive output after restoring a saved layout.
            (False, 4, 0, 0, False),
            (False, 4, 1, 1, False),
        ):
            with self.subTest(active=active, source=source, size=(width, height)):
                cg.CGDisplayIsActive.return_value = active
                cg.CGDisplayMirrorsDisplay.return_value = source
                cg.CGDisplayPixelsWide.return_value = width
                cg.CGDisplayPixelsHigh.return_value = height
                displays = self.detector.get_online_displays()
                self.assertEqual(len(displays), int(expected))
                if expected:
                    self.assertFalse(self.detector.is_display_ignored(displays[0]))
                    self.assertEqual(displays[0].is_active, active)
                    self.assertEqual(displays[0].mirror_source_id, source or None)

        cg.CGDisplayIsActive.return_value = False
        cg.CGDisplayMirrorsDisplay.return_value = 0
        cg.CGDisplayPixelsWide.return_value = 1920
        cg.CGDisplayPixelsHigh.return_value = 1080
        # Hardware identity alone, or a BetterDisplay identifier alone, keeps an online output.
        self.mock_bd_cli.get_display_identifiers.return_value = []
        self.assertEqual(len(self.detector.get_online_displays()), 1)
        cg.CGDisplayVendorNumber.return_value = cg.CGDisplayModelNumber.return_value = 0
        self.mock_bd_cli.get_display_identifiers.return_value = [{"displayID": 3, "name": "CH7218"}]
        self.assertEqual(len(self.detector.get_online_displays()), 1)
        # Do not turn an inactive virtual/Sidecar session into a ready display.
        for vendor, model in ((2198, 7218), (0x6161706C, 7218), (0x0D2D, 0x69506164)):
            with self.subTest(vendor=vendor, model=model):
                cg.CGDisplayVendorNumber.return_value = vendor
                cg.CGDisplayModelNumber.return_value = model
                self.assertEqual(self.detector.get_online_displays(), [])


if __name__ == "__main__":
    unittest.main()
