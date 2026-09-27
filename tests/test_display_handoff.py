"""Display handoff regressions; all hardware operations are mocked."""
import json
import runpy
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from core.betterdisplay import BetterDisplayCLI
from core.config import Config
from core.detector import DisplayDetector
from core.models import ActualState, DisplayInfo, DisplayRole, IpadConfig, OperationMode
from core.state_engine import StateEngine


class DisplayHandoffTests(unittest.TestCase):
    physical_uuid = "11111111-2222-3333-4444-555555555555"

    def setUp(self):
        self.cfg = Config(mode=OperationMode.MANUAL_ONLY, auto_detect_ipad=False,
                          ipad=IpadConfig(name="Target iPad", sidecar_uuid="TARGET"),
                          retry_interval=0, debounce_seconds=0)
        self.bd = MagicMock(spec=BetterDisplayCLI)
        self.bd.set_main_display.return_value = True
        self.bd.stop_mirroring.return_value = True
        self.detector = MagicMock(spec=DisplayDetector)
        self.engine = StateEngine(self.cfg, self.detector, self.bd)
        self.state = self.make_state()

        def disconnect_virtual(name):
            self.assertEqual(name, self.cfg.virtual_display_name)
            self.assertEqual(self.state.main_display.uuid, self.physical_uuid)
            self.assertTrue(self.state.main_display.is_active)
            self.assertIsNone(self.state.main_display.mirror_source_id)
            self.state.virtual_display_connected = False
            return True

        self.bd.disconnect_virtual_display.side_effect = disconnect_virtual
        self.detector.observe.side_effect = lambda **_: (
            self.state, (tuple(sorted(d.display_id for d in self.state.physical_displays)), False))
        for target in ("core.state_engine.write_atomic_status", "core.state_engine.notify_error",
                       "core.state_engine.StateEngine._wait"):
            mock = patch(target)
            mock.start()
            self.addCleanup(mock.stop)

    def make_state(self, *, connected=True, physical_main=True, mirror_source=None,
                   display_id=3, other_first=False):
        physical = DisplayInfo(display_id, "CH7218", uuid=self.physical_uuid,
                               is_main=physical_main, is_active=mirror_source is None,
                               mirror_source_id=mirror_source, width=1920, height=1080)
        virtual = DisplayInfo(4, self.cfg.virtual_display_name, is_virtual=True,
                              is_main=not physical_main, width=1920, height=1080)
        physicals = [physical]
        if other_first:
            physicals.insert(0, DisplayInfo(8, "Other monitor", uuid="OTHER", width=1920, height=1080))
        online = physicals + [virtual]
        if connected:
            online.append(DisplayInfo(5, "Target iPad", is_sidecar=True))
        return ActualState(physical_displays=physicals, online_displays=online,
                           main_display=physical if physical_main else virtual,
                           virtual_display_exists=True, virtual_display_connected=True,
                           sidecar_available=True, sidecar_connected=connected,
                           sidecar_display_online=connected)

    def wire_recovery(self, *, new_display_id=3, other_first=False):
        def disconnect(specifier):
            self.assertEqual(specifier, "TARGET")
            self.state = self.make_state(connected=False, physical_main=False, mirror_source=4)
            return True

        def stop_mirroring(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state = self.make_state(connected=self.state.sidecar_connected, physical_main=False,
                                         display_id=new_display_id, other_first=other_first)
            return True

        def set_main(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state = self.make_state(connected=self.state.sidecar_connected,
                                         display_id=new_display_id, other_first=other_first)
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.bd.set_main_display.side_effect = set_main

    def disconnect(self):
        self.engine.set_user_override(DisplayRole.IPAD_DISCONNECTED, async_transition=False)

    def assert_recovered(self):
        self.assertFalse(self.engine.actual.sidecar_connected)
        self.assertFalse(self.engine.actual.virtual_display_connected)
        self.assertEqual(self.engine.actual.main_display.uuid, self.physical_uuid)
        self.assertTrue(self.engine.actual.main_display.is_active)
        self.assertIsNone(self.engine.actual.main_display.mirror_source_id)
        self.assertIsNone(self.engine.runtime.last_error)
        self.assertEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")
        self.bd.connect_sidecar.assert_not_called()

    def test_manual_disconnect_recovers_physical_after_saved_mirror_layout_returns(self):
        self.wire_recovery()
        self.disconnect()
        self.assert_recovered()
        self.assertEqual(self.bd.mock_calls, [call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name),
                                             call.disconnect_sidecar("TARGET"),
                                             call.stop_mirroring(self.physical_uuid),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])

    def test_reverse_virtual_mirror_is_detached_before_disconnect(self):
        virtual = next(d for d in self.state.online_displays if d.is_virtual)
        virtual.is_active = False
        virtual.mirror_source_id = self.state.main_display.display_id

        def stop_mirroring(specifier):
            self.assertEqual(specifier, virtual.name)
            virtual.is_active = True
            virtual.mirror_source_id = None
            return True

        def disconnect(_):
            self.assertIsNone(virtual.mirror_source_id)
            self.assertFalse(self.state.virtual_display_connected)
            self.state.sidecar_connected = self.state.sidecar_display_online = False
            return True

        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.bd.disconnect_sidecar.side_effect = disconnect
        self.disconnect()
        self.assert_recovered()
        self.assertEqual(self.bd.mock_calls, [call.stop_mirroring(virtual.name),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name),
                                             call.disconnect_sidecar("TARGET")])

    def test_automatic_takeover_detaches_reverse_sidecar_mirror(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.AUTOMATIC
        ipad = next(d for d in self.state.online_displays if d.is_sidecar)
        ipad.is_active = False
        ipad.mirror_source_id = self.state.main_display.display_id

        def stop_mirroring(specifier):
            self.assertEqual(specifier, ipad.name)
            ipad.is_active = True
            ipad.mirror_source_id = None
            return True

        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.bd.set_main_display.return_value = True
        self.engine.evaluate(async_transition=False)
        self.assertTrue(self.engine.actual.sidecar_connected)
        self.assertEqual(self.engine.actual.main_display.uuid, self.physical_uuid)
        self.assertIsNone(self.engine.runtime.last_error)
        self.assertEqual(self.bd.mock_calls, [call.stop_mirroring(ipad.name),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])

    def test_physical_only_mirror_group_is_preserved(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.AUTOMATIC
        follower = DisplayInfo(8, "Other monitor", is_active=False, mirror_source_id=3)
        self.state.physical_displays.append(follower)
        self.state.online_displays.append(follower)
        self.engine.evaluate(async_transition=False)
        self.assertEqual(self.bd.mock_calls, [call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])
        self.assertEqual(follower.mirror_source_id, 3)

    def test_inactive_physical_without_mirror_source_is_reactivated(self):
        self.state = self.make_state(physical_main=False)
        self.state.physical_displays[0].is_active = False

        def stop_mirroring(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state.physical_displays[0].is_active = True
            return True

        def set_main(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state = self.make_state()
            return True

        def disconnect(_):
            self.assertFalse(self.state.virtual_display_connected)
            self.state.sidecar_connected = self.state.sidecar_display_online = False
            return True

        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.bd.set_main_display.side_effect = set_main
        self.bd.disconnect_sidecar.side_effect = disconnect
        self.disconnect()
        self.assert_recovered()
        self.assertEqual(self.bd.mock_calls, [call.stop_mirroring(self.physical_uuid),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name),
                                             call.disconnect_sidecar("TARGET")])

    def test_missing_physical_is_retried_using_its_pinned_uuid(self):
        def disconnect(_):
            self.state = self.make_state(connected=False, physical_main=False)
            self.state.physical_displays = []
            self.state.online_displays = [self.state.main_display]
            return True

        def stop_mirroring(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state = self.make_state(connected=False, physical_main=False)
            return True

        def set_main(specifier):
            self.assertEqual(specifier, self.physical_uuid)
            self.state = self.make_state(connected=False)
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.bd.set_main_display.side_effect = set_main
        self.disconnect()
        self.assert_recovered()
        self.assertEqual(self.bd.mock_calls, [call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name),
                                             call.disconnect_sidecar("TARGET"),
                                             call.stop_mirroring(self.physical_uuid),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])

    def test_recovery_keeps_uuid_when_unmirroring_changes_display_id_and_order(self):
        self.wire_recovery(new_display_id=23, other_first=True)
        self.disconnect()
        self.assert_recovered()
        self.assertEqual(self.engine.actual.main_display.display_id, 23)
        self.assertEqual(self.bd.set_main_display.call_args_list, [call(self.physical_uuid), call(self.physical_uuid)])
        self.assertEqual(self.engine.runtime.topology_generation,
                         self.engine.runtime.user_override.topology_generation)

    def test_failed_recovery_stays_unsatisfied_when_manual_policy_becomes_no_change(self):
        self.wire_recovery()
        set_main = self.bd.set_main_display.side_effect
        self.bd.set_main_display.side_effect = lambda specifier: set_main(specifier) if self.state.sidecar_connected else False
        self.disconnect()
        self.assertFalse(self.engine.actual.sidecar_connected)
        error = self.engine.runtime.last_error
        self.assertIn("after disconnecting iPad", error)
        self.assertNotEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")
        self.engine.runtime.user_override = None
        self.engine.evaluate(async_transition=False)
        self.assertEqual(self.engine.desired.target_display_role, DisplayRole.NO_CHANGE)
        self.assertEqual(self.engine.runtime.last_error, error)
        self.assertNotEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")
        self.bd.connect_sidecar.assert_not_called()

    def test_automatic_physical_takeover_unmirrors_before_setting_main(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.AUTOMATIC
        self.state = self.make_state(connected=False, physical_main=False, mirror_source=4)
        self.wire_recovery()
        self.engine.evaluate(async_transition=False)
        self.assert_recovered()
        self.assertEqual(self.bd.mock_calls, [call.stop_mirroring(self.physical_uuid),
                                             call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])

    def test_accepted_commands_without_observed_recovery_are_not_success(self):
        self.wire_recovery()
        self.bd.stop_mirroring.side_effect = self.bd.set_main_display.side_effect = None
        self.bd.stop_mirroring.return_value = self.bd.set_main_display.return_value = True
        self.disconnect()
        self.assertEqual(self.engine.actual.physical_displays[0].mirror_source_id, 4)
        self.assertIn("after disconnecting iPad", self.engine.runtime.last_error)
        self.assertNotEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")
        self.bd.connect_sidecar.assert_not_called()

    def test_manual_mode_without_command_preserves_existing_mirror_layout(self):
        self.state = self.make_state(connected=False, physical_main=False, mirror_source=4)
        self.engine.evaluate(async_transition=False)
        self.assertEqual(self.engine.desired.target_display_role, DisplayRole.NO_CHANGE)
        self.assertEqual(self.bd.mock_calls, [])
        self.assertEqual(self.engine.actual.physical_displays[0].mirror_source_id, 4)

    def test_failed_unmirror_before_disconnect_keeps_ipad_connected(self):
        self.state = self.make_state(physical_main=False, mirror_source=4)
        self.bd.stop_mirroring.return_value = False
        self.disconnect()
        self.assertTrue(self.engine.actual.sidecar_connected)
        self.assertIn("iPad was not disconnected", self.engine.runtime.last_error)
        self.bd.disconnect_sidecar.assert_not_called()
        self.bd.set_main_display.assert_not_called()
        self.bd.stop_mirroring.assert_called_with(self.physical_uuid)

    def test_prefer_ipad_does_not_reconnect_after_its_own_display_id_change(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.PREFER_IPAD
        self.wire_recovery(new_display_id=23, other_first=True)
        self.disconnect()
        self.engine.evaluate(async_transition=False)
        self.assert_recovered()
        self.assertEqual(self.engine.desired.target_display_role, DisplayRole.PHYSICAL)
        self.assertEqual(self.engine.runtime.user_override.target_role, DisplayRole.IPAD_DISCONNECTED)
        self.assertEqual(self.engine.runtime.topology_generation,
                         self.engine.runtime.user_override.topology_generation)

    def test_failed_disconnect_command_still_recovers_observed_disconnection(self):
        self.wire_recovery()
        disconnect = self.bd.disconnect_sidecar.side_effect

        def timed_out(specifier):
            disconnect(specifier)
            return False

        self.bd.disconnect_sidecar.side_effect = timed_out
        self.disconnect()
        self.assert_recovered()
        self.bd.disconnect_sidecar.assert_called_once_with("TARGET")
        self.bd.stop_mirroring.assert_called_once_with(self.physical_uuid)
        self.assertEqual(self.bd.set_main_display.call_args_list, [call(self.physical_uuid), call(self.physical_uuid)])

    def test_missing_physical_after_disconnect_keeps_error_across_watchdogs(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.PREFER_IPAD

        def disconnect(_):
            self.state = self.make_state(connected=False, physical_main=False)
            self.state.physical_displays = []
            self.state.online_displays = [self.state.main_display]
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.disconnect()
        error = self.engine.runtime.last_error
        self.assertIn("after disconnecting iPad", error)
        previous_calls = list(self.bd.mock_calls)
        for _ in range(3):
            self.engine.evaluate(trigger="watchdog", async_transition=False)
            self.assertEqual(self.engine.runtime.last_error, error)
            self.assertNotEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")
            self.assertEqual(self.bd.mock_calls, previous_calls)
        self.bd.connect_sidecar.assert_called_once_with("TARGET")
        self.bd.disconnect_sidecar.assert_called_once_with("TARGET")

    def test_virtual_is_not_retired_until_the_physical_main_is_observed(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.AUTOMATIC
        self.state = self.make_state(connected=False, physical_main=False)
        self.bd.set_main_display.return_value = True
        self.engine.evaluate(async_transition=False)
        self.bd.set_main_display.assert_called_once_with(self.physical_uuid)
        self.bd.disconnect_virtual_display.assert_not_called()
        self.assertTrue(self.engine.actual.virtual_display_connected)
        self.assertIsNotNone(self.engine.runtime.last_error)

    def test_virtual_retirement_preserves_the_current_physical_main(self):
        self.cfg.mode = self.engine.runtime.mode = OperationMode.AUTOMATIC
        self.state = self.make_state(connected=False, other_first=True)
        self.engine.evaluate(async_transition=False)
        self.assertEqual(self.engine.actual.main_display.uuid, self.physical_uuid)
        self.assertFalse(self.engine.actual.virtual_display_connected)
        self.assertEqual(self.bd.mock_calls, [call.set_main_display(self.physical_uuid),
                                             call.disconnect_virtual_display(self.cfg.virtual_display_name)])

    def test_headless_disconnect_keeps_virtual_fallback_connected(self):
        self.state = self.make_state(physical_main=False)
        self.state.physical_displays = []
        self.state.online_displays = [d for d in self.state.online_displays if d.is_virtual or d.is_sidecar]

        def disconnect(_):
            self.state.sidecar_connected = self.state.sidecar_display_online = False
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.disconnect()
        self.assertFalse(self.engine.actual.sidecar_connected)
        self.assertTrue(self.engine.actual.virtual_display_connected)
        self.bd.disconnect_virtual_display.assert_not_called()

    def test_unobserved_virtual_retirement_does_not_disconnect_ipad(self):
        self.bd.disconnect_virtual_display.side_effect = None
        self.bd.disconnect_virtual_display.return_value = True
        self.disconnect()
        self.assertTrue(self.engine.actual.virtual_display_connected)
        self.assertTrue(self.engine.actual.sidecar_connected)
        self.bd.disconnect_sidecar.assert_not_called()
        self.assertIsNotNone(self.engine.runtime.last_error)

    def test_missing_physical_rolls_back_to_the_same_ipad_only_once(self):
        def disconnect(_):
            self.state = self.make_state(connected=False, physical_main=False)
            self.state.physical_displays = []
            self.state.online_displays = [self.state.main_display]
            return True

        def reconnect(specifier):
            self.assertEqual(specifier, "TARGET")
            self.state = self.make_state()
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.bd.connect_sidecar.side_effect = reconnect
        self.disconnect()
        self.assertTrue(self.engine.actual.sidecar_connected)
        self.assertTrue(self.engine._disconnect_failed)
        self.assertEqual(self.engine.runtime.last_error,
                         "Could not safely disconnect iPad; iPad was reconnected to restore the display.")
        previous_calls = list(self.bd.mock_calls)
        for _ in range(3):
            self.engine.evaluate(trigger="watchdog", async_transition=False)
        self.assertEqual(self.bd.mock_calls, previous_calls)
        self.bd.connect_sidecar.assert_called_once_with("TARGET")
        self.bd.disconnect_sidecar.assert_called_once_with("TARGET")
        self.assertNotEqual(self.engine._last_snapshot.status_details["actual_role_satisfied"], "✓ Satisfied")

    def test_changed_target_blocks_missing_physical_rollback(self):
        def disconnect(_):
            self.state = self.make_state(connected=False, physical_main=False)
            self.state.physical_displays = []
            self.state.online_displays = [self.state.main_display]
            self.cfg.ipad = IpadConfig(name="Other iPad", sidecar_uuid="OTHER")
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.disconnect()
        self.bd.connect_sidecar.assert_not_called()
        self.assertIsNotNone(self.engine.runtime.last_error)

    def test_cancellation_after_physical_loss_blocks_rollback(self):
        def disconnect(_):
            self.state = self.make_state(connected=False, physical_main=False)
            self.state.physical_displays = []
            self.state.online_displays = [self.state.main_display]
            return True

        def stop_mirroring(_):
            self.engine.request_manual_mode(self.cfg)
            return True

        self.bd.disconnect_sidecar.side_effect = disconnect
        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.disconnect()
        self.bd.connect_sidecar.assert_not_called()
        self.assertIsNone(self.engine.runtime.user_override)

    def test_delayed_saved_layout_is_observed_after_disconnect_settle(self):
        self.wire_recovery()

        def disconnect(_):
            self.state = self.make_state(connected=False)
            return True

        restored = False

        def settle(seconds):
            nonlocal restored
            self.assertEqual(seconds, 0.5)
            if not restored and not self.state.sidecar_connected:
                self.state = self.make_state(connected=False, physical_main=False, mirror_source=4)
                restored = True

        self.bd.disconnect_sidecar.side_effect = disconnect
        with patch.object(self.engine, "_wait", side_effect=settle):
            self.disconnect()
        self.assertTrue(restored)
        self.assert_recovered()
        self.bd.stop_mirroring.assert_called_once_with(self.physical_uuid)
        self.assertEqual(self.bd.set_main_display.call_args_list, [call(self.physical_uuid), call(self.physical_uuid)])

    def test_manual_cancellation_after_unmirror_stops_main_and_disconnect_commands(self):
        self.state = self.make_state(physical_main=False, mirror_source=4)

        def stop_mirroring(_):
            self.engine.request_manual_mode(self.cfg)
            return True

        self.bd.stop_mirroring.side_effect = stop_mirroring
        self.disconnect()
        self.assertEqual(self.engine.runtime.mode, OperationMode.MANUAL_ONLY)
        self.assertIsNone(self.engine.runtime.user_override)
        self.assertTrue(self.engine.actual.sidecar_connected)
        self.bd.stop_mirroring.assert_called_once_with(self.physical_uuid)
        self.bd.set_main_display.assert_not_called()
        self.bd.disconnect_sidecar.assert_not_called()

    def test_ui_mode_command_releases_failed_disconnect_and_resumes_transition(self):
        daemon_cls = runpy.run_path(str(Path(__file__).resolve().parents[1] / "bin/sidecarswitchd"))["SidecarSwitchDaemon"]
        daemon = daemon_cls.__new__(daemon_cls)
        daemon.config, daemon.bd_cli = self.cfg, self.bd
        daemon.detector, daemon.engine = self.detector, self.engine
        self.state = self.make_state(connected=False, physical_main=False, mirror_source=4)
        self.engine._disconnect_failed = True
        self.engine.runtime.last_error = "Could not restore the fallback display after disconnecting iPad."
        save = MagicMock()
        request = {"command": "set_mode", "params": {"mode": "automatic"},
                   "expected_revision": self.cfg.revision}
        with patch.dict(daemon_cls.handle_client_cmd.__globals__, save_config=save), \
             patch.object(self.engine, "_trigger_transition") as transition:
            response = json.loads(daemon.handle_client_cmd(json.dumps(request)))
        self.assertTrue(response["ok"])
        self.assertEqual(self.engine.runtime.mode, OperationMode.AUTOMATIC)
        self.assertFalse(self.engine._disconnect_failed)
        self.assertEqual(self.engine.desired.target_display_role, DisplayRole.PHYSICAL)
        transition.assert_called_once_with()
        save.assert_called_once_with(daemon.config)


if __name__ == "__main__":
    unittest.main()
