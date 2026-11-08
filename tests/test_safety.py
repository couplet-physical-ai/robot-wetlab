from wetlab.hw.interfaces import ArmState, DoorState
from wetlab.safety import Interlock
from wetlab.spec import Safety

OK_DOOR, ARM = DoorState(True, True), ArmState([0.0] * 6, 110.0)


def il(**kw):
    return Interlock(Safety(**kw))


def test_can_start_needs_door_closed_and_locked():
    i = il()
    i.evaluate(DoorState(True, False), ARM, 0.0, False, active=False)
    assert not i.can_start() and i.current == ["door_unlocked"] and not i.tripped   # idle faults don't latch
    i.evaluate(DoorState(False, False, 30.0), ARM, 0.0, False, active=False)
    assert i.current == ["door_open"]
    i.evaluate(OK_DOOR, ARM, 0.0, False, active=False)
    assert i.can_start()
    assert il(require_door_closed_and_locked=False).faults(DoorState(False, False), ARM, 0.0, False) == []


def test_trip_latches_first_reason_and_reset_needs_clear_conditions():
    i = il()
    assert i.evaluate(DoorState(False, False), ARM, 0.0, False, active=True) == "door_open"
    assert i.tripped and i.reason == "door_open"
    assert i.evaluate(DoorState(False, False), ArmState([0] * 6, 110, "protective_stop"), 0.0, False, active=True) is None
    assert i.reason == "door_open"          # first reason wins
    i.evaluate(OK_DOOR, ARM, 0.0, False, active=False)
    assert i.tripped and not i.can_start()  # conditions clear but still latched
    assert i.reset(DoorState(True, False), ARM, 0.0, False) == ["door_unlocked"] and i.tripped
    assert i.reset(OK_DOOR, ARM, 0.0, False) == [] and not i.tripped and i.can_start()


def test_reasons():
    i = il(stop_on_camera_loss_s=2.0)
    assert i.faults(OK_DOOR, ARM, 2.5, False) == ["camera_lost"]
    assert i.faults(OK_DOOR, ArmState([0] * 6, 110, "safeguard_stop"), 0.0, False) == ["arm_safety:safeguard_stop"]
    assert i.faults(OK_DOOR, ArmState([0] * 6, 110, "estop"), 0.0, False) == ["estop"]
    assert i.evaluate(OK_DOOR, ARM, 0.0, True, active=False) == "estop"   # E-stop latches even when idle
    assert il().trip("slot_ended") == "slot_ended"
    j = il()
    j.trip("entry_requested")
    assert j.trip("door_open") is None and j.reason == "entry_requested"
