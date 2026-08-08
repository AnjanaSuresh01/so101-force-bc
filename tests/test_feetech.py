"""Feetech STS3215 protocol.

There is no servo on this machine, so what is testable is the wire format: how
packets are framed, how the checksum is computed, and how the two encodings that
are easy to get wrong -- position counts and the signed load field -- round
trip. `FakePort` answers like a bus would, so `FeetechBus` is exercised end to
end without hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from griff.teleop.feetech import (
    ADDR_GOAL_POSITION,
    ADDR_PRESENT_LOAD,
    ADDR_PRESENT_POSITION,
    ADDR_TORQUE_ENABLE,
    BROADCAST_ID,
    COUNTS_PER_TURN,
    DEFAULT_IDS,
    HEADER,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    FeetechBus,
    FeetechError,
    build_packet,
    build_sync_read,
    checksum,
    counts_to_radians,
    decode_load,
    encode_load,
    encode_u16,
    parse_status,
    radians_to_counts,
)


class FakePort:
    """A bus that answers sync reads with values it was primed with."""

    def __init__(self, values: dict[int, int] | None = None, width: int = 2) -> None:
        self.values = values or {}
        self.width = width
        self.written: list[bytes] = []
        self._pending = b""

    def reset_input_buffer(self) -> None:
        self._pending = b""

    def write(self, data: bytes) -> int:
        self.written.append(data)
        if len(data) > 4 and data[4] == INST_SYNC_READ:
            responses = []
            for servo_id in DEFAULT_IDS:
                payload = encode_u16(self.values.get(servo_id, 0))[: self.width]
                body = bytes([servo_id, len(payload) + 2, 0x00]) + payload
                responses.append(HEADER + body + bytes([checksum(body)]))
            self._pending = b"".join(responses)
        return len(data)

    def read(self, size: int) -> bytes:
        chunk, self._pending = self._pending[:size], self._pending[size:]
        return chunk


def test_checksum_matches_the_protocol_definition() -> None:
    body = bytes([0x01, 0x02, 0x01])
    assert checksum(body) == (~sum(body)) & 0xFF
    assert build_packet(1, 0x01)[-1] == checksum(body)


def test_packets_are_framed_correctly() -> None:
    packet = build_packet(3, 0x02, bytes([ADDR_PRESENT_POSITION, 2]))
    assert packet[:2] == HEADER
    assert packet[2] == 3
    assert packet[3] == 4  # params + 2
    assert packet[4] == 0x02
    assert packet[-1] == checksum(packet[2:-1])


def test_sync_read_addresses_every_configured_servo() -> None:
    packet = build_sync_read(DEFAULT_IDS, ADDR_PRESENT_POSITION, 2)
    assert packet[2] == BROADCAST_ID
    assert packet[4] == INST_SYNC_READ
    assert list(packet[5:7]) == [ADDR_PRESENT_POSITION, 2]
    assert list(packet[7 : 7 + len(DEFAULT_IDS)]) == list(DEFAULT_IDS)


def test_parse_status_rejects_a_bad_checksum() -> None:
    body = bytes([1, 3, 0, 42])
    corrupted = HEADER + body + bytes([(checksum(body) + 1) & 0xFF])
    with pytest.raises(FeetechError, match="checksum"):
        parse_status(corrupted)


def test_parse_status_rejects_a_bad_header() -> None:
    with pytest.raises(FeetechError, match="header"):
        parse_status(b"\x00\x00\x01\x02\x00\xfc")


def test_parse_status_rejects_a_truncated_packet() -> None:
    with pytest.raises(FeetechError, match="short"):
        parse_status(b"\xff\xff\x01")


@pytest.mark.parametrize("counts", [0, 1, 2048, 4095])
def test_position_encoding_round_trips(counts: int) -> None:
    assert radians_to_counts(counts_to_radians(counts)) == counts


def test_position_encoding_is_centred_at_half_a_turn() -> None:
    assert counts_to_radians(COUNTS_PER_TURN // 2) == pytest.approx(0.0)
    assert radians_to_counts(0.0) == COUNTS_PER_TURN // 2


def test_position_encoding_clamps_rather_than_wrapping() -> None:
    """A wrap here commands the far side of the circle. Clamp instead."""
    assert radians_to_counts(10.0) == COUNTS_PER_TURN - 1
    assert radians_to_counts(-10.0) == 0


@pytest.mark.parametrize("value", [0.0, 0.5, -0.5, 1.0, -1.0, 0.123])
def test_load_encoding_round_trips_with_its_sign(value: float) -> None:
    assert decode_load(encode_load(value)) == pytest.approx(value, abs=1e-3)


def test_load_sign_bit_is_not_read_as_magnitude() -> None:
    """The bug this encoding invites: reading the field as a plain uint16.

    Bit 10 is direction. Treated as magnitude it makes a joint pushing one way
    report a load 1024 counts higher than the same effort the other way.
    """
    negative_half = encode_load(-0.5)
    assert negative_half & 0x400
    assert decode_load(negative_half) < 0
    assert (negative_half & 0x3FF) == encode_load(0.5)


def test_bus_reads_positions_for_every_servo() -> None:
    port = FakePort({servo: 2048 + 100 * servo for servo in DEFAULT_IDS})
    bus = FeetechBus(port)
    positions = bus.read_positions()
    assert positions.shape == (len(DEFAULT_IDS),)
    assert np.allclose(positions, [counts_to_radians(2048 + 100 * s) for s in DEFAULT_IDS])


def test_bus_reads_signed_loads() -> None:
    port = FakePort({1: encode_load(0.25), 2: encode_load(-0.75)})
    loads = FeetechBus(port).read_loads()
    assert loads[0] == pytest.approx(0.25, abs=1e-3)
    assert loads[1] == pytest.approx(-0.75, abs=1e-3)


def test_bus_write_positions_is_one_sync_write() -> None:
    port = FakePort()
    FeetechBus(port).write_positions(np.zeros(6))
    assert len(port.written) == 1
    packet = port.written[0]
    assert packet[2] == BROADCAST_ID
    assert packet[4] == INST_SYNC_WRITE
    assert packet[5] == ADDR_GOAL_POSITION


def test_bus_write_positions_checks_the_joint_count() -> None:
    with pytest.raises(ValueError, match="6 joint targets"):
        FeetechBus(FakePort()).write_positions(np.zeros(3))


def test_torque_off_is_a_single_broadcast() -> None:
    """Making the leader back-drivable is one write, not six."""
    port = FakePort()
    FeetechBus(port).set_torque(False)
    assert len(port.written) == 1
    assert port.written[0][5] == ADDR_TORQUE_ENABLE


def test_present_load_address_is_not_present_position() -> None:
    """Guards a transposition that would silently read angles as forces."""
    assert ADDR_PRESENT_POSITION != ADDR_PRESENT_LOAD
    assert ADDR_PRESENT_LOAD == 0x3C
