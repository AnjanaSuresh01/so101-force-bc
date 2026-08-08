"""Feetech STS3215 serial protocol -- the bus both SO-101 arms sit on.

NOT EXECUTED AGAINST HARDWARE. There is no SO-101 on this machine, and no
serial port. What is tested is everything that does not need one: packet
framing, checksums, the 0-4095 position encoding, and the STS series' signed
load format, all exercised through `FakePort` in tests/test_feetech.py. What is
not tested is whether a real servo answers. Treat the timing constants and the
error recovery as unverified.

The control-table addresses are the STS3215's. The two that matter here:

    Present_Position (0x38, 2 bytes)  the leader's joint angle, which is the
                                      teleoperation command
    Present_Load     (0x3C, 2 bytes)  an 11-bit magnitude with bit 10 as sign,
                                      as a fraction of stall torque -- the only
                                      force-related quantity an SO-101 exposes,
                                      and the input to griff.sensing

Reading the whole arm is a SYNC READ so that all six joints come from one bus
transaction. Polling them one at a time at 30 Hz does not fit in the budget at
1 Mbaud once you count turnaround, and staggered reads would put a per-joint
skew into a dataset whose entire point is synchronised channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

BROADCAST_ID = 0xFE
HEADER = b"\xff\xff"

# Instructions
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83

# Control table (STS3215)
ADDR_MODE = 0x21
ADDR_TORQUE_ENABLE = 0x28
ADDR_GOAL_POSITION = 0x2A
ADDR_PRESENT_POSITION = 0x38
ADDR_PRESENT_SPEED = 0x3A
ADDR_PRESENT_LOAD = 0x3C
ADDR_PRESENT_VOLTAGE = 0x3E
ADDR_PRESENT_TEMPERATURE = 0x3F

#: One full turn of the output shaft, in encoder counts.
COUNTS_PER_TURN = 4096
#: Servo IDs on an SO-101, shoulder to gripper.
DEFAULT_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


class SerialPort(Protocol):
    """The slice of pyserial this driver uses."""

    def write(self, data: bytes) -> int | None: ...
    def read(self, size: int) -> bytes: ...
    def reset_input_buffer(self) -> None: ...


class FeetechError(RuntimeError):
    pass


def checksum(payload: bytes) -> int:
    """Feetech checksum: bitwise NOT of the sum of everything after the header."""
    return (~sum(payload)) & 0xFF


def build_packet(servo_id: int, instruction: int, parameters: bytes = b"") -> bytes:
    body = bytes([servo_id, len(parameters) + 2, instruction]) + parameters
    return HEADER + body + bytes([checksum(body)])


def build_sync_read(ids: tuple[int, ...], address: int, length: int) -> bytes:
    return build_packet(BROADCAST_ID, INST_SYNC_READ, bytes([address, length, *ids]))


def build_sync_write(ids: tuple[int, ...], address: int, values: list[bytes]) -> bytes:
    if len(ids) != len(values):
        raise ValueError("ids and values must be the same length")
    width = len(values[0])
    if any(len(v) != width for v in values):
        raise ValueError("all sync-write values must be the same width")
    parameters = bytearray([address, width])
    for servo_id, value in zip(ids, values, strict=True):
        parameters.append(servo_id)
        parameters.extend(value)
    return build_packet(BROADCAST_ID, INST_SYNC_WRITE, bytes(parameters))


def encode_u16(value: int) -> bytes:
    """Little-endian, as the STS series expects."""
    return bytes([value & 0xFF, (value >> 8) & 0xFF])


def decode_u16(data: bytes) -> int:
    return data[0] | (data[1] << 8)


def decode_load(raw: int) -> float:
    """STS present-load to a signed fraction of stall torque.

    The field is 11 bits: 0-1023 of magnitude in bits 0-9, direction in bit 10.
    Reading it as a plain unsigned 16-bit value -- which is the obvious mistake,
    because every other field on this bus is one -- gives a load that jumps to
    +1024 the instant the joint pushes the other way.
    """
    magnitude = raw & 0x3FF
    sign = -1.0 if raw & 0x400 else 1.0
    return sign * magnitude / 1023.0


def encode_load(value: float) -> int:
    """Inverse of `decode_load`, for tests and for the fake bus."""
    magnitude = min(1023, int(round(abs(value) * 1023)))
    return magnitude | (0x400 if value < 0 else 0)


def counts_to_radians(counts: int) -> float:
    return (counts - COUNTS_PER_TURN / 2) * (2 * np.pi / COUNTS_PER_TURN)


def radians_to_counts(radians: float) -> int:
    counts = int(round(radians * (COUNTS_PER_TURN / (2 * np.pi)) + COUNTS_PER_TURN / 2))
    return int(np.clip(counts, 0, COUNTS_PER_TURN - 1))


@dataclass(frozen=True)
class StatusPacket:
    servo_id: int
    error: int
    parameters: bytes


def parse_status(data: bytes) -> StatusPacket:
    """Parse one status packet, raising on framing or checksum failure."""
    if len(data) < 6:
        raise FeetechError(f"status packet too short: {len(data)} bytes")
    if data[:2] != HEADER:
        raise FeetechError(f"bad header {data[:2]!r}")
    servo_id, length = data[2], data[3]
    end = 4 + length
    if len(data) < end:
        raise FeetechError(f"truncated status packet: want {end} bytes, have {len(data)}")
    body = data[2 : end - 1]
    if checksum(body) != data[end - 1]:
        raise FeetechError(f"checksum mismatch on servo {servo_id}")
    return StatusPacket(servo_id=servo_id, error=data[4], parameters=data[5 : end - 1])


class FeetechBus:
    """Half-duplex bus shared by all six servos of one arm."""

    def __init__(self, port: SerialPort, ids: tuple[int, ...] = DEFAULT_IDS) -> None:
        self.port = port
        self.ids = ids

    def _read_status(self, expected_parameters: int) -> StatusPacket:
        # 2 header + id + length + error + params + checksum
        raw = self.port.read(6 + expected_parameters)
        return parse_status(raw)

    def ping(self, servo_id: int) -> bool:
        self.port.reset_input_buffer()
        self.port.write(build_packet(servo_id, INST_PING))
        try:
            return self._read_status(0).servo_id == servo_id
        except FeetechError:
            return False

    def sync_read(self, address: int, length: int) -> list[int]:
        """One transaction, one value per servo, in `self.ids` order."""
        self.port.reset_input_buffer()
        self.port.write(build_sync_read(self.ids, address, length))
        values = []
        for servo_id in self.ids:
            status = self._read_status(length)
            if status.servo_id != servo_id:
                raise FeetechError(
                    f"sync read out of order: expected servo {servo_id}, got {status.servo_id}"
                )
            values.append(decode_u16(status.parameters) if length == 2 else status.parameters[0])
        return values

    def read_positions(self) -> np.ndarray:
        return np.array(
            [counts_to_radians(c) for c in self.sync_read(ADDR_PRESENT_POSITION, 2)]
        )

    def read_loads(self) -> np.ndarray:
        """Signed load per joint, as a fraction of stall torque."""
        return np.array([decode_load(v) for v in self.sync_read(ADDR_PRESENT_LOAD, 2)])

    def write_positions(self, radians: np.ndarray) -> None:
        if len(radians) != len(self.ids):
            raise ValueError(f"expected {len(self.ids)} joint targets, got {len(radians)}")
        values = [encode_u16(radians_to_counts(float(r))) for r in radians]
        self.port.write(build_sync_write(self.ids, ADDR_GOAL_POSITION, values))

    def set_torque(self, enabled: bool) -> None:
        """Torque off is how the leader arm is made back-drivable."""
        values = [bytes([1 if enabled else 0])] * len(self.ids)
        self.port.write(build_sync_write(self.ids, ADDR_TORQUE_ENABLE, values))
