import math

import numpy as np
import pytest

from wetlab.spec import DEFAULT_LAB, load_lab
from wetlab.twin.kinematics import Q_MAX, ArmModel, fk_base, in_limits


@pytest.fixture(scope="module")
def arm():
    return ArmModel(load_lab(DEFAULT_LAB))


def test_zero_pose_matches_ur_dh():
    # UR DH at q=0: arm stretched along base -x; flange at (a2+a3, -(d4+d6), d1-d5)
    np.testing.assert_allclose(fk_base(np.zeros(6))[-1][:3, 3], [-0.8172, -0.2329, 0.0628], atol=1e-9)


def test_frames_and_tcp_offset(arm):
    fs = arm.frames(arm.home_q)
    assert len(fs) == 8
    np.testing.assert_allclose(fs[0], arm.T_room_base)
    np.testing.assert_allclose(fs[7][:3, 3], fs[6][:3, 3] + 0.2 * fs[6][:3, 2], atol=1e-12)


def test_ik_round_trip_random_reachable(arm):
    rng = np.random.default_rng(42)
    for _ in range(300):
        q = rng.uniform(-math.pi, math.pi, 6)
        T = arm.tcp(q)
        q_any = arm.ik(T, rng.uniform(-math.pi, math.pi, 6))   # any seed: some valid solution
        assert q_any is not None and in_limits(q_any)
        np.testing.assert_allclose(arm.tcp(q_any), T, atol=1e-6)
        np.testing.assert_allclose(arm.ik(T, q), q, atol=1e-6)  # seeded at the answer: picks it


def test_ik_wraps_to_seed_within_limits(arm):
    q = np.array([0.3, -1.2, 1.4, -1.6, -1.5, 0.2])
    seed = q + [-2 * math.pi, 0, 0, 2 * math.pi, 0, -2 * math.pi]   # all still inside +-360 deg
    np.testing.assert_allclose(arm.ik(arm.tcp(q), seed), seed, atol=1e-6)
    over = q + [2 * math.pi, 0, 0, 0, 0, 0]                         # q1 + 360 deg is past the limit
    assert in_limits(arm.ik(arm.tcp(q), over))
    assert Q_MAX[2] == pytest.approx(math.pi)   # elbow +-180 deg


def test_ik_unreachable(arm):
    T = arm.tcp(arm.home_q)
    T[:3, 3] += [3.0, 0, 0]
    assert arm.ik(T, arm.home_q) is None
