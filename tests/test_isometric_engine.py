import numpy as np

from app.vision.engine import IsometricSquatStateMachine, PlankStateMachine, joint_angle


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_joint_angle_straight_line():
    angle = joint_angle(np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    assert abs(angle - 180.0) < 1e-9


def test_isometric_squat_counts_only_valid_time():
    clock = FakeClock()
    template = {
        "duracion_objetivo_s": 3,
        "descenso_min_m": 0.28,
        "descenso_max_m": 0.42,
    }
    sm = IsometricSquatStateMachine(template, clock=clock)
    sm.calibrate(1.0)

    clock.advance(1.0)
    s1 = sm.update_position(1.35, True)
    assert s1["tiempo_valido_s"] == 1.0

    clock.advance(1.0)
    s2 = sm.update_position(1.10, True)
    assert s2["tiempo_valido_s"] == 1.0
    assert s2["postura_correcta"] is False

    clock.advance(2.0)
    s3 = sm.update_position(1.35, True)
    assert s3["tiempo_valido_s"] == 3.0
    assert s3["completed"] is True
    assert s3["cumplimiento_postural"] == 0.75


def test_plank_alignment_and_corrections():
    clock = FakeClock()
    template = {"duracion_objetivo_s": 2, "tolerancia_alineacion_deg": 15}
    sm = PlankStateMachine(template, clock=clock)
    sm.start_hold()

    shoulder = np.array([0.0, 0.0, 0.0])
    hip = np.array([1.0, 0.0, 0.0])
    ankle = np.array([2.0, 0.0, 0.0])

    clock.advance(1.0)
    good = sm.update_points(shoulder, hip, ankle)
    assert good["postura_correcta"] is True
    assert good["tiempo_valido_s"] == 1.0
    assert good["error_postural_deg"] == 0.0

    clock.advance(1.0)
    bad = sm.update_points(shoulder, hip, np.array([1.0, 1.0, 0.0]))
    assert bad["postura_correcta"] is False
    assert bad["tiempo_valido_s"] == 1.0
    assert bad["correcciones"] == 1

    clock.advance(1.0)
    done = sm.update_points(shoulder, hip, ankle)
    assert done["completed"] is True
    assert done["tiempo_valido_s"] == 2.0
