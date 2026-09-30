import math
import time

import numpy as np


class SquatStateMachine:
    """Máquina de estados de sentadilla dinámica conducida por plantilla.

    Fases: ESPERANDO -> (calibración) -> DE_PIE -> BAJANDO -> SQUAT_PROFUNDO -> DE_PIE
    """

    def __init__(self, template: dict):
        self.template = template
        self.fase = "ESPERANDO"
        self.y_inicial = None
        self.repeticiones = 0

    @property
    def calibrado(self) -> bool:
        return self.y_inicial is not None

    def calibrate(self, y: float):
        self.y_inicial = y
        self.fase = "DE_PIE"
        self.repeticiones = 0

    def update(self, y_actual: float, postura_correcta: bool) -> dict:
        """Procesa un frame y devuelve el estado actual."""
        if not self.calibrado:
            return {
                "fase": self.fase,
                "desplazamiento_y": 0.0,
                "repeticiones": 0,
                "rep_valid": False,
                "rep_rejected": False,
                "rep_rejection_reason": "",
            }

        desplazamiento_y = y_actual - self.y_inicial
        rep_valid = False
        rep_rejected = False
        rep_rejection_reason = ""
        tpl = self.template

        if self.fase == "DE_PIE":
            if desplazamiento_y > tpl["descenso_inicio_m"]:
                self.fase = "BAJANDO"
        elif self.fase == "BAJANDO":
            if desplazamiento_y >= tpl["profundidad_objetivo_m"]:
                self.fase = "SQUAT_PROFUNDO"
        elif self.fase == "SQUAT_PROFUNDO":
            if desplazamiento_y < tpl["subida_completa_m"]:
                if postura_correcta:
                    self.repeticiones += 1
                    rep_valid = True
                else:
                    rep_rejected = True
                    rep_rejection_reason = "postura"
                self.fase = "DE_PIE"

        return {
            "fase": self.fase,
            "desplazamiento_y": round(desplazamiento_y, 3),
            "repeticiones": self.repeticiones,
            "rep_valid": rep_valid,
            "rep_rejected": rep_rejected,
            "rep_rejection_reason": rep_rejection_reason,
        }


class IsometricHoldStateMachine:
    """Cronómetro de tiempo válido para ejercicios isométricos.

    El tiempo total comienza al llamar :meth:`start`. El tiempo válido solo
    avanza durante intervalos en los que ``postura_correcta`` es verdadera.
    Esto permite distinguir duración nominal de la sesión y tiempo realmente
    ejecutado dentro de los criterios posturales definidos por la plantilla.
    """

    def __init__(self, target_s: float, clock=None):
        self.target_s = float(target_s)
        self.clock = clock or time.monotonic
        self.fase = "ESPERANDO"
        self.started_at = None
        self.last_ts = None
        self.valid_time_s = 0.0
        self.correction_count = 0
        self._was_valid = None

    @property
    def started(self) -> bool:
        return self.started_at is not None

    def start(self, now=None):
        now = self.clock() if now is None else float(now)
        self.started_at = now
        self.last_ts = now
        self.valid_time_s = 0.0
        self.correction_count = 0
        self._was_valid = None
        self.fase = "MANTENER"

    def update(self, postura_correcta: bool, now=None) -> dict:
        if not self.started:
            return self.snapshot(postura_correcta=False, now=now)

        now = self.clock() if now is None else float(now)
        dt = max(0.0, now - self.last_ts)
        self.last_ts = now

        if postura_correcta:
            self.valid_time_s += dt
        elif self._was_valid is True:
            self.correction_count += 1

        self._was_valid = postura_correcta
        completed = self.valid_time_s >= self.target_s
        self.fase = "COMPLETADO" if completed else ("MANTENER" if postura_correcta else "CORREGIR")
        return self.snapshot(postura_correcta=postura_correcta, now=now)

    def snapshot(self, postura_correcta: bool, now=None) -> dict:
        if self.started_at is None:
            total = 0.0
        else:
            now = self.clock() if now is None else float(now)
            total = max(0.0, now - self.started_at)
        compliance = (self.valid_time_s / total) if total > 0 else 0.0
        return {
            "fase": self.fase,
            "tiempo_total_s": round(total, 3),
            "tiempo_valido_s": round(self.valid_time_s, 3),
            "objetivo_tiempo_s": self.target_s,
            "cumplimiento_postural": round(min(1.0, compliance), 4),
            "correcciones": self.correction_count,
            "postura_correcta": postura_correcta,
            "completed": self.valid_time_s >= self.target_s,
        }


class IsometricSquatStateMachine(IsometricHoldStateMachine):
    """Sentadilla isométrica basada en el descenso de los hombros.

    Reutiliza la calibración de la sentadilla dinámica: ``y_inicial`` se toma
    de pie y el mantenimiento es válido cuando el descenso está dentro de la
    banda configurada y los hombros permanecen nivelados.
    """

    def __init__(self, template: dict, clock=None):
        super().__init__(template["duracion_objetivo_s"], clock=clock)
        self.template = template
        self.y_inicial = None

    @property
    def calibrado(self) -> bool:
        return self.y_inicial is not None

    def calibrate(self, y: float, now=None):
        self.y_inicial = float(y)
        self.start(now=now)

    def update_position(self, y_actual: float, hombros_nivelados: bool, now=None) -> dict:
        if not self.calibrado:
            state = self.snapshot(False, now=now)
            state["desplazamiento_y"] = 0.0
            state["en_banda"] = False
            return state

        desplazamiento = float(y_actual) - self.y_inicial
        en_banda = (
            self.template["descenso_min_m"]
            <= desplazamiento
            <= self.template["descenso_max_m"]
        )
        state = super().update(en_banda and hombros_nivelados, now=now)
        state["desplazamiento_y"] = round(desplazamiento, 3)
        state["en_banda"] = en_banda
        return state


class PlankStateMachine(IsometricHoldStateMachine):
    """Plancha isométrica evaluada por alineación hombro-cadera-tobillo."""

    def __init__(self, template: dict, clock=None):
        super().__init__(template["duracion_objetivo_s"], clock=clock)
        self.template = template

    def start_hold(self, now=None):
        self.start(now=now)

    def update_points(self, shoulder, hip, ankle, now=None) -> dict:
        angle = joint_angle(shoulder, hip, ankle)
        error = abs(180.0 - angle)
        tol = float(self.template["tolerancia_alineacion_deg"])
        state = super().update(error <= tol, now=now)
        state["angulo_corporal_deg"] = round(angle, 2)
        state["error_postural_deg"] = round(error, 2)
        state["postura_error_normalizado"] = round(min(1.0, error / max(tol, 1e-6)), 4)
        return state


def shoulder_angle(p_izq, p_der) -> float:
    """Ángulo de nivelación de hombros en grados (0 = nivelados)."""
    vector = p_der - p_izq
    return math.degrees(math.atan2(vector[1], vector[0]))


def joint_angle(a, b, c) -> float:
    """Ángulo ABC en grados, estable para puntos 2D o 3D."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
