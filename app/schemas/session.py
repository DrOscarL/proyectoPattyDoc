from pydantic import BaseModel, Field


class Plan(BaseModel):
    calentamiento: str
    entrenamiento: str
    enfriamiento: str


class SessionStartRequest(BaseModel):
    plan: Plan


class SessionStartResponse(BaseModel):
    session_id: str
    plan: Plan
    status: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    plan: Plan
    current_exercise: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


class ObservationRequest(BaseModel):
    exercise: str
    frame_ts: float = Field(default=0.0, ge=0)
    fase: str = Field(default="")
    desplazamiento_y: float = Field(default=0.0)
    postura_correcta: bool = True
    hombros_visibles: bool = False
    repeticiones: int = Field(default=0, ge=0)
    rep_valid: bool = False
    rep_rejected: bool = False
    rep_rejection_reason: str = ""
    tiempo_total_s: float = Field(default=0.0, ge=0)
    tiempo_valido_s: float = Field(default=0.0, ge=0)
    objetivo_tiempo_s: float = Field(default=0.0, ge=0)
    cumplimiento_postural: float = Field(default=0.0, ge=0, le=1)
    correcciones: int = Field(default=0, ge=0)
    angulo_corporal_deg: float = Field(default=0.0, ge=0, le=180)
    error_postural_deg: float = Field(default=0.0, ge=0, le=180)


class CorrectionResponse(BaseModel):
    level: str
    message_es: str
    siguiente_paso: str
    evento_voz: str | None = None
    mensaje_voz: str | None = None
    id_evento_voz: str | None = None
