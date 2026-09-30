# Ejercicios isométricos: sentadilla y plancha

Esta extensión incorpora dos mecánicas isométricas al gemelo digital de PattyDoc sin eliminar la sentadilla dinámica existente.

## 1. Sentadilla isométrica

La sentadilla isométrica reutiliza las AprilTags existentes:

- ID 0: hombro izquierdo.
- ID 1: hombro derecho.

### Flujo

1. El usuario se coloca de pie frente a la cámara.
2. Pulsa **Calibrar** para registrar la altura inicial de los hombros.
3. Desciende hasta la banda configurada en `data/exercise_templates.json`.
4. El cronómetro de **tiempo válido** avanza únicamente si:
   - el descenso está dentro de `[descenso_min_m, descenso_max_m]`;
   - los hombros permanecen dentro de `tolerancia_hombros_deg`.
5. Si la postura sale de la banda válida, el tiempo total continúa pero el tiempo válido se pausa.

### Métricas

- `tiempo_total_s`: duración transcurrida desde la calibración.
- `tiempo_valido_s`: tiempo acumulado con postura aceptada.
- `cumplimiento_postural`: `tiempo_valido_s / tiempo_total_s`.
- `correcciones`: número de transiciones desde postura válida hacia postura inválida.
- `desplazamiento_y`: descenso de los hombros respecto de la calibración.

## 2. Plancha

La plancha usa tres AprilTags laterales adicionales y requiere vista lateral de la persona:

- ID 2: hombro lateral.
- ID 3: cadera lateral.
- ID 4: tobillo lateral.

El sistema estima el ángulo hombro-cadera-tobillo. Una alineación ideal corresponde aproximadamente a 180 grados. El tiempo se considera válido cuando el error respecto de 180 grados permanece dentro de `tolerancia_alineacion_deg`.

### Métricas

- `angulo_corporal_deg`.
- `error_postural_deg = |180 - angulo_corporal_deg|`.
- `tiempo_total_s`.
- `tiempo_valido_s`.
- `cumplimiento_postural`.
- `correcciones`.

## 3. Configuración inicial

Los umbrales incluidos son valores de ingeniería para validación tecnológica y deben calibrarse experimentalmente antes de interpretarlos como criterios biomecánicos o clínicos.

Configuración inicial:

```json
{
  "Sentadilla isométrica": {
    "duracion_objetivo_s": 30,
    "descenso_min_m": 0.28,
    "descenso_max_m": 0.42,
    "tolerancia_hombros_deg": 10
  },
  "Plancha": {
    "duracion_objetivo_s": 30,
    "tolerancia_alineacion_deg": 15
  }
}
```

## 4. Uso experimental

Las observaciones se almacenan en SQLite junto con las métricas isométricas. Esto permite exportar posteriormente series temporales para comparar el sistema contra anotación manual o vídeo de referencia y calcular, por ejemplo:

- error absoluto del tiempo válido;
- precisión/recall/F1 de postura válida vs. inválida;
- latencia de corrección;
- Postural Compliance Index basado en la proporción de tiempo válido y el error postural.

## 5. Compatibilidad

La mecánica se selecciona mediante la propiedad `mecanica` de cada plantilla:

- `squat`: sentadilla dinámica existente;
- `isometric_squat`: sentadilla isométrica;
- `plank`: plancha isométrica.

El `CameraController` actúa como dispatcher y mantiene el comportamiento previo para `squat`.
