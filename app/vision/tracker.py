import cv2
import numpy as np
from pupil_apriltags import Detector

ID_HOMBRO_IZQ = 0
ID_HOMBRO_DER = 1
# Etiquetas laterales opcionales para ejercicios isométricos de alineación.
ID_HOMBRO_LATERAL = 2
ID_CADERA_LATERAL = 3
ID_TOBILLO_LATERAL = 4

DEFAULT_CAMERA_PARAMS = (600.0, 600.0, 320.0, 240.0)
DEFAULT_TAG_SIZE = 0.05


class CameraTracker:
    """Abre la cámara (o un video), detecta AprilTags y estima sus posiciones 3D."""

    def __init__(
        self,
        source=0,
        tag_size=DEFAULT_TAG_SIZE,
        camera_params=DEFAULT_CAMERA_PARAMS,
    ):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la fuente de video: {source}. "
                "Verifica que la cámara esté disponible o que la ruta del archivo sea correcta."
            )
        self.detector = Detector(families="tag36h11")
        self.tag_size = tag_size
        self.camera_params = list(camera_params)

    def read(self):
        """Devuelve (frame, detecciones) o (None, None) si termina el video.

        detecciones = {
            "positions_3d": {tag_id: np.array(xyz)},
            "centers_2d": {tag_id: (x, y)},
            "corners": {tag_id: ndarray},
            "detected_ids": [tag_id, ...],
        }
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        positions_3d, centers_2d, corners = {}, {}, {}
        detected_ids = []
        for r in results:
            tag_id = int(r.tag_id)
            detected_ids.append(tag_id)
            centers_2d[tag_id] = (int(r.center[0]), int(r.center[1]))
            corners[tag_id] = r.corners
            t = r.pose_t
            if t is not None:
                positions_3d[tag_id] = np.array([t[0][0], t[1][0], t[2][0]])

        detecciones = {
            "positions_3d": positions_3d,
            "centers_2d": centers_2d,
            "corners": corners,
            "detected_ids": sorted(detected_ids),
        }
        return frame, detecciones

    @staticmethod
    def draw_tags(frame, corners):
        for tag_id, pts in corners.items():
            (ptA, ptB, ptC, ptD) = pts
            cv2.polylines(
                frame, [np.int32([ptA, ptB, ptC, ptD])], True, (255, 0, 0), 2
            )
            anchor = tuple(np.int32(ptA))
            cv2.putText(
                frame, f"ID {tag_id}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
