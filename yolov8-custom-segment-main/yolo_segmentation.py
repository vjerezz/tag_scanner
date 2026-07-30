#https://pysource.com/2023/02/21/yolo-v8-segmentation
from ultralytics import YOLO
import numpy as np


class YOLOSEG:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        
        segmentation_contours_idx = []
        
        # Verificamos que existan resultados y máscaras detectadas
        if len(result) > 0 and result.masks is not None:
            # CAMBIO AQUÍ: Usamos .xy en lugar de .segments
            for seg in result.masks.xy:
                # Nota: .xy ya viene en coordenadas de píxeles, por lo que 
                # ya no es estrictamente necesario multiplicar por width y height 
                # a menos que el formato requiera un ajuste específico.
                segment = np.array(seg, dtype=np.int32)
                segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores