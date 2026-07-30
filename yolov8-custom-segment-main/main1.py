import cv2
from yolo_segmentation import YOLOSEG
import cvzone
import os

# Obtiene la ruta de la carpeta donde se encuentra este script actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Carga el modelo y las clases usando la ruta completa
ys = YOLOSEG(os.path.join(current_dir, "best.pt"))

my_file = open(os.path.join(current_dir, "coco1.txt"), "r")
data = my_file.read()
class_list = data.split("\n")

cap = cv2.VideoCapture(os.path.join(current_dir, 'etiquetas.mp4'))
count=0
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    overlay = frame.copy()
    alpha = 0.7

    bboxes, classes, segmentations, scores = ys.detect(frame)
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
        (x, y, x2, y2) = bbox
        c=class_list[class_id]
    
        if c == "etiqueta mala":
            color = (0, 0, 255)  # Rojo
        elif c == "etiqueta buena":
            color = (0, 255, 0)  # Verde
        

        # Dibuja el rectángulo en el color correspondiente
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
        cv2.polylines(frame, [seg], True, (0, 0, 255), 4)
        cv2.fillPoly(overlay, [seg], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 2, frame)
        cvzone.putTextRect(frame, f'{c}', (x, y), 1, 1)
    
        
    
    cv2.imshow("RGB",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
  