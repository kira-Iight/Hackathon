import cv2
from TreeDetection import TreeDetector


def main():

    img_path = "ing_test/photo_5384555870446286053_y.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_path}")

    detection = TreeDetector(weights_path="runs/train/tree_detection_finetune/weights/best.pt")
    
    boxes, classes = detection.bboxes(img)  # или bboxes_search — как у вас
    detection.visualize_boxes(img, boxes, classes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Вырезаем
        cropped = img[y1:y2, x1:x2]

        # Получаем имя класса
        cls_name = classes.get(int(cls_id), 'unknown')

        # Показываем
        cv2.imshow('Cropped', cropped)
        cv2.waitKey(0)  # Ждём нажатия клавиши

if __name__ == "__main__":
    main()