import cv2
from PlantDetection import TreeDetection


def main():

    img_path = "ing_test/0HulTZgTU3U (1).jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить: {img_path}")

    detection = TreeDetection(weights_path="runs/train/tree_detection_finetune/weights/best.pt")
    
    boxes, classes = detection.bboxes(img)  # или bboxes_search — как у вас
    detection.visualize_boxes(img, boxes, classes)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Защита от некорректных bbox
        if x2 <= x1 or y2 <= y1:
            print(f"Пропущен bbox {i}: некорректные координаты")
            continue

        # Вырезаем
        cropped = img[y1:y2, x1:x2]

        # Получаем имя класса
        cls_name = classes.get(int(cls_id), 'unknown')

        # Показываем
        cv2.imshow('Cropped', cropped)
        cv2.waitKey(0)  # Ждём нажатия клавиши

if __name__ == "__main__":
    main()