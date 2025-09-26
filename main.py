import cv2
from PlantDetection import TreeDetection


def main():
    img = cv2.imread("ing_test/0HulTZgTU3U (1).jpg")
    detection = TreeDetection(weights_path="runs/train/tree_detection_finetune/weights/best.pt")
    detection.find_bbox(img)

if __name__ == "__main__":
    main()