from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)  # разрешаем запросы с фронта

# Загружаем модель один раз
model = YOLO("runs/train/tree_detection_finetune/weights/best.pt")

def filter_small_boxes(boxes, image_shape, min_area_percent=0.001, min_side_percent=0.01):
    """Фильтрует слишком маленькие боксы по процентам от площади изображения"""
    if len(boxes) == 0:
        return boxes
    
    img_height, img_width = image_shape[:2]
    total_image_area = img_width * img_height
    
    min_area = total_image_area * min_area_percent
    min_side = min(img_width, img_height) * min_side_percent
    
    filtered_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if (area >= min_area and width >= min_side and height >= min_side):
            filtered_boxes.append(box)
    
    return np.array(filtered_boxes)

def advanced_merge_boxes(boxes, size_weight=0.7, conf_weight=0.3, distance_threshold=100):
    """Объединяет боксы через кластеризацию и выбирает наиболее подходящий"""
    if len(boxes) == 0:
        return []
    
    centers = []
    sizes = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        centers.append([center_x, center_y])
        sizes.append(width * height)
    
    centers = np.array(centers)
    sizes = np.array(sizes)
    
    n_boxes = len(boxes)
    visited = [False] * n_boxes
    clusters = []
    
    for i in range(n_boxes):
        if visited[i]:
            continue
            
        cluster = [i]
        visited[i] = True
        
        for j in range(i+1, n_boxes):
            if visited[j]:
                continue
                
            distance = np.sqrt(((centers[i] - centers[j]) ** 2).sum())
            
            if distance < distance_threshold:
                cluster.append(j)
                visited[j] = True
        
        clusters.append(cluster)
    
    merged_boxes = []
    
    for cluster in clusters:
        cluster_boxes = boxes[cluster]
        cluster_sizes = sizes[cluster]
        
        if len(cluster_boxes) == 1:
            merged_boxes.append(cluster_boxes[0])
        else:
            best_score = -1
            best_box = None
            
            for i, box in enumerate(cluster_boxes):
                x1, y1, x2, y2, conf, cls = box
                size_score = cluster_sizes[i] / max(sizes) if max(sizes) > 0 else 0
                conf_score = conf
                
                total_score = size_weight * size_score + conf_weight * conf_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_box = box
            
            merged_boxes.append(best_box)
    
    return np.array(merged_boxes)

def visualize_boxes(image, boxes, class_names=None):
    img_display = image.copy()
    
    if class_names is None:
        class_names = {0: "tree", 1: "bush"}
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        class_name = class_names.get(class_id, f"class_{class_id}")
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
        cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
        label = f"{class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img_display, 
                      (int(x1), int(y1) - label_size[1] - 10),
                      (int(x1) + label_size[0], int(y1)),
                      color, -1)
        cv2.putText(img_display, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    height, width = img_display.shape[:2]
    max_display_size = 800
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_display = cv2.resize(img_display, (new_width, new_height))
    
    return img_display


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    img_bytes = file.read()
    
    # читаем картинку из байтов
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Не удалось прочитать изображение"}), 400

    # Предсказание
    results = model.predict(img, conf=0.3)
    boxes = results[0].boxes.data.cpu().numpy()

    # тут твои функции фильтрации/объединения
    filtered_boxes = filter_small_boxes(boxes, img.shape, 
                                      min_area_percent=0.01, 
                                      min_side_percent=0.1)
    merged_boxes = advanced_merge_boxes(filtered_boxes, size_weight=0.8, conf_weight=0.2)
    class_names = {0: "tree", 1: "bush"}

    if len(merged_boxes) > 0:
        final_display = visualize_boxes(img, merged_boxes, class_names)
    else:
        final_display = img.copy()

    # Кодируем обратно в base64
    _, buffer = cv2.imencode(".jpg", final_display)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_img})

if __name__ == "__main__":
    app.run(debug=True)
