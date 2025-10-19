from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import warnings
import os
from torchvision import models

app = Flask(__name__)
CORS(app)  # разрешаем запросы с фронта

# Отключаем warnings
warnings.filterwarnings('ignore')

# Определяем устройство
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {DEVICE}")

# Параметры для классификации
IMG_SIZE = (224, 224)

# Загружаем модель детекции
print("🔍 Загрузка моделей...")
detection_model = YOLO("models/detection_model2.pt")

# Функция для загрузки моделей EfficientNet
def load_efficientnet_model(model_path, num_classes, device):
    """Загрузка модели EfficientNet B2"""
    try:
        model = models.efficientnet_b2(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        class_names = checkpoint.get('class_names', [])
        print(f"Модель {os.path.basename(model_path)} загружена успешно")
        print(f"Классы модели: {class_names}")
        return model, class_names
    except Exception as e:
        print(f"Ошибка загрузки модели {model_path}: {e}")
        return None, []

# Функция для загрузки модели дефектов YOLO
def load_defects_model(model_path, device):
    """Загрузка YOLO модели для детекции дефектов"""
    try:
        model = YOLO(model_path)
        print(f"Модель дефектов YOLO загружена успешно")
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели дефектов {model_path}: {e}")
        return None

# Загрузка моделей классификации
def load_classification_models():
    """Загрузка моделей для классификации пород и дефектов"""
    tree_model, bush_model, defects_model = None, None, None
    tree_class_names, bush_class_names, defects_class_names = [], [], []
    
    try:
        # Загрузка модели деревьев (EfficientNet)
        tree_model, tree_class_names = load_efficientnet_model(
            'models/model_tree_sota.pth', 
            num_classes=14,  # для деревьев
            device=DEVICE
        )
    except Exception as e:
        print(f"⚠️ Модель деревьев не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели кустов (EfficientNet)
        bush_model, bush_class_names = load_efficientnet_model(
            'models/model_bush_sota.pth', 
            num_classes=15,  # для кустов
            device=DEVICE
        )
    except Exception as e:
        print(f"⚠️ Модель кустов не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели дефектов (YOLO)
        defects_model = load_defects_model('models/defects_model.pt', DEVICE)
        # Классы для модели дефектов (из вашего тестирования)
        defects_class_names = ["duplo", "gnilye", "pni", "rak", "sukhie", 
                              "sukhobochina", "treshchina", "vrediteli", "korni"]
        print(f"Классы модели дефектов: {defects_class_names}")
    except Exception as e:
        print(f"Модель дефектов не найдена или ошибка загрузки: {e}")
    
    return (tree_model, bush_model, defects_model, 
            tree_class_names, bush_class_names, defects_class_names)

# Загружаем модели классификации
(tree_model, bush_model, defects_model, 
 tree_class_names, bush_class_names, defects_class_names) = load_classification_models()

# Трансформации для классификации EfficientNet
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_plant_top2(plant_roi, model, class_names, top_k=2):
    """Классификация растения с возвратом топ-K предсказаний для EfficientNet"""
    if model is None:
        return []
    
    try:
        # Преобразуем ROI в PIL Image
        plant_roi_rgb = cv2.cvtColor(plant_roi, cv2.COLOR_BGR2RGB)
        plant_pil = Image.fromarray(plant_roi_rgb)
        
        # Применяем трансформации
        image_tensor = transform(plant_pil).unsqueeze(0).to(DEVICE)
        
        # Классификация
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Получаем топ-K предсказаний
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[i].item()
            confidence = top_probs[i].item()
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            results.append({
                'name': class_name,
                'confidence': confidence
            })
        
        return results
        
    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        return [{'name': 'Ошибка классификации', 'confidence': 0.0}]

def detect_defects_with_boxes(plant_roi, model, conf_threshold=0.4):
    """Детекция дефектов с bounding boxes"""
    if model is None:
        return [], []
    
    try:
        # Предсказание дефектов
        results = model.predict(plant_roi, conf=conf_threshold)
        
        defects_info = []
        all_boxes = []
        
        for r in results:
            boxes = r.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                class_id = int(cls)
                class_name = defects_class_names[class_id] if class_id < len(defects_class_names) else f"class_{class_id}"
                
                defects_info.append({
                    'name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
                
                all_boxes.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_name': class_name,
                    'confidence': float(conf)
                })
        
        return defects_info, all_boxes
        
    except Exception as e:
        print(f"Ошибка при детекции дефектов: {e}")
        return [], []

def visualize_defects_boxes(image, defects_boxes, border_margin=5):
    """Визуализация bounding boxes дефектов на изображении с отступами от границ"""
    # Создаем копию, чтобы не изменять оригинал
    img_display = image.copy()
    
    # Если изображение пустое, возвращаем как есть
    if img_display.size == 0:
        return img_display
    
    img_height, img_width = img_display.shape[:2]
    
    colors = {
        "duplo": (255, 0, 0),      # Красный
        "gnilye": (0, 255, 0),     # Зеленый
        "pni": (0, 0, 255),        # Синий
        "rak": (255, 255, 0),      # Голубой
        "sukhie": (255, 0, 255),   # Пурпурный
        "sukhobochina": (0, 255, 255),  # Желтый
        "treshchina": (128, 0, 128),    # Фиолетовый
        "vrediteli": (128, 128, 0),     # Оливковый
        "korni": (0, 128, 128)          # Бирюзовый
    }
    
    for defect in defects_boxes:
        bbox = defect['bbox']
        class_name = defect['class_name']
        confidence = defect['confidence']
        
        color = colors.get(class_name, (128, 128, 128))  # Серый по умолчанию
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Корректируем координаты бокса, если он слишком близко к границам
        x1_adj = max(x1, border_margin)
        y1_adj = max(y1, border_margin)
        x2_adj = min(x2, img_width - border_margin)
        y2_adj = min(y2, img_height - border_margin)
        
        # Проверяем валидность bounding box после корректировки
        if x2_adj > x1_adj and y2_adj > y1_adj:
            # Рисуем прямоугольник с скорректированными координатами
            cv2.rectangle(img_display, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 2)
            
            # Подпись
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Позиция для текста (над bounding box) с учетом границ
            text_x = x1_adj
            text_y = max(y1_adj - 10, label_size[1] + 5)
            
            # Корректируем позицию текста, если он выходит за границы
            if text_y - label_size[1] - 5 < border_margin:
                text_y = y1_adj + label_size[1] + 10  # Перемещаем текст под бокс
            
            if text_x + label_size[0] > img_width - border_margin:
                text_x = x2_adj - label_size[0] - 5  # Сдвигаем текст влево
            
            # Убедимся, что текст не выходит за нижнюю границу
            if text_y > img_height - border_margin:
                text_y = y1_adj - 10
            
            # Фон для текста с учетом границ
            bg_x1 = max(text_x, border_margin)
            bg_y1 = max(text_y - label_size[1] - 5, border_margin)
            bg_x2 = min(text_x + label_size[0], img_width - border_margin)
            bg_y2 = min(text_y, img_height - border_margin)
            
            # Рисуем фон только если он валиден
            if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                cv2.rectangle(img_display, 
                             (bg_x1, bg_y1),
                             (bg_x2, bg_y2),
                             color, -1)
            
            # Текст (двойной для лучшей читаемости)
            if border_margin <= text_x <= img_width - border_margin and border_margin <= text_y <= img_height - border_margin:
                cv2.putText(img_display, label, (text_x, text_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  
                cv2.putText(img_display, label, (text_x, text_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(f"⚠️ Бокс дефекта '{class_name}' был слишком близко к границе и был пропущен")
    
    return img_display

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
def visualize_boxes_with_classification(image, boxes, classification_results, class_names=None, border_margin=10):
    """
    Визуализация боксов растений (деревья/кусты) с подписями внутри изображения.
    НЕ выполняет изменение размера, чтобы сохранить координаты для корректной вставки дефектов.
    """

    img_display = image.copy()
    img_height, img_width = img_display.shape[:2]

    # Словарь имён классов по умолчанию
    if class_names is None:
        class_names = {0: "Tree", 1: "Bush"}

    for i, (box, result) in enumerate(zip(boxes, classification_results)):
        # Достаем координаты бокса
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        class_name = class_names.get(class_id, f"class_{class_id}")

        # Цвет рамки: зелёный для деревьев, розовый для кустов
        color = (69, 252, 3) if class_id == 0 else (207, 109, 132)

        # Корректируем координаты, если они близко к границам
        x1_adj = max(int(x1), border_margin)
        y1_adj = max(int(y1), border_margin)
        x2_adj = min(int(x2), img_width - border_margin)
        y2_adj = min(int(y2), img_height - border_margin)

        # Проверка на валидность бокса
        if x1_adj < x2_adj and y1_adj < y2_adj:
            # Рисуем прямоугольник
            cv2.rectangle(img_display, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 3)

            # Формируем подпись (тип растения и индекс)
            label = f"{class_name} #{i+1}"

            # Размер текста для фона
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Позиция подписи ВНУТРИ бокса (в левом верхнем углу)
            text_x = x1_adj + 5
            text_y = y1_adj + label_size[1] + 10

            # Корректируем позицию текста, если выходит за границы
            if text_y > img_height - border_margin:
                text_y = y1_adj - 10
            if text_x + label_size[0] > img_width - border_margin:
                text_x = x2_adj - label_size[0] - 5
            if text_y - label_size[1] - 5 < border_margin:
                text_y = y1_adj + label_size[1] + 10

            # Фон под подписью (внутри бокса)
            bg_x1 = max(text_x - 2, border_margin)
            bg_y1 = max(text_y - label_size[1] - 5, border_margin)
            bg_x2 = min(text_x + label_size[0] + 2, img_width - border_margin)
            bg_y2 = min(text_y + 2, img_height - border_margin)

            # Рисуем фон, если он корректен
            if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                cv2.rectangle(
                    img_display,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    color,
                    -1
                )

            # Наносим текст
            cv2.putText(
                img_display,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

        else:
            print(f"⚠️ Предупреждение: бокс #{i+1} слишком близко к границе и пропущен")

    # Не выполняем ресайз — чтобы координаты оставались в оригинальном масштабе
    # Это важно, чтобы вставка ROI с дефектами оставалась корректной.
    return img_display


def visualize_boxes_with_classification_with_expantion(image, boxes, classification_results, class_names=None, expand_percent=0.15):
    """Визуализация боксов с информацией о классификации с расширением на 10%"""
    img_display = image.copy()
    
    if class_names is None:
        class_names = {0: "Tree", 1: "Bush"}  
    
    for i, (box, result) in enumerate(zip(boxes, classification_results)):
        x1, y1, x2, y2, conf, cls = box
        
        # Расширяем bounding box на 10%
        width = x2 - x1
        height = y2 - y1
        expand_x = width * expand_percent
        expand_y = height * expand_percent
        
        x1_expanded = max(0, int(x1 - expand_x))
        y1_expanded = max(0, int(y1 - expand_y))
        x2_expanded = min(img_display.shape[1], int(x2 + expand_x))
        y2_expanded = min(img_display.shape[0], int(y2 + expand_y))
        
        class_id = int(cls)
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Разные цвета для деревьев и кустов
        color = (69, 252, 3) if class_id == 0 else (207, 109, 132)
        
        # Рисуем прямоугольник (расширенный)
        cv2.rectangle(img_display, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), color, 3)
        
        # Формируем подпись - только номер и класс
        label = f"{class_name} #{i+1}"
        
        # Размер текста для фона
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Позиция подписи ВНУТРИ расширенного бокса (в левом верхнем углу)
        text_x = x1_expanded + 5
        text_y = y1_expanded + label_size[1] + 10
        
        # Убедимся что текст не выходит за границы изображения
        if text_y > img_display.shape[0]:
            text_y = y1_expanded - 10
        
        # Фон для текста (внутри расширенного бокса)
        cv2.rectangle(img_display, 
                     (text_x - 2, text_y - label_size[1] - 5),
                     (text_x + label_size[0] + 2, text_y + 2),
                     color, -1)
        
        # Текст
        cv2.putText(img_display, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    # Изменяем размер для отображения (оставляем как было)
    height, width = img_display.shape[:2]
    max_display_size = 800
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_display = cv2.resize(img_display, (new_width, new_height))
    
    return img_display
# Словарь для перевода дефектов с английского на русский
DEFECTS_TRANSLATION = {
    "duplo": "Дупло",
    "gnilye": "Гнилые участки", 
    "korni": "Корни",
    "pni": "Пень",
    "rak": "Рак дерева",
    "sukhie": "Сухие ветви",
    "sukhobochina": "Сухобочина",
    "treshchina": "Трещина",
    "vrediteli": "Вредители",
    "zdorovye": "Здоровое"
}

def translate_defect(defect_name):
    """Переводит название дефекта на русский"""
    if not defect_name:
        return "Нормальное"
    
    defect_lower = defect_name.lower().strip()
    return DEFECTS_TRANSLATION.get(defect_lower, defect_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        print("Получен запрос на загрузку")
        file = request.files["file"]
        if not file:
            return jsonify({"error": "Файл не предоставлен"}), 400
            
        img_bytes = file.read()
        print(f"Размер файла: {len(img_bytes)} байт")
        
        # читаем картинку из байтов
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("Не удалось декодировать изображение")
            return jsonify({"error": "Не удалось прочитать изображение"}), 400

        print(f"Размер изображения: {img.shape}")

        # Предсказание детекции
        print("🔍 Запуск детекции...")
        results = detection_model.predict(img, conf=0.3)
        boxes = results[0].boxes.data.cpu().numpy()

        print(f"📦 Найдено боксов до фильтрации: {len(boxes)}")

        # Фильтрация и объединение боксов
        filtered_boxes = filter_small_boxes(boxes, img.shape, 
                                          min_area_percent=0.01, 
                                          min_side_percent=0.1)
        merged_boxes = advanced_merge_boxes(filtered_boxes, size_weight=0.8, conf_weight=0.2)
        
        print(f"Обнаружено боксов после фильтрации: {len(merged_boxes)}")
        
        # ЕСЛИ НИЧЕГО НЕ ОБНАРУЖЕНО - возвращаем специальный флаг
        if len(merged_boxes) == 0:
            print("На фото не обнаружены деревья или кустарники")
            # Кодируем оригинальное изображение
            _, buffer = cv2.imencode(".jpg", img)
            encoded_img = base64.b64encode(buffer).decode("utf-8")
            
            return jsonify({
                "image": encoded_img,
                "table_data": [],
                "no_objects_detected": True
            })
        
        print(f"Обнаружено объектов: {len(merged_boxes)}")
        
        classification_results = []
        table_data = []
        
        for i, box in enumerate(merged_boxes):
            x1, y1, x2, y2, conf, detection_class = box
            
            # Вырезаем область с растением с расширением на 10% для классификации
            expand_percent = 0.10  # 10% расширение
            
            # Вычисляем расширенные координаты для классификации
            width = x2 - x1
            height = y2 - y1
            expand_x = width * expand_percent
            expand_y = height * expand_percent
            
            x1_expanded = max(0, int(x1 - expand_x))
            y1_expanded = max(0, int(y1 - expand_y))
            x2_expanded = min(img.shape[1], int(x2 + expand_x))
            y2_expanded = min(img.shape[0], int(y2 + expand_y))
            
            # ROI для классификации (расширенный)
            plant_roi_for_classification = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            # ROI для отображения (оригинальный bbox)
            # plant_roi_for_display = img[int(y1):int(y2), int(x1):int(x2)]
            plant_roi_for_display = img[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
            
            if plant_roi_for_classification.size == 0:
                print(f"⚠️ Не удалось вырезать область для растения {i+1}")
                classification_results.append({
                    'species': None,
                    'species_confidence': 0.0,
                    'defects': None,
                    'defects_confidence': 0.0,
                    'defects_boxes': []
                })
                continue
            
            # Проверка размера ROI для классификации
            if (plant_roi_for_classification.shape[0] < 10 or 
                plant_roi_for_classification.shape[1] < 10):
                print(f"⚠️ Слишком маленькая область для растения {i+1}: {plant_roi_for_classification.shape}")
                classification_results.append({
                    'species': None,
                    'species_confidence': 0.0,
                    'defects': None,
                    'defects_confidence': 0.0,
                    'defects_boxes': []
                })
                continue
            
            # Классификация породы (используем расширенный ROI)
            species_name, species_confidence = "", 0.0
            plant_type = ""
            species_top2 = []
            if detection_class == 0 and tree_model is not None:  # Дерево
                species_top2 = classify_plant_top2(plant_roi_for_classification, tree_model, tree_class_names, top_k=2)
                species_name = species_top2[0]['name'] if species_top2 else ""
                species_confidence = species_top2[0]['confidence'] if species_top2 else 0.0
                plant_type = "Дерево"
                print(f"Растение {i+1} (Дерево): {species_name} (уверенность: {species_confidence:.2%})")
            elif detection_class == 1 and bush_model is not None:  # Куст
                species_top2 = classify_plant_top2(plant_roi_for_classification, bush_model, bush_class_names, top_k=2)
                species_name = species_top2[0]['name'] if species_top2 else ""
                species_confidence = species_top2[0]['confidence'] if species_top2 else 0.0
                plant_type = "Куст"
                print(f"🪴 Растение {i+1} (Куст): {species_name} (уверенность: {species_confidence:.2%})")

            # Детекция дефектов с bounding boxes (используем расширенный ROI)
            defects_info, defects_boxes = [], []
            defects_name, defects_confidence = "", 0.0
            defects_top2 = []
            
            if defects_model is not None:
                defects_info, defects_boxes = detect_defects_with_boxes(plant_roi_for_classification, defects_model)
                
                if defects_info:
                    # Берем самый уверенный дефект
                    best_defect = max(defects_info, key=lambda x: x['confidence'])
                    defects_name = best_defect['name']
                    defects_confidence = best_defect['confidence']
                    
                    # Формируем топ-2 дефектов
                    sorted_defects = sorted(defects_info, key=lambda x: x['confidence'], reverse=True)[:2]
                    defects_top2 = [{'name': d['name'], 'confidence': d['confidence']} for d in sorted_defects]
                    
                    print(f"Растение {i+1} - Дефекты: {defects_name} (уверенность: {defects_confidence:.2%}), найдено bbox: {len(defects_boxes)}")
                else:
                    defects_name = "zdorovye"
                    defects_confidence = 1.0
                    defects_top2 = [{'name': 'zdorovye', 'confidence': 1.0}]
                    print(f"🔧 Растение {i+1} - Дефекты не обнаружены, статус: здоровое")

            # Сохраняем результаты классификации для визуализации
            classification_results.append({
                'species': species_name,
                'species_confidence': species_confidence,
                'species_top2': species_top2,
                'defects': defects_name,
                'defects_confidence': defects_confidence,
                'defects_top2': defects_top2,
                'defects_boxes': defects_boxes,
                'expanded_coords': (x1_expanded, y1_expanded, x2_expanded, y2_expanded)  # Сохраняем для отображения дефектов
            })

            # ФОРМИРУЕМ ДАННЫЕ ДЛЯ ТАБЛИЦЫ
            translated_defect = translate_defect(defects_name)
            # Для альтернативных вариантов тоже переводим
            alt_defects = []
            if defects_top2 and len(defects_top2) > 1:
                for defect in defects_top2[1:]:
                    alt_defects.append({
                        'name': translate_defect(defect['name']),
                        'confidence': round(defect['confidence'] * 100, 1)
                    })

            table_data.append({
                'id': i + 1,
                'plant_type': plant_type,
                'species': species_name if species_name else "Неизвестно",
                'species_confidence': round(species_confidence * 100, 1),
                'species_alt': species_top2[1] if len(species_top2) > 1 else None,
                'species_alt_confidence': round(species_top2[1]['confidence'] * 100, 1) if len(species_top2) > 1 else 0,
                'status': translated_defect,
                'defects_confidence': round(defects_confidence * 100, 1),
                'defects_alt': alt_defects[0] if alt_defects else None,
                'defects_alt_confidence': alt_defects[0]['confidence'] if alt_defects else 0,
                'defects_count': len(defects_boxes)
            })
            print(table_data)
        
        # Визуализация результатов
        print(f"🎨 Визуализация {len(merged_boxes)} боксов с {len(classification_results)} результатами классификации")
        
        # Сначала визуализируем детекцию растений (оригинальные bbox)
        class_names = {0: "Дерево", 1: "Куст"}
        final_display = visualize_boxes_with_classification(img, merged_boxes, classification_results)

        # Затем добавляем bounding boxes дефектов для каждого растения
        for i, (box, result) in enumerate(zip(merged_boxes, classification_results)):
            if result['defects_boxes']:
                x1_exp, y1_exp, x2_exp, y2_exp = result['expanded_coords']
                
                # Создаем ROI для текущего растения (расширенная область)
                plant_roi = final_display[y1_exp:y2_exp, x1_exp:x2_exp]
                
                if plant_roi.size > 0:  # Проверяем, что ROI не пустой
                    # Визуализируем дефекты на ROI
                    plant_with_defects = visualize_defects_boxes(plant_roi, result['defects_boxes'])
                    
                    # Проверяем совпадение размеров перед вставкой
                    if plant_with_defects.shape == plant_roi.shape:
                        final_display[y1_exp:y2_exp, x1_exp:x2_exp] = plant_with_defects
                    else:
                        print(f"⚠️ Размеры не совпадают для растения {i+1}: ROI {plant_roi.shape}, defects {plant_with_defects.shape}")
                        # Пробуем изменить размер
                        try:
                            plant_with_defects_resized = cv2.resize(plant_with_defects, (plant_roi.shape[1], plant_roi.shape[0]))
                            final_display[y1_exp:y2_exp, x1_exp:x2_exp] = plant_with_defects_resized
                        except Exception as resize_error:
                            print(f"Ошибка изменения размера для растения {i+1}: {resize_error}")

        # Кодируем обратно в base64
        _, buffer = cv2.imencode(".jpg", final_display)
        encoded_img = base64.b64encode(buffer).decode("utf-8")

        print(f"Отправка результата: изображение {len(encoded_img)} символов, таблица {len(table_data)} записей")

        return jsonify({
            "image": encoded_img,
            "table_data": table_data,
            "no_objects_detected": False
        })

    except Exception as e:
        print(f"Критическая ошибка в upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)