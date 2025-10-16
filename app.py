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

app = Flask(__name__)
CORS(app)  # разрешаем запросы с фронта

# Отключаем warnings
warnings.filterwarnings('ignore')

# Определяем устройство
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Используемое устройство: {DEVICE}")

# Параметры для классификации
IMG_SIZE = (224, 224)

# Загружаем модели
print("🔍 Загрузка моделей...")
detection_model = YOLO("models/detection_model.pt")

# Функция для создания архитектуры модели (должна совпадать с обучением)
def create_improved_model(num_classes):
    """Улучшенная модель с batch normalization и большей емкостью"""
    class ImprovedCNN(nn.Module):
        def __init__(self, num_classes):
            super(ImprovedCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return ImprovedCNN(num_classes)

# Загрузка моделей классификации
def load_classification_models():
    """Загрузка моделей для классификации пород и дефектов"""
    tree_model, bush_model, defects_model = None, None, None
    tree_class_names, bush_class_names, defects_class_names = [], [], []
    
    try:
        # Загрузка модели деревьев
        checkpoint = torch.load('models/model_trees.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        tree_model = create_improved_model(num_classes)
        tree_model.load_state_dict(checkpoint['model_state_dict'])
        tree_model.to(DEVICE)
        tree_model.eval()
        tree_class_names = checkpoint['class_names']
        print("✅ Модель деревьев загружена")
    except Exception as e:
        print(f"⚠️ Модель деревьев не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели кустов
        checkpoint = torch.load('models/model_bushes.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        bush_model = create_improved_model(num_classes)
        bush_model.load_state_dict(checkpoint['model_state_dict'])
        bush_model.to(DEVICE)
        bush_model.eval()
        bush_class_names = checkpoint['class_names']
        print("✅ Модель кустов загружена")
    except Exception as e:
        print(f"⚠️ Модель кустов не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели дефектов
        checkpoint = torch.load('models/model_defects.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        defects_model = create_improved_model(num_classes)
        defects_model.load_state_dict(checkpoint['model_state_dict'])
        defects_model.to(DEVICE)
        defects_model.eval()
        defects_class_names = checkpoint['class_names']
        print("✅ Модель дефектов загружена")
    except Exception as e:
        print(f"⚠️ Модель дефектов не найдена или ошибка загрузки: {e}")
    # Добавьте эту проверку после загрузки моделей
    print("🔍 Классы модели деревьев:", tree_class_names)
    print("🔍 Классы модели кустов:", bush_class_names)
    print("🔍 Классы модели дефектов:", defects_class_names)
    return (tree_model, bush_model, defects_model, 
            tree_class_names, bush_class_names, defects_class_names)

# Загружаем модели классификации
(tree_model, bush_model, defects_model, 
 tree_class_names, bush_class_names, defects_class_names) = load_classification_models()

# Трансформации для классификации
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_plant(plant_roi, model, class_names):
    """Классификация растения с помощью модели"""
    if model is None:
        return "Модель не загружена", 0.0
    
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
            confidence, predicted_class = torch.max(probabilities, 0)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        species_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
        
        return species_name, confidence
        
    except Exception as e:
        print(f"❌ Ошибка при классификации: {e}")
        return "Ошибка классификации", 0.0

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

def visualize_boxes_with_classification(image, boxes, classification_results, class_names=None):
    """Визуализация боксов с информацией о классификации"""
    img_display = image.copy()
    
    if class_names is None:
        class_names = {0: "Tree", 1: "Bush"}  
    
    for i, (box, result) in enumerate(zip(boxes, classification_results)):
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        class_name = class_names.get(class_id, f"class_{class_id}")
        
        # Разные цвета для деревьев и кустов
        color = (69, 252, 3) if class_id == 0 else (207, 109, 132)
        
        # Рисуем прямоугольник
        cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Формируем подпись - только номер и класс
        label = f"{class_name} #{i+1}"
        
        # Размер текста для фона
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Позиция подписи ВНУТРИ бокса (в левом верхнем углу)
        text_x = int(x1) + 5
        text_y = int(y1) + label_size[1] + 10
        
        # Убедимся что текст не выходит за границы изображения
        if text_y > img_display.shape[0]:
            text_y = int(y1) - 10
        
        # Фон для текста (внутри бокса)
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

def translate_defect(defect_name):
    """Переводит название дефекта на русский"""
    if not defect_name:
        return "Нормальное"
    
    defect_lower = defect_name.lower().strip()
    return DEFECTS_TRANSLATION.get(defect_lower, defect_name)  # Если нет в словаре, возвращаем как есть

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        print("📨 Получен запрос на загрузку")
        file = request.files["file"]
        if not file:
            return jsonify({"error": "Файл не предоставлен"}), 400
            
        img_bytes = file.read()
        print(f"📊 Размер файла: {len(img_bytes)} байт")
        
        # читаем картинку из байтов
        npimg = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            print("❌ Не удалось декодировать изображение")
            return jsonify({"error": "Не удалось прочитать изображение"}), 400

        print(f"🖼️ Размер изображения: {img.shape}")

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
        
        print(f"🔍 Обнаружено боксов после фильтрации: {len(merged_boxes)}")
        
        # ЕСЛИ НИЧЕГО НЕ ОБНАРУЖЕНО - возвращаем специальный флаг
        if len(merged_boxes) == 0:
            print("❌ На фото не обнаружены деревья или кустарники")
            # Кодируем оригинальное изображение
            _, buffer = cv2.imencode(".jpg", img)
            encoded_img = base64.b64encode(buffer).decode("utf-8")
            
            return jsonify({
                "image": encoded_img,
                "table_data": [],
                "no_objects_detected": True  # ← ВАЖНО: устанавливаем флаг
            })
        
        print(f"✅ Обнаружено объектов: {len(merged_boxes)}")
        
        classification_results = []
        table_data = []
        
        for i, box in enumerate(merged_boxes):
            x1, y1, x2, y2, conf, detection_class = box
            
            # Вырезаем область с растением
            padding = 5
            x1_padded = max(0, int(x1) - padding)
            y1_padded = max(0, int(y1) - padding)
            x2_padded = min(img.shape[1], int(x2) + padding)
            y2_padded = min(img.shape[0], int(y2) + padding)
            
            plant_roi = img[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if plant_roi.size == 0:
                print(f"⚠️ Не удалось вырезать область для растения {i+1}")
                # Добавляем пустой результат для соответствия количеству боксов
                classification_results.append({
                    'species': None,
                    'species_confidence': 0.0,
                    'defects': None,
                    'defects_confidence': 0.0
                })
                continue
            
            # Классификация породы
            species_name, species_confidence = "", 0.0
            plant_type = ""
            if detection_class == 0 and tree_model is not None:  # Дерево
                species_name, species_confidence = classify_plant(plant_roi, tree_model, tree_class_names)
                plant_type = "Дерево"
                print(f"🌳 Растение {i+1} (Дерево): {species_name} (уверенность: {species_confidence:.2%})")
            elif detection_class == 1 and bush_model is not None:  # Куст
                species_name, species_confidence = classify_plant(plant_roi, bush_model, bush_class_names)
                plant_type = "Куст"
                print(f"🪴 Растение {i+1} (Куст): {species_name} (уверенность: {species_confidence:.2%})")
            
            # Классификация дефектов
            defects_name, defects_confidence = "", 0.0
            if defects_model is not None:
                defects_name, defects_confidence = classify_plant(plant_roi, defects_model, defects_class_names)
                print(f"🔧 Растение {i+1} - Дефекты: {defects_name} (уверенность: {defects_confidence:.2%})")
            
            # Сохраняем результаты классификации для визуализации
            classification_results.append({
                'species': species_name,
                'species_confidence': species_confidence,
                'defects': defects_name,
                'defects_confidence': defects_confidence
            })
            
            # ФОРМИРУЕМ ДАННЫЕ ДЛЯ ТАБЛИЦЫ
            translated_defect = translate_defect(defects_name)
            table_data.append({
                'id': i + 1,
                'plant_type': plant_type,
                'species': species_name if species_name else "Неизвестно",
                'species_confidence': round(species_confidence * 100, 1),
                'status': translated_defect,
                'defects_confidence': round(defects_confidence * 100, 1)
            })
        
        # Визуализация результатов
        print(f"🎨 Визуализация {len(merged_boxes)} боксов с {len(classification_results)} результатами классификации")
        
        class_names = {0: "Дерево", 1: "Куст"}
        final_display = visualize_boxes_with_classification(img, merged_boxes, classification_results)

        # Кодируем обратно в base64
        _, buffer = cv2.imencode(".jpg", final_display)
        encoded_img = base64.b64encode(buffer).decode("utf-8")

        print(f"✅ Отправка результата: изображение {len(encoded_img)} символов, таблица {len(table_data)} записей")

        return jsonify({
            "image": encoded_img,
            "table_data": table_data,
            "no_objects_detected": False  # ← ВАЖНО: устанавливаем флаг
        })

    except Exception as e:
        print(f"❌ Критическая ошибка в upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Внутренняя ошибка сервера: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)