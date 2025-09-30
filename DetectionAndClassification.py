import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os
from PIL import Image
import cv2
from collections import Counter
from ultralytics import YOLO



from ClassificationModel import ImprovedCNN

# Отключаем warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 256
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {DEVICE}")


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

def detect_and_classify_plants(image_path, detection_model, tree_model, bush_model, 
                              tree_class_names, bush_class_names, 
                              min_confidence=0.3, min_area_percent=0.01):
    """Детекция с последующей классификацией по типу растения"""
    
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, []
    
    print(f"🔍 Детекция растений на изображении: {Path(image_path).name}")
    
    # Детекция с помощью YOLO
    try:
        results = detection_model.predict(img, conf=min_confidence)
        boxes = results[0].boxes.data.cpu().numpy()
    except Exception as e:
        print(f"Ошибка при детекции: {e}")
        return None, []
    
    # Фильтрация маленьких боксов
    filtered_boxes = filter_small_boxes(boxes, img.shape, 
                                      min_area_percent=min_area_percent, 
                                      min_side_percent=0.1)
    
    # Объединение пересекающихся боксов
    merged_boxes = advanced_merge_boxes(filtered_boxes, size_weight=0.8, conf_weight=0.2)
    
    print(f"Обнаружено растений: {len(merged_boxes)}")
    
    if len(merged_boxes) == 0:
        print("Растения не обнаружены")
        return img, []
    
    # Подготовка трансформаций для классификации
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classification_results = []
    
    # Классификация каждого обнаруженного растения
    for i, box in enumerate(merged_boxes):
        x1, y1, x2, y2, conf, detection_class = box
        
        # detection_class: 0 - дерево, 1 - куст
        plant_type = "дерево" if detection_class == 0 else "куст"
        
        # Выбираем соответствующую модель и классы
        if detection_class == 0 and tree_model is not None:  # Дерево
            classification_model = tree_model
            class_names = tree_class_names
            model_type = "деревьев"
        elif detection_class == 1 and bush_model is not None:  # Куст
            classification_model = bush_model  
            class_names = bush_class_names
            model_type = "кустов"
        else:
            print(f"Для {plant_type} нет модели классификации")
            # Сохраняем информацию о детекции без классификации
            classification_results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'plant_type': plant_type,
                'detection_confidence': conf,
                'species': "Неизвестно",
                'classification_confidence': 0.0,
                'species_id': -1,
                'detection_class': detection_class
            })
            continue
        
        # Вырезаем область с растением
        padding = 5
        x1_padded = max(0, int(x1) - padding)
        y1_padded = max(0, int(y1) - padding)
        x2_padded = min(img.shape[1], int(x2) + padding)
        y2_padded = min(img.shape[0], int(y2) + padding)
        
        plant_roi = img[y1_padded:y2_padded, x1_padded:x2_padded]
        
        if plant_roi.size == 0:
            print(f"Не удалось вырезать область для {plant_type} {i+1}")
            continue
            
        # Преобразуем в PIL Image и классифицируем
        try:
            plant_roi_rgb = cv2.cvtColor(plant_roi, cv2.COLOR_BGR2RGB)
            plant_pil = Image.fromarray(plant_roi_rgb)
            
            image_tensor = transform(plant_pil).unsqueeze(0).to(DEVICE)
            
            classification_model.eval()
            with torch.no_grad():
                outputs = classification_model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            
            species_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
            
            classification_results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'plant_type': plant_type,
                'detection_confidence': conf,
                'species': species_name,
                'classification_confidence': confidence,
                'species_id': predicted_class,
                'detection_class': detection_class
            })
            
            print(f"Растение {i+1} ({plant_type}): {species_name} (уверенность: {confidence:.2%})")
            
        except Exception as e:
            print(f"Ошибка при классификации {plant_type} {i+1}: {e}")
            # Сохраняем информацию о детекции без классификации
            classification_results.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'plant_type': plant_type,
                'detection_confidence': conf,
                'species': "Ошибка классификации",
                'classification_confidence': 0.0,
                'species_id': -1,
                'detection_class': detection_class
            })
            continue
    
    return img, classification_results

def visualize_detection_with_classification(image, boxes, classification_results):
    """Визуализация детекции с классификацией с улучшенным управлением окном"""
    img_display = image.copy()
    
    for i, (box, result) in enumerate(zip(boxes, classification_results)):
        x1, y1, x2, y2 = result['box']
        plant_type = result['plant_type']
        species = result['species']
        det_conf = result['detection_confidence']
        cls_conf = result['classification_confidence']
        
        # Разные цвета для деревьев и кустов
        if result['detection_class'] == 0:  # Дерево
            color = (0, 255, 0)  # Зеленый
        else:  # Куст
            color = (0, 165, 255)  # Оранжевый
        
        # Рисуем прямоугольник
        cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Подпись с видом растения и уверенностью
        label = f"{plant_type}: {species}"
        confidence_label = f"det({det_conf:.2f}), cls({cls_conf:.2f})"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        confidence_size = cv2.getTextSize(confidence_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        total_height = label_size[1] + confidence_size[1] + 15
        
        # Фон для текста
        cv2.rectangle(img_display, 
                     (int(x1), int(y1) - total_height),
                     (int(x1) + max(label_size[0], confidence_size[0]), int(y1)),
                     color, -1)
        
        # Текст
        cv2.putText(img_display, label, 
                   (int(x1), int(y1) - confidence_size[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(img_display, confidence_label,
                   (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Номер растения
        cv2.putText(img_display, f"#{i+1}", 
                   (int(x1), int(y1) - total_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Изменяем размер для удобного отображения
    height, width = img_display.shape[:2]
    max_display_size = 1200
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_display = cv2.resize(img_display, (new_width, new_height))
    
    # Показываем изображение с возможностью закрытия
    window_name = "Детекция и классификация растений (нажмите любую клавишу для продолжения)"
    cv2.imshow(window_name, img_display)
    print("Изображение показано. Нажмите любую клавишу в окне изображения чтобы продолжить...")
    
    # Ждем нажатия клавиши (0 - бесконечное ожидание)
    cv2.waitKey(0)
    
    # Закрываем все окна OpenCV
    cv2.destroyAllWindows()
    
    # Небольшая задержка чтобы окна точно закрылись
    import time
    time.sleep(0.5)
    
    # Сохранение результата
    output_path = "detection_classification_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    print(f"Результат сохранен как: {output_path}")
    
    return img_display

def load_separated_models():
    """Загрузка раздельных моделей для деревьев и кустов"""
    tree_model, bush_model = None, None
    tree_class_names, bush_class_names = [], []
    
    try:
        # Загрузка модели деревьев
        checkpoint = torch.load('model_trees.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        tree_model = ImprovedCNN(num_classes)
        tree_model.load_state_dict(checkpoint['model_state_dict'])
        tree_class_names = checkpoint['class_names']
        print("Модель деревьев загружена")
    except Exception as e:
        print(f"Модель деревьев не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели кустов
        checkpoint = torch.load('model_bushes.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        bush_model = ImprovedCNN(num_classes)
        bush_model.load_state_dict(checkpoint['model_state_dict'])
        bush_class_names = checkpoint['class_names']
        print("Модель кустов загружена")
    except Exception as e:
        print(f"Модель кустов не найдена или ошибка загрузки: {e}")
    
    return tree_model, bush_model, tree_class_names, bush_class_names

def detect_and_classify_complete(image_path, detection_model, tree_model, bush_model, defects_model,
                               tree_class_names, bush_class_names, defects_class_names,
                               min_confidence=0.3, min_area_percent=0.01):
    """Полная классификация: породы + дефекты"""
    
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None, []
    
    print(f"🔍 Детекция и полная классификация: {Path(image_path).name}")
    
    # Детекция с помощью YOLO
    try:
        results = detection_model.predict(img, conf=min_confidence)
        boxes = results[0].boxes.data.cpu().numpy()
    except Exception as e:
        print(f"Ошибка при детекции: {e}")
        return None, []
    
    # Фильтрация маленьких боксов
    filtered_boxes = filter_small_boxes(boxes, img.shape, 
                                      min_area_percent=min_area_percent, 
                                      min_side_percent=0.1)
    
    # Объединение пересекающихся боксов
    merged_boxes = advanced_merge_boxes(filtered_boxes, size_weight=0.8, conf_weight=0.2)
    
    print(f"Обнаружено растений: {len(merged_boxes)}")
    
    if len(merged_boxes) == 0:
        print("Растения не обнаружены")
        return img, []
    
    # Подготовка трансформаций для классификации
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classification_results = []
    
    # Классификация каждого обнаруженного растения
    for i, box in enumerate(merged_boxes):
        x1, y1, x2, y2, conf, detection_class = box
        
        # detection_class: 0 - дерево, 1 - куст
        plant_type = "дерево" if detection_class == 0 else "куст"
        
        # ВЫБОР МОДЕЛИ ДЛЯ КЛАССИФИКАЦИИ ПОРОДЫ
        species_model = None
        species_class_names = []
        if detection_class == 0 and tree_model is not None:  # Дерево
            species_model = tree_model
            species_class_names = tree_class_names
        elif detection_class == 1 and bush_model is not None:  # Куст
            species_model = bush_model  
            species_class_names = bush_class_names
        
        # Вырезаем область с растением
        padding = 5
        x1_padded = max(0, int(x1) - padding)
        y1_padded = max(0, int(y1) - padding)
        x2_padded = min(img.shape[1], int(x2) + padding)
        y2_padded = min(img.shape[0], int(y2) + padding)
        
        plant_roi = img[y1_padded:y2_padded, x1_padded:x2_padded]
        
        if plant_roi.size == 0:
            print(f"Не удалось вырезать область для растения {i+1}")
            continue
        
        species_name = "Неизвестно"
        species_confidence = 0.0
        defects_name = "Неизвестно"
        defects_confidence = 0.0
        
        # КЛАССИФИКАЦИЯ ПОРОДЫ
        if species_model is not None:
            try:
                plant_roi_rgb = cv2.cvtColor(plant_roi, cv2.COLOR_BGR2RGB)
                plant_pil = Image.fromarray(plant_roi_rgb)
                
                image_tensor = transform(plant_pil).unsqueeze(0).to(DEVICE)
                
                species_model.eval()
                with torch.no_grad():
                    outputs = species_model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_class = torch.max(probabilities, 0)
                
                predicted_class = predicted_class.item()
                species_confidence = confidence.item()
                
                species_name = species_class_names[predicted_class] if predicted_class < len(species_class_names) else f"Class {predicted_class}"
                
                print(f"Растение {i+1} ({plant_type}): {species_name} (уверенность: {species_confidence:.2%})")
                
            except Exception as e:
                print(f"Ошибка при классификации породы растения {i+1}: {e}")
        
        # КЛАССИФИКАЦИЯ ДЕФЕКТОВ
        if defects_model is not None:
            try:
                defects_model.eval()
                with torch.no_grad():
                    outputs = defects_model(image_tensor)  # Используем тот же тензор
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_class = torch.max(probabilities, 0)
                
                predicted_class = predicted_class.item()
                defects_confidence = confidence.item()
                
                defects_name = defects_class_names[predicted_class] if predicted_class < len(defects_class_names) else f"Class {predicted_class}"
                
                print(f"Растение {i+1} - Дефекты: {defects_name} (уверенность: {defects_confidence:.2%})")
                
            except Exception as e:
                print(f"Ошибка при классификации дефектов растения {i+1}: {e}")
        
        classification_results.append({
            'box': (int(x1), int(y1), int(x2), int(y2)),
            'plant_type': plant_type,
            'detection_confidence': conf,
            'species': species_name,
            'species_confidence': species_confidence,
            'defects': defects_name,
            'defects_confidence': defects_confidence,
            'detection_class': detection_class,
            'plant_number': i+1  # Добавляем номер растения
        })
    
    return img, classification_results

def visualize_complete_detection(image, classification_results):
    img_display = image.copy()
    
    for result in classification_results:
        x1, y1, x2, y2 = result['box']
        species = result['species']
        defects = result['defects']
        det_conf = result['detection_confidence']
        species_conf = result['species_confidence']
        defects_conf = result['defects_confidence']
        plant_number = result['plant_number']
        
        # Разные цвета для деревьев и кустов
        if result['detection_class'] == 0:  # Дерево
            color = (0, 255, 0)  # Зеленый
        else:  # Куст
            color = (0, 165, 255)  # Оранжевый
        
        # Рисуем прямоугольник
        cv2.rectangle(img_display, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # Текст для отображения
        number_text = f"{plant_number}"
        species_text = f"Порода: {species}"
        defects_text = f"Дефекты: {defects}"
        confidence_text = f"Детекция: {det_conf:.2f}, Порода: {species_conf:.2f}, Дефекты: {defects_conf:.2f}"
        
        # Размеры текста
        number_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        species_size = cv2.getTextSize(species_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        defects_size = cv2.getTextSize(defects_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        confidence_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        
        # Вычисляем общую высоту блока текста
        total_height = number_size[1] + species_size[1] + defects_size[1] + confidence_size[1] + 20
        max_width = max(number_size[0], species_size[0], defects_size[0], confidence_size[0])
        
        # Фон для текста - БОЛЕЕ ЯРКИЙ И ЧЕТКИЙ
        cv2.rectangle(img_display, 
                     (int(x1), int(y1) - total_height),
                     (int(x1) + max_width + 10, int(y1)),
                     (50, 50, 50), -1)  # Темно-серый фон
        
        # Рамка вокруг фона
        cv2.rectangle(img_display, 
                     (int(x1), int(y1) - total_height),
                     (int(x1) + max_width + 10, int(y1)),
                     color, 2)
        
        # Отображаем текст с разными размерами и цветами
        y_offset = y1 - 5
        
        # Номер растения - КРУПНЫЙ И ЯРКИЙ
        cv2.putText(img_display, number_text, 
                   (int(x1) + 5, int(y_offset)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset -= (number_size[1] + 5)
        
        # Порода
        cv2.putText(img_display, species_text,
                   (int(x1) + 5, int(y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset -= (species_size[1] + 5)
        
        # Дефекты
        cv2.putText(img_display, defects_text,
                   (int(x1) + 5, int(y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)  # Желтый цвет для дефектов
        y_offset -= (defects_size[1] + 5)
        
        # Уверенности (мелким шрифтом)
        cv2.putText(img_display, confidence_text,
                   (int(x1) + 5, int(y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Изменяем размер для удобного отображения
    height, width = img_display.shape[:2]
    max_display_size = 1200
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_display = cv2.resize(img_display, (new_width, new_height))
    
    # Показываем изображение
    window_name = "Детекция и классификация растений (нажмите любую клавишу)"
    cv2.imshow(window_name, img_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    import time
    time.sleep(0.5)
    
    # Сохранение результата
    output_path = "complete_detection_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    print(f"Результат сохранен как: {output_path}")
    
    return img_display

def load_defects_model():
    """Загрузка модели классификации дефектов"""
    defects_model = None
    defects_class_names = []
    
    try:
        checkpoint = torch.load('model_defects.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        defects_model = ImprovedCNN(num_classes)
        defects_model.load_state_dict(checkpoint['model_state_dict'])
        defects_class_names = checkpoint['class_names']
        print("Модель дефектов загружена")
    except Exception as e:
        print(f"Модель дефектов не найдена или ошибка загрузки: {e}")
    
    return defects_model, defects_class_names
