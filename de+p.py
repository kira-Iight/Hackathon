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

# Отключаем warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 256
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {DEVICE}")

class TreeDataset(Dataset):
    """Кастомный датасет для изображений растений с надежной обработкой ошибок"""
    def __init__(self, image_paths, labels, plant_types=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.plant_types = plant_types
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        plant_type = self.plant_types[idx] if self.plant_types is not None else 0
        
        image = self.load_image_safe(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, plant_type
    
    def load_image_safe(self, img_path):
        try:
            image = cv2.imread(str(img_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                return image
        except:
            pass
        
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except:
            pass
        
        print(f"⚠️ Не удалось загрузить {img_path}, создаем черное изображение")
        return Image.new('RGB', IMG_SIZE, color='black')

def load_tree_species_data_separated(porody_folder_path):
    """Загрузка данных с разделением на деревья и кусты"""
    porody_path = Path(porody_folder_path)
    
    print(f"🔍 Загрузка данных из: {porody_path.absolute()}")
    
    if not porody_path.exists():
        print(f"Папка не существует: {porody_path}")
        return [], [], [], [], [], []
    
    # Ищем CSV файл
    csv_path = porody_path / "labels" / "labels.csv"
    if not csv_path.exists():
        csv_path = porody_path / "labels.csv"
        if not csv_path.exists():
            print("CSV файл не найден")
            return [], [], [], [], [], []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"CSV загружен: {len(df)} записей")
        
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        return [], [], [], [], [], []
    
    # Папка с изображениями
    images_dir = porody_path / "images"
    if not images_dir.exists():
        print("Папка 'images' не найдена")
        return [], [], [], [], [], []
    
    # Разделяем на деревья и кусты
    tree_images = []
    tree_labels = []
    tree_species = []
    
    bush_images = []
    bush_labels = [] 
    bush_species = []
    
    successful = 0
    for _, row in df.iterrows():
        try:
            filename = str(row['filename']).strip()
            img_path = images_dir / filename
            
            if not img_path.exists():
                print(f"⚠️ Изображение не найдено: {filename}")
                continue
                
            # Определяем тип растения (0-дерево, 1-куст)
            if 'plant_type' in df.columns:
                plant_type = int(row['plant_type'])
            else:
                # Автоматическое определение по названию, если колонки нет
                species_name = str(row['species_name']).lower()
                if any(word in species_name for word in ['роза', 'куст', 'кустарник', 'можжевельник', 'сирень']):
                    plant_type = 1
                else:
                    plant_type = 0
            
            if plant_type == 0:  # Дерево
                tree_images.append(str(img_path))
                tree_labels.append(int(row['species_label']))
                tree_species.append(str(row['species_name']))
            else:  # Куст
                bush_images.append(str(img_path))
                bush_labels.append(int(row['species_label']))
                bush_species.append(str(row['species_name']))
                
            successful += 1
                
        except Exception as e:
            print(f"Ошибка обработки строки: {e}")
            continue
    
    print(f"Успешно загружено {successful}/{len(df)} изображений")
    print(f"Деревья: {len(tree_images)} изображений, {len(set(tree_labels))} видов")
    print(f"Кусты: {len(bush_images)} изображений, {len(set(bush_labels))} видов")
    
    return tree_images, tree_labels, tree_species, bush_images, bush_labels, bush_species

def get_enhanced_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_improved_model(num_classes):
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

def get_class_weights(labels, num_classes):
    class_counts = Counter(labels)
    print(f"📊 Распределение классов: {dict(class_counts)}")
    
    # Вычисляем веса обратно пропорционально частоте классов
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weights.append(1.0 / class_counts[i])
        else:
            weights.append(1.0)  # Если класса нет в данных
    
    # Нормализуем веса
    weights = np.array(weights)
    weights = weights / weights.sum() * len(weights)
    weights = torch.FloatTensor(weights).to(DEVICE)
    
    print(f"Веса классов: {weights.cpu().numpy()}")
    return weights

def check_class_balance(labels, class_names, dataset_name):
    """Проверка баланса классов"""
    label_counts = Counter(labels)
    
    print(f"\nБАЛАНС КЛАССОВ ({dataset_name}):")
    total_samples = len(labels)
    for label, count in label_counts.items():
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} изображений ({percentage:.1f}%)")

def train_single_model(model, train_loader, val_loader, num_epochs, num_classes, model_type="plant"):

    model = model.to(DEVICE)
    
    # Получаем веса классов для несбалансированных данных
    train_labels = []
    for _, labels, _ in train_loader:
        train_labels.extend(labels.cpu().numpy())
    
    class_weights = get_class_weights(train_labels, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_accuracy = 0
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        try:
            # Обучение
            model.train()
            running_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, labels, _) in enumerate(train_loader):
                try:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 10 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
                        
                except Exception as e:
                    print(f"Ошибка в батче {batch_idx}: {e}")
                    continue
            
            # Валидация
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    try:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    except Exception as e:
                        print(f"Ошибка при валидации: {e}")
                        continue
            
            accuracy = 100 * correct / total if total > 0 else 0
            avg_train_loss = running_loss / batch_count if batch_count > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            train_losses.append(avg_train_loss)
            val_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.2e}')
            

            scheduler.step(accuracy)
            
            # Early stopping и сохранение лучшей модели
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"Новый рекорд точности: {accuracy:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch+1}")
                break
            
        except Exception as e:
            print(f"Критическая ошибка в эпохе {epoch+1}: {e}")
            continue
    
    # Загружаем weights лучшей модели
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Загружены веса лучшей модели с точностью {best_accuracy:.2f}%")
    
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, {'train_loss': train_losses, 'val_accuracy': val_accuracies, 'learning_rates': learning_rates}

def can_use_stratified_split(labels, test_size=0.2):
    """Проверяет, можно ли использовать стратифицированное разделение"""
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())
    
    # Для стратификации нужно минимум 2 образца в каждом классе
    return min_samples_per_class >= 2 and all(count >= int(1/test_size) + 1 for count in label_counts.values())

def apply_data_augmentation_balance(image_paths, labels, max_samples_per_class=100):
    """Балансировка данных через аугментацию для миноритарных классов"""
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    
    # Если данные достаточно сбалансированы, возвращаем как есть
    if max_count <= min(label_counts.values()) * 2:
        return image_paths, labels
    
    print("🔄 Балансировка данных через аугментацию...")
    
    augmented_images = list(image_paths)
    augmented_labels = list(labels)
    
    for class_label, count in label_counts.items():
        if count < max_count:
            # Находим индексы изображений этого класса
            class_indices = [i for i, label in enumerate(labels) if label == class_label]
            needed_samples = min(max_count - count, max_samples_per_class - count)
            
            # Добавляем существующие изображения несколько раз (упрощенная аугментация)
            for i in range(needed_samples):
                original_idx = class_indices[i % len(class_indices)]
                augmented_images.append(image_paths[original_idx])
                augmented_labels.append(class_label)
    
    print(f"После балансировки: {len(augmented_images)} изображений")
    return augmented_images, augmented_labels

def prepare_and_train_model(image_paths, labels, species_names, model_type="plant"):
    """Подготовка данных и обучение модели"""
    if len(image_paths) == 0:
        print(f"Нет данных для обучения модели {model_type}")
        return None, None, []
    
    print(f"Данные загружены: {len(image_paths)} изображений, {len(set(labels))} классов")
    
    # Приводим метки к диапазону 0..C-1
    unique_labels_sorted = sorted(set(labels))
    original_to_compact = {orig: idx for idx, orig in enumerate(unique_labels_sorted)}
    labels_mapped_all = [original_to_compact[l] for l in labels]

    # Формируем список имён классов
    class_names = []
    for orig_label in unique_labels_sorted:
        for path_i, lab in enumerate(labels):
            if lab == orig_label:
                class_names.append(species_names[path_i])
                break

    # Проверяем баланс классов
    check_class_balance(labels_mapped_all, class_names, f"{model_type} (исходные)")
    
    # Балансируем данные
    balanced_paths, balanced_labels = apply_data_augmentation_balance(image_paths, labels_mapped_all)
    
    # Разделение данных с проверкой возможности стратификации
    if len(balanced_paths) <= 3:
        print("Очень мало данных! Используем все для обучения")
        train_paths, train_labels = balanced_paths, balanced_labels
        val_paths, val_labels = balanced_paths, balanced_labels
    elif can_use_stratified_split(balanced_labels):
        print("Используем стратифицированное разделение")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            balanced_paths, balanced_labels, test_size=0.2, random_state=42, stratify=balanced_labels
        )
    else:
        print("Используем случайное разделение (стратификация невозможна)")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            balanced_paths, balanced_labels, test_size=0.2, random_state=42, stratify=None
        )
    
    print(f"Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    check_class_balance(train_labels, class_names, f"{model_type} (тренировочные)")
    
    # Создаем трансформации и даталоадеры
    train_transform, val_transform = get_enhanced_transforms()
    
    train_dataset = TreeDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = TreeDataset(val_paths, val_labels, transform=val_transform)
    
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    num_classes = len(unique_labels_sorted)
    model = create_improved_model(num_classes)
    
    print(f"Архитектура модели {model_type}: {num_classes} классов")
    print(f"Размер батча: {batch_size}")
    
    # Обучение
    model, history = train_single_model(model, train_loader, val_loader, 
                                      NUM_EPOCHS, num_classes, model_type)

    return model, history, class_names

def train_separated_models(porody_folder_path):
    """Обучение отдельных моделей для деревьев и кустов"""
    print("ЗАГРУЗКА ДАННЫХ С РАЗДЕЛЕНИЕМ НА ДЕРЕВЬЯ И КУСТЫ")
    print("=" * 60)
    
    (tree_images, tree_labels, tree_species,
     bush_images, bush_labels, bush_species) = load_tree_species_data_separated(porody_folder_path)
    
    # Обучаем модель для деревьев
    tree_model, tree_history, tree_class_names = None, None, []
    if len(tree_images) > 0:
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ДЕРЕВЬЕВ")
        print("="*50)
        tree_model, tree_history, tree_class_names = prepare_and_train_model(
            tree_images, tree_labels, tree_species, "деревья"
        )
    
    # Обучаем модель для кустов  
    bush_model, bush_history, bush_class_names = None, None, []
    if len(bush_images) > 0:
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ МОДЕЛИ ДЛЯ КУСТОВ")
        print("="*50)
        bush_model, bush_history, bush_class_names = prepare_and_train_model(
            bush_images, bush_labels, bush_species, "кусты"
        )
    
    return tree_model, bush_model, tree_class_names, bush_class_names

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
        tree_model = create_improved_model(num_classes)
        tree_model.load_state_dict(checkpoint['model_state_dict'])
        tree_class_names = checkpoint['class_names']
        print("Модель деревьев загружена")
    except Exception as e:
        print(f"Модель деревьев не найдена или ошибка загрузки: {e}")
    
    try:
        # Загрузка модели кустов
        checkpoint = torch.load('model_bushes.pth', map_location=DEVICE)
        num_classes = len(checkpoint['class_names'])
        bush_model = create_improved_model(num_classes)
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
                
                print(f"🔧 Растение {i+1} - Дефекты: {defects_name} (уверенность: {defects_confidence:.2%})")
                
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
    print("Изображение показано. Нажмите любую клавишу в окне изображения...")
    
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
        defects_model = create_improved_model(num_classes)
        defects_model.load_state_dict(checkpoint['model_state_dict'])
        defects_class_names = checkpoint['class_names']
        print("Модель дефектов загружена")
    except Exception as e:
        print(f"Модель дефектов не найдена или ошибка загрузки: {e}")
    
    return defects_model, defects_class_names

def train_defects_model_improved(characteristiki_folder_path):

    # Используем существующую функцию загрузки данных дефектов
    image_paths, labels, defect_descriptions = load_defects_data(characteristiki_folder_path)
    
    if len(image_paths) == 0:
        print("Не найдены изображения для обучения дефектов!")
        return None, None, [], [], []
    
    print(f"Данные дефектов загружены: {len(image_paths)} изображений, {len(set(labels))} классов")
    
    # Подготовка и обучение модели (аналогично породам)
    model, history, class_names = prepare_and_train_model(
        image_paths, labels, defect_descriptions, "дефекты"
    )
    
    return model, history, class_names, image_paths, labels

def load_defects_data(characteristiki_folder_path):
    """Загрузка данных для классификации характеристик/дефектов"""
    char_path = Path(characteristiki_folder_path)
    
    print(f"Загрузка данных дефектов из: {char_path.absolute()}")
    
    if not char_path.exists():
        print(f"Папка не существует: {char_path}")
        return [], [], []
    
    # Ищем CSV файл
    csv_path = char_path / "labels" / "labels.csv"
    if not csv_path.exists():
        csv_path = char_path / "labels.csv"
        if not csv_path.exists():
            print("CSV файл не найден")
            return [], [], []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"CSV загружен: {len(df)} записей")
        
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        return [], [], []
    
    # Папка с изображениями
    images_dir = char_path / "images"
    if not images_dir.exists():
        print("Папка 'images' не найдена")
        return [], [], []
    
    images = []
    labels = []
    defect_descriptions = []
    
    successful = 0
    for _, row in df.iterrows():
        try:
            filename = str(row['filename']).strip()
            img_path = images_dir / filename
            
            if img_path.exists():
                images.append(str(img_path))
                labels.append(int(row['defect_label']))
                defect_descriptions.append(str(row['defect_description']))
                successful += 1
            else:
                print(f"Изображение не найдено: {filename}")
                
        except Exception as e:
            print(f"Ошибка обработки строки: {e}")
            continue
    
    
    return images, labels, defect_descriptions

def main_improved():
    porody_path = "data/породы"
    char_path = "data/характеристики"  # Путь к данным дефектов
    
    try:
        detection_model = YOLO("best.pt")
    except Exception as e:
        print(f"{e}")
        return
    


    tree_model, bush_model, tree_class_names, bush_class_names = load_separated_models()
    
    if tree_model is None and bush_model is None:

        tree_model, bush_model, tree_class_names, bush_class_names = train_separated_models(porody_path)
        
        # Сохранение моделей пород
        if tree_model is not None:
            torch.save({
                'model_state_dict': tree_model.state_dict(),
                'class_names': tree_class_names,
                'plant_type': 'tree'
            }, 'model_trees.pth')
            print("Модель деревьев сохранена")
        
        if bush_model is not None:
            torch.save({
                'model_state_dict': bush_model.state_dict(),
                'class_names': bush_class_names, 
                'plant_type': 'bush'
            }, 'model_bushes.pth')
            print("Модель кустов сохранена")
    
    # ЗАГРУЗКА ИЛИ ОБУЧЕНИЕ МОДЕЛИ ДЕФЕКТОВ
    print("\nЗАГРУЗКА МОДЕЛИ КЛАССИФИКАЦИИ ДЕФЕКТОВ...")
    defects_model, defects_class_names = load_defects_model()
    
    if defects_model is None:
        print("ОБУЧЕНИЕ НОВОЙ МОДЕЛИ ДЕФЕКТОВ...")
        defects_model, defects_history, defects_class_names, _, _ = train_defects_model_improved(char_path)
        
        if defects_model is not None:
            torch.save({
                'model_state_dict': defects_model.state_dict(),
                'class_names': defects_class_names,
                'model_type': 'defects'
            }, 'model_defects.pth')
            print("Модель дефектов сохранена")

if __name__ == "__main__":
    main_improved()