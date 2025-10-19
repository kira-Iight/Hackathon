import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
import os
from PIL import Image
import cv2
from collections import Counter

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
        
        print(f"Не удалось загрузить {img_path}, создаем черное изображение")
        return Image.new('RGB', IMG_SIZE, color='black')

def load_tree_species_data_separated(porody_folder_path):
    """Загрузка данных с разделением на деревья и кусты"""
    porody_path = Path(porody_folder_path)
    
    print(f"Загрузка данных из: {porody_path.absolute()}")
    
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
                print(f"Изображение не найдено: {filename}")
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


def get_class_weights(labels, num_classes):
    class_counts = Counter(labels)
    print(f"Распределение классов: {dict(class_counts)}")
    
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
        print(f"Загружены веса лучшей модели с точностью {best_accuracy:.2f}%")
    
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
    
    print("Балансировка данных через аугментацию...")
    
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
    model = ImprovedCNN(num_classes)
    
    model, history = train_single_model(model, train_loader, val_loader, 
                                      NUM_EPOCHS, num_classes, model_type)

    return model, history, class_names

def train_separated_models(porody_folder_path):
    
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