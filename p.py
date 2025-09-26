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

# Отключаем warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 2
NUM_EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Используемое устройство: {DEVICE}")

class TreeDataset(Dataset):
    """Кастомный датасет для изображений деревьев с надежной обработкой ошибок"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Пытаемся загрузить изображение разными способами
        image = self.load_image_safe(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def load_image_safe(self, img_path):
        """Безопасная загрузка изображения с несколькими fallback'ами"""
        try:
            # Способ 1: Используем OpenCV
            image = cv2.imread(str(img_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                return image
        except:
            pass
        
        try:
            # Способ 2: Используем PIL напрямую
            image = Image.open(img_path).convert('RGB')
            return image
        except:
            pass
        
        # Способ 3: Создаем черное изображение как fallback
        print(f"⚠️ Не удалось загрузить {img_path}, создаем черное изображение")
        return Image.new('RGB', IMG_SIZE, color='black')

def load_tree_species_data(porody_folder_path):
    """Загрузка данных для классификации пород деревьев"""
    porody_path = Path(porody_folder_path)
    
    print(f"🔍 Загрузка данных из: {porody_path.absolute()}")
    
    if not porody_path.exists():
        print(f"❌ Папка не существует: {porody_path}")
        return [], [], []
    
    # Ищем CSV файл
    csv_path = porody_path / "labels" / "labels.csv"
    if not csv_path.exists():
        csv_path = porody_path / "labels.csv"
        if not csv_path.exists():
            print("❌ CSV файл не найден")
            return [], [], []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ CSV загружен: {len(df)} записей")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")
        return [], [], []
    
    # Папка с изображениями
    images_dir = porody_path / "images"
    if not images_dir.exists():
        print("❌ Папка 'images' не найдена")
        return [], [], []
    
    images = []
    labels = []
    species_names = []
    
    successful = 0
    for _, row in df.iterrows():
        try:
            filename = str(row['filename']).strip()
            img_path = images_dir / filename
            
            if img_path.exists():
                images.append(str(img_path))
                labels.append(int(row['species_label']))
                species_names.append(str(row['species_name']))
                successful += 1
            else:
                print(f"⚠️ Изображение не найдено: {filename}")
                
        except Exception as e:
            print(f"❌ Ошибка обработки строки: {e}")
            continue
    
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений")
    print(f"🎯 Количество классов: {len(set(labels))}")
    
    return images, labels, species_names

def load_defects_data(characteristiki_folder_path):
    """Загрузка данных для классификации характеристик/дефектов"""
    char_path = Path(characteristiki_folder_path)
    
    print(f"🔍 Загрузка данных из: {char_path.absolute()}")
    
    if not char_path.exists():
        print(f"❌ Папка не существует: {char_path}")
        return [], [], []
    
    # Ищем CSV файл
    csv_path = char_path / "labels" / "labels.csv"
    if not csv_path.exists():
        csv_path = char_path / "labels.csv"
        if not csv_path.exists():
            print("❌ CSV файл не найден")
            return [], [], []
    
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ CSV загружен: {len(df)} записей")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки CSV: {e}")
        return [], [], []
    
    # Папка с изображениями
    images_dir = char_path / "images"
    if not images_dir.exists():
        print("❌ Папка 'images' не найдена")
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
                print(f"⚠️ Изображение не найдено: {filename}")
                
        except Exception as e:
            print(f"❌ Ошибка обработки строки: {e}")
            continue
    
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений")
    print(f"🎯 Количество классов: {len(set(labels))}")
    
    return images, labels, defect_descriptions

def get_transforms():
    """Возвращает трансформации для обучения и валидации"""
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_simple_model(num_classes):
    """Создание упрощенной модели на PyTorch"""
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    return SimpleCNN(num_classes)

def train_model_safe(model, train_loader, val_loader, num_epochs, num_classes, model_type="породы"):
    """Безопасное обучение модели с обработкой ошибок"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_losses = []
    val_accuracies = []
    
    print(f"🎯 НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ {model_type.upper()}...")
    
    for epoch in range(num_epochs):
        try:
            # Обучение
            model.train()
            running_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    
                    if batch_idx % 5 == 0:
                        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
                        
                except Exception as e:
                    print(f"❌ Ошибка в батче {batch_idx}: {e}")
                    continue
            
            # Валидация
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    try:
                        images = images.to(DEVICE)
                        labels = labels.to(DEVICE)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    except Exception as e:
                        print(f"❌ Ошибка при валидации: {e}")
                        continue
            
            accuracy = 100 * correct / total if total > 0 else 0
            avg_loss = running_loss / batch_count if batch_count > 0 else 0
            
            train_losses.append(avg_loss)
            val_accuracies.append(accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
        except Exception as e:
            print(f"❌ Критическая ошибка в эпохе {epoch+1}: {e}")
            continue
    
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, {'train_loss': train_losses, 'val_accuracy': val_accuracies}

def can_use_stratified_split(labels, test_size=0.2):
    """Проверяет, можно ли использовать стратифицированное разделение"""
    from collections import Counter
    label_counts = Counter(labels)
    min_samples_per_class = min(label_counts.values())
    
    # Для стратификации нужно минимум 2 образца в каждом классе
    # и хотя бы 1 образец каждого класса должен остаться в тестовой выборке
    return min_samples_per_class >= 2 and all(count >= int(1/test_size) + 1 for count in label_counts.values())

def train_tree_species_model_simple(porody_folder_path):
    """Упрощенное обучение модели для классификации пород деревьев"""
    
    print("🌳 ЗАГРУЗКА ДАННЫХ ПОРОД ДЕРЕВЬЕВ")
    print("=" * 50)
    
    image_paths, labels, species_names = load_tree_species_data(porody_folder_path)
    
    if len(image_paths) == 0:
        print("❌ Не найдены изображения для обучения!")
        return None, None, [], [], []
    
    print(f"✅ Данные загружены: {len(image_paths)} изображений, {len(set(labels))} классов")
    
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

    # Разделение данных с проверкой возможности стратификации
    if len(image_paths) <= 3:
        print("⚠️ Очень мало данных! Используем все для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths, labels_mapped_all
    elif can_use_stratified_split(labels_mapped_all):
        print("📊 Используем стратифицированное разделение")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=labels_mapped_all
        )
    else:
        print("📊 Используем случайное разделение (стратификация невозможна)")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=None
        )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
    # Создаем трансформации и даталоадеры
    train_transform, val_transform = get_transforms()
    
    train_dataset = TreeDataset(train_paths, train_labels, train_transform)
    val_dataset = TreeDataset(val_paths, val_labels, val_transform)
    
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)
    
    # Обучение
    model, history = train_model_safe(model, train_loader, val_loader, 
                                    min(NUM_EPOCHS, 5), num_classes, "породы")

    return model, history, class_names, image_paths, labels_mapped_all

def train_defects_model_simple(characteristiki_folder_path):
    """Упрощенное обучение модели для классификации характеристик/дефектов"""
    
    print("🔍 ЗАГРУЗКА ДАННЫХ ХАРАКТЕРИСТИК")
    print("=" * 50)
    
    image_paths, labels, defect_descriptions = load_defects_data(characteristiki_folder_path)
    
    if len(image_paths) == 0:
        print("❌ Не найдены изображения для обучения!")
        return None, None, [], [], []
    
    print(f"✅ Данные загружены: {len(image_paths)} изображений, {len(set(labels))} классов")
    
    # Приводим метки к диапазону 0..C-1
    unique_labels_sorted = sorted(set(labels))
    original_to_compact = {orig: idx for idx, orig in enumerate(unique_labels_sorted)}
    labels_mapped_all = [original_to_compact[l] for l in labels]

    # Список имён классов
    class_names = []
    for orig_label in unique_labels_sorted:
        for path_i, lab in enumerate(labels):
            if lab == orig_label:
                class_names.append(defect_descriptions[path_i])
                break

    # Разделение данных с проверкой возможности стратификации
    if len(image_paths) <= 3:
        print("⚠️ Очень мало данных! Используем все для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths, labels_mapped_all
    elif can_use_stratified_split(labels_mapped_all):
        print("📊 Используем стратифицированное разделение")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=labels_mapped_all
        )
    else:
        print("📊 Используем случайное разделение (стратификация невозможна)")
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=None
        )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
    # Создаем трансформации и даталоадеры
    train_transform, val_transform = get_transforms()
    
    train_dataset = TreeDataset(train_paths, train_labels, train_transform)
    val_dataset = TreeDataset(val_paths, val_labels, val_transform)
    
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)
    
    # Обучение
    model, history = train_model_safe(model, train_loader, val_loader, 
                                    min(NUM_EPOCHS, 5), num_classes, "характеристики")

    return model, history, class_names, image_paths, labels_mapped_all

def simple_test_model(model, test_image_path, class_names, model_type="породы"):
    """Упрощенное тестирование модели"""
    try:
        # Трансформации для тестирования
        transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Загрузка изображения с использованием OpenCV как fallback
        try:
            image = Image.open(test_image_path).convert('RGB')
        except:
            image_cv = cv2.imread(test_image_path)
            if image_cv is not None:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_cv)
            else:
                print(f"❌ Не удалось загрузить тестовое изображение: {test_image_path}")
                return None, 0
        
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Предсказание
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        
        # Получаем название класса
        if predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Класс {predicted_class}"
        
        print(f"\n🔍 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({model_type}):")
        print(f"📸 Изображение: {Path(test_image_path).name}")
        print(f"🎯 Предсказание: {class_name}")
        print(f"📊 Уверенность: {confidence:.2%}")
        
        return class_name, confidence
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании {test_image_path}: {e}")
        return None, 0

def evaluate_model_on_all_images(model, image_paths, true_labels, class_names, model_type="породы"):
    """Оценка модели на всех изображениях"""
    print(f"\n📊 ОЦЕНКА МОДЕЛИ НА ВСЕХ ИЗОБРАЖЕНИЯХ ({model_type}):")
    print("=" * 60)
    
    if len(image_paths) == 0:
        print("❌ Нет данных для оценки")
        return 0
    
    # Трансформации для оценки
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    correct_predictions = 0
    total_images = 0
    
    print(f"🔢 Всего изображений для оценки: {len(image_paths)}")
    print("-" * 60)
    
    for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
        try:
            # Загрузка изображения с fallback
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image_cv = cv2.imread(img_path)
                if image_cv is not None:
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image_cv)
                else:
                    continue
            
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Предсказание
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted_class = torch.max(outputs.data, 1)
            
            predicted_class = predicted_class.item()
            
            # Проверка правильности предсказания
            is_correct = (predicted_class == true_label)
            if is_correct:
                correct_predictions += 1
            
            total_images += 1
            
            # Получаем названия классов
            true_class_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
            
            # Вывод результата с эмодзи
            status = "✅" if is_correct else "❌"
            print(f"{status} {i+1:2d}/{len(image_paths)}: {Path(img_path).name:15} | "
                  f"Истина: {true_class_name:15} | Предсказание: {pred_class_name:15}")
            
        except Exception as e:
            print(f"❌ Ошибка при оценке {img_path}: {e}")
            continue
    
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print("-" * 60)
    print(f"🎯 ИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.2%} ({correct_predictions}/{total_images})")
    
    return accuracy

# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    print("🌲 ЗАПУСК УПРОЩЕННОЙ СИСТЕМЫ КЛАССИФИКАЦИИ ДЕРЕВЬЕВ (PyTorch)")
    print("=" * 60)
    
    porody_path = "data/породы"
    char_path = "data/характеристики"
    
    try:
        # Сначала проверяем существование папок
        if not Path(porody_path).exists():
            print(f"❌ Папка с породами не найдена: {porody_path}")
            porody_path = input("Введите правильный путь к папке с породами: ")
        
        if not Path(char_path).exists():
            print(f"❌ Папка с характеристиками не найдена: {char_path}")
            char_path = input("Введите правильный путь к папке с характеристиками: ")
        
        # Обучаем модель для пород
        porody_model, porody_history, species_names, porody_images, porody_labels = train_tree_species_model_simple(porody_path)
        
        print("\n" + "=" * 60)
        
        # Обучаем модель для характеристик (если данные доступны)
        defects_model, defects_history, defect_descriptions, defects_images, defects_labels = train_defects_model_simple(char_path)
        
        print("\n" + "=" * 60)
        print("🧪 ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
        print("=" * 60)
        
        # Тестирование модели пород
        if porody_model is not None and len(species_names) > 0 and len(porody_images) > 0:
            print("🌳 ТЕСТИРОВАНИЕ МОДЕЛИ ПОРОД:")
            test_image = porody_images[0]
            simple_test_model(porody_model, test_image, species_names, "породы")
            
            # Оценка на всех изображениях
            porody_accuracy = evaluate_model_on_all_images(
                porody_model, porody_images, porody_labels, species_names, "породы"
            )
            
            # Сохранение модели
            try:
                torch.save({
                    'model_state_dict': porody_model.state_dict(),
                    'class_names': species_names,
                    'accuracy': porody_accuracy
                }, 'model_porody_simple.pth')
                print("✅ Модель пород сохранена как 'model_porody_simple.pth'")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить модель пород: {e}")
        
        # Тестирование модели характеристик
        if defects_model is not None and len(defect_descriptions) > 0 and len(defects_images) > 0:
            print("\n🔍 ТЕСТИРОВАНИЕ МОДЕЛИ ХАРАКТЕРИСТИК:")
            test_image = defects_images[0]
            simple_test_model(defects_model, test_image, defect_descriptions, "характеристики")
            
            # Оценка на всех изображениях
            defects_accuracy = evaluate_model_on_all_images(
                defects_model, defects_images, defects_labels, defect_descriptions, "характеристики"
            )
            
            # Сохранение модели
            try:
                torch.save({
                    'model_state_dict': defects_model.state_dict(),
                    'class_names': defect_descriptions,
                    'accuracy': defects_accuracy
                }, 'model_defects_simple.pth')
                print("✅ Модель характеристик сохранена как 'model_defects_simple.pth'")
            except Exception as e:
                print(f"⚠️ Не удалось сохранить модель характеристик: {e}")
        else:
            print("\n⚠️ Модель характеристик не была обучена из-за проблем с данными")
        
        print("\n🎉 ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
        
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        print("💡 Попробуйте перезапустить программу")
