import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import os

# Отключаем warnings и настраиваем TensorFlow для macOS
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Отключаем многопоточность
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 2
NUM_EPOCHS = 5  # Еще меньше эпох для стабильности

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
        # Пробуем разные разделители и обработку ошибок
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

def load_and_preprocess_image(img_path, img_size=IMG_SIZE):
    """Загрузка и предобработка одного изображения"""
    try:
        img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"❌ Ошибка загрузки изображения {img_path}: {e}")
        return None

def create_simple_dataset(image_paths, labels, batch_size=2, img_size=IMG_SIZE):
    """Создание простого датасета без многопоточности"""
    images = []
    valid_labels = []
    
    # Загружаем все изображения заранее
    for i, img_path in enumerate(image_paths):
        img_array = load_and_preprocess_image(img_path, img_size)
        if img_array is not None:
            images.append(img_array)
            valid_labels.append(labels[i])
    
    images = np.array(images)
    valid_labels = np.array(valid_labels)
    
    # Создаем tf.data.Dataset без многопоточности
    dataset = tf.data.Dataset.from_tensor_slices((images, valid_labels))
    dataset = dataset.batch(batch_size)
    
    return dataset

def create_simple_model(num_classes):
    """Создание упрощенной модели"""
    # Простая модель без предобученных весов
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
    
    model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')])
    
    return model

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

    # Разделение данных
    if len(image_paths) <= 3:
        print("⚠️ Очень мало данных! Используем все для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths, labels_mapped_all
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42
        )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
    # Создаем датасеты
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
    train_dataset = create_simple_dataset(train_paths, train_labels, batch_size)
    val_dataset = create_simple_dataset(val_paths, val_labels, batch_size)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)

    # Компиляция модели
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ ПОРОД...")

    # Обучение
    history = model.fit(
        train_dataset,
        epochs=min(NUM_EPOCHS, 5),
        validation_data=val_dataset,
        verbose=1
    )

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
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

    # Разделение данных
    if len(image_paths) <= 3:
        print("⚠️ Очень мало данных! Используем все для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths, labels_mapped_all
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels_mapped_all, test_size=0.2, random_state=42
        )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
    # Создаем датасеты
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
    train_dataset = create_simple_dataset(train_paths, train_labels, batch_size)
    val_dataset = create_simple_dataset(val_paths, val_labels, batch_size)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ ХАРАКТЕРИСТИК...")

    history = model.fit(
        train_dataset,
        epochs=min(NUM_EPOCHS, 5),
        validation_data=val_dataset,
        verbose=1
    )

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, history, class_names, image_paths, labels_mapped_all

def simple_test_model(model, test_image_path, class_names, model_type="породы"):
    """Упрощенное тестирование модели"""
    try:
        # Загрузка и подготовка изображения
        img_array = load_and_preprocess_image(test_image_path)
        if img_array is None:
            return None, 0
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # Предсказание
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
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
        print(f"❌ Ошибка при тестировании: {e}")
        return None, 0

def evaluate_model_on_all_images(model, image_paths, true_labels, class_names, model_type="породы"):
    """Оценка модели на всех изображениях с выводом ✅/❌ для каждого предсказания"""
    print(f"\n📊 ОЦЕНКА МОДЕЛИ НА ВСЕХ ИЗОБРАЖЕНИЯХ ({model_type}):")
    print("=" * 60)
    
    if len(image_paths) == 0:
        print("❌ Нет данных для оценки")
        return 0
    
    correct_predictions = 0
    total_images = len(image_paths)
    
    print(f"🔢 Всего изображений для оценки: {total_images}")
    print("-" * 60)
    
    for i, (img_path, true_label) in enumerate(zip(image_paths, true_labels)):
        try:
            # Загрузка и подготовка изображения
            img_array = load_and_preprocess_image(img_path)
            if img_array is None:
                continue
                
            img_array = np.expand_dims(img_array, axis=0)
            
            # Предсказание
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Проверка правильности предсказания
            is_correct = (predicted_class == true_label)
            if is_correct:
                correct_predictions += 1
            
            # Получаем названия классов
            true_class_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_class_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
            
            # Вывод результата с эмодзи
            status = "✅" if is_correct else "❌"
            print(f"{status} {i+1:2d}/{total_images}: {Path(img_path).name:15} | "
                  f"Истина: {true_class_name:20} | Предсказание: {pred_class_name:20} | "
                  f"Уверенность: {confidence:.2%}")
            
        except Exception as e:
            print(f"❌ Ошибка при оценке {img_path}: {e}")
            continue
    
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print("-" * 60)
    print(f"🎯 ИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.2%} ({correct_predictions}/{total_images})")
    
    return accuracy

# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    print("🌲 ЗАПУСК УПРОЩЕННОЙ СИСТЕМЫ КЛАССИФИКАЦИИ ДЕРЕВЬЕВ")
    print("=" * 60)
    
    # Дополнительные настройки для стабильности
    tf.config.set_soft_device_placement(True)
    
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
                porody_model.save('model_porody_simple.h5')
                print("✅ Модель пород сохранена как 'model_porody_simple.h5'")
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
                defects_model.save('model_defects_simple.h5')
                print("✅ Модель характеристик сохранена как 'model_defects_simple.h5'")
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





