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
<<<<<<< HEAD
import os

# Отключаем warnings и настраиваем TensorFlow для macOS
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Отключаем многопоточность
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
=======
warnings.filterwarnings('ignore')
>>>>>>> e4a4378eac530946868097685580eb82d315742b

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 2
<<<<<<< HEAD
NUM_EPOCHS = 5  # Еще меньше эпох для стабильности
=======
NUM_EPOCHS = 50
# Количество эпох для первичного обучения перед дообучением (fine-tuning)
WARMUP_EPOCHS = 8
FINE_TUNE_LR = 5e-5

def create_proper_csv_files():
    """Создает правильные CSV файлы на основе имеющихся изображений"""
    
    # Папки с данными
    porody_path = Path("data/породы")
    char_path = Path("data/характеристики")
    
    # Создаем CSV для пород
    porody_images_dir = porody_path / "images"
    if porody_images_dir.exists():
        images = list(porody_images_dir.glob("*.jpg")) + list(porody_images_dir.glob("*.png")) + list(porody_images_dir.glob("*.jpeg"))
        print(f"📁 Найдено {len(images)} изображений пород")
        
        # Создаем mapping на основе имен файлов
        porody_data = []
        for i, img_path in enumerate(images):
            img_name = img_path.stem
            
            # Простое сопоставление по номеру файла
            species_name = f"Порода_{img_name}"
            if img_name == "1":
                species_name = "Клен остролистный"
            elif img_name == "2":
                species_name = "Лиственница"
            elif img_name == "3":
                species_name = "Туя"
            elif img_name == "4":
                species_name = "Рябина"
            elif img_name == "5":
                species_name = "Сосна"
            elif img_name == "6":
                species_name = "Можжевельник"
            elif img_name == "7":
                species_name = "Береза"
            elif img_name == "8":
                species_name = "Каштан"
            elif img_name == "9":
                species_name = "Ива"
            elif img_name == "10":
                species_name = "Осина"
            
            porody_data.append({
                'filename': img_path.name,
                'species_label': i,  # Уникальная метка для каждого изображения
                'species_name': species_name
            })
        
        porody_df = pd.DataFrame(porody_data)
        porody_df.to_csv(porody_path / "proper_labels.csv", index=False, encoding='utf-8')
        print(f"✅ Создан CSV для пород: {len(porody_data)} записей")
    
    # Создаем CSV для характеристик
    char_images_dir = char_path / "images"
    if char_images_dir.exists():
        images = list(char_images_dir.glob("*.jpg")) + list(char_images_dir.glob("*.png")) + list(char_images_dir.glob("*.jpeg"))
        print(f"📁 Найдено {len(images)} изображений характеристик")
        
        char_data = []
        for i, img_path in enumerate(images):
            img_name = img_path.stem
            
            # Простое сопоставление для характеристик
            defect_description = f"Дефект_{img_name}"
            if img_name == "1":
                defect_description = "Комлевая гниль"
            elif img_name == "2":
                defect_description = "Сухобочина"
            elif img_name == "3":
                defect_description = "Стволовая гниль"
            elif img_name == "4":
                defect_description = "Механические повреждения"
            elif img_name == "5":
                defect_description = "Плодовые тела"
            elif img_name == "6":
                defect_description = "Отслоение коры"
            elif img_name == "7":
                defect_description = "Сухие ветви"
            elif img_name == "8":
                defect_description = "Сухостой"
            elif img_name == "9":
                defect_description = "Дупло"
            elif img_name == "10":
                defect_description = "Повреждения вредителями"
            
            char_data.append({
                'filename': img_path.name,
                'defect_label': i,  # Уникальная метка для каждого изображения
                'defect_description': defect_description
            })
        
        char_df = pd.DataFrame(char_data)
        char_df.to_csv(char_path / "proper_labels.csv", index=False, encoding='utf-8')
        print(f"✅ Создан CSV для характеристик: {len(char_data)} записей")
>>>>>>> e4a4378eac530946868097685580eb82d315742b

def load_tree_species_data(porody_folder_path):
    """Загрузка данных для классификации пород деревьев"""
    porody_path = Path(porody_folder_path)
    
    print(f"🔍 Загрузка данных из: {porody_path.absolute()}")
    
    if not porody_path.exists():
        print(f"❌ Папка не существует: {porody_path}")
        return [], [], []
    
<<<<<<< HEAD
    # Ищем CSV файл
    csv_path = porody_path / "labels" / "labels.csv"
    if not csv_path.exists():
        csv_path = porody_path / "labels.csv"
        if not csv_path.exists():
            print("❌ CSV файл не найден")
=======
    # Ищем правильный CSV файл
    csv_path = porody_path / "proper_labels.csv"
    if not csv_path.exists():
        print("❌ proper_labels.csv не найден, создаем...")
        create_proper_csv_files()
        
        if not csv_path.exists():
            print("❌ Не удалось создать CSV файл")
>>>>>>> e4a4378eac530946868097685580eb82d315742b
            return [], [], []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ CSV загружен: {len(df)} записей")
<<<<<<< HEAD
=======
        print("Структура CSV:")
        print(df.head())
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        
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
    
<<<<<<< HEAD
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений")
=======
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений пород")
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    print(f"🎯 Количество классов: {len(set(labels))}")
    
    return images, labels, species_names

def load_defects_data(characteristiki_folder_path):
    """Загрузка данных для классификации характеристик/дефектов"""
    char_path = Path(characteristiki_folder_path)
    
    print(f"🔍 Загрузка данных из: {char_path.absolute()}")
    
    if not char_path.exists():
        print(f"❌ Папка не существует: {char_path}")
        return [], [], []
    
<<<<<<< HEAD
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
=======
    # Ищем правильный CSV файл
    csv_path = char_path / "proper_labels.csv"
    if not csv_path.exists():
        print("❌ proper_labels.csv не найден, создаем...")
        create_proper_csv_files()
        
        if not csv_path.exists():
            print("❌ Не удалось создать CSV файл")
            return [], [], []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ CSV загружен: {len(df)} записей")
        print("Структура CSV:")
        print(df.head())
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        
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
    
<<<<<<< HEAD
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений")
=======
    print(f"✅ Успешно загружено {successful}/{len(df)} изображений характеристик")
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    print(f"🎯 Количество классов: {len(set(labels))}")
    
    return images, labels, defect_descriptions

<<<<<<< HEAD
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
=======
class AdvancedDataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=2, img_size=(224, 224), 
                 shuffle=True, augmentation=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()
        
        if self.augmentation:
            self.datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            self.datagen = keras.preprocessing.image.ImageDataGenerator()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        X = []
        y = []
        
        for i, img_path in enumerate(batch_paths):
            try:
                img = keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                
                X.append(img_array)
                y.append(batch_labels[i])
            except Exception as e:
                print(f"❌ Ошибка загрузки изображения {img_path}: {e}")
                continue
        
        if not X:
            return np.zeros((1, self.img_size[0], self.img_size[1], 3)), np.zeros(1)
        
        X = np.array(X)
        y = np.array(y)
        
        if self.augmentation:
            X = self.datagen.flow(X, batch_size=len(X), shuffle=False).next()
        
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

def create_regularized_model(num_classes, model_name='tree_species'):
    """Создание модели с регуляризацией"""
    
    # Пытаемся загрузить веса ImageNet, если нет доступа к интернету — используем случайную инициализацию
    try:
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    except Exception as e:
        print(f"⚠️ Не удалось загрузить веса ImageNet для EfficientNetB0: {e}\nИспользуем случайную инициализацию весов.")
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
    base_model.trainable = False
    
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def _get_base_model_from_sequential(model: keras.Model):
    """Возвращает вложенную базовую модель EfficientNet внутри Sequential."""
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.startswith('efficientnet'):
            return layer
    return None

def _compute_class_weights(labels):
    """Вычисляет веса классов для несбалансированных данных."""
    if not labels:
        return None
    
    # Убедимся, что метки начинаются с 0 и идут последовательно
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)
    
    # Если метки уже в правильном диапазоне 0..num_classes-1
    if min(labels) == 0 and max(labels) == num_classes - 1:
        unique, counts = np.unique(labels, return_counts=True)
    else:
        # Переиндексируем метки
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        remapped_labels = [label_mapping[label] for label in labels]
        unique, counts = np.unique(remapped_labels, return_counts=True)
    
    total = np.sum(counts)
    class_weights = {}
    
    for cls, cnt in zip(unique, counts):
        class_weights[int(cls)] = float(total / (num_classes * cnt))
    
    print(f"📊 Веса классов: {class_weights}")
    return class_weights

def _enable_fine_tuning(model: keras.Model, trainable_ratio: float = 0.2):
    """Размораживает верхнюю часть EfficientNet для дообучения."""
    base_model = _get_base_model_from_sequential(model)
    if base_model is None:
        print("❌ Не удалось найти базовую модель EfficientNet")
        return False
    
    # Размораживаем только верхние слои эффективнета
    total_layers = len(base_model.layers)
    trainable_from = int(total_layers * (1.0 - trainable_ratio))
    
    print(f"🛠️ Размораживаем {total_layers - trainable_from} из {total_layers} слоев")
    
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= trainable_from)
        if layer.trainable:
            print(f"   ✅ Слой {i}: {layer.name} - разморожен")
    
    return True

def train_tree_species_model(porody_folder_path):
    """Обучение модели для классификации пород деревьев"""
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    
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

<<<<<<< HEAD
    # Формируем список имён классов
    class_names = []
    for orig_label in unique_labels_sorted:
=======
    # Формируем список имён классов в порядке compact-меток
    class_names = []
    for orig_label in unique_labels_sorted:
        # Берём первое вхождение имени для данного оригинального лейбла
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        for path_i, lab in enumerate(labels):
            if lab == orig_label:
                class_names.append(species_names[path_i])
                break

<<<<<<< HEAD
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
=======
    # Для малого количества данных используем все для обучения
    if len(image_paths) <= 5:
        print("⚠️ Мало данных! Используем все данные для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths[:1], labels_mapped_all[:1]  # Одно изображение для валидации
    else:
        # Для большего количества данных используем обычное разделение
        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=labels_mapped_all
            )
        except:
            # Если не получается разделить с stratify, пробуем без него
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=None
            )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
    # Убедимся, что batch_size не больше количества данных
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
<<<<<<< HEAD
    train_dataset = create_simple_dataset(train_paths, train_labels, batch_size)
    val_dataset = create_simple_dataset(val_paths, val_labels, batch_size)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)

    # Компиляция модели
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
=======
    train_gen = AdvancedDataGenerator(train_paths, train_labels, 
                                    batch_size=batch_size, 
                                    img_size=IMG_SIZE, augmentation=True)
    
    val_gen = AdvancedDataGenerator(val_paths, val_labels, 
                                  batch_size=min(BATCH_SIZE, len(val_paths)), 
                                  img_size=IMG_SIZE, augmentation=False, shuffle=False)
    
    num_classes = len(unique_labels_sorted)
    model = create_regularized_model(num_classes, 'tree_species')

    # Убедимся, что веса классов корректны
    class_weights = _compute_class_weights(train_labels)
    if class_weights:
        # Проверяем, что ключи соответствуют диапазону классов
        expected_keys = set(range(num_classes))
        actual_keys = set(class_weights.keys())
        if expected_keys != actual_keys:
            print(f"⚠️ Исправляем веса классов: ожидались {expected_keys}, получены {actual_keys}")
            # Создаем правильные веса
            correct_weights = {}
            for i in range(num_classes):
                if i in class_weights:
                    correct_weights[i] = class_weights[i]
                else:
                    correct_weights[i] = 1.0  # Значение по умолчанию
            class_weights = correct_weights

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Увеличим learning rate для warmup
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

<<<<<<< HEAD
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
=======
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath='best_tree_species.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6),
        checkpoint_cb,
    ]

    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ ПОРОД (WARMUP)...")

    epochs = min(NUM_EPOCHS, 30)  # Уменьшим общее количество эпох
    warmup_epochs = min(WARMUP_EPOCHS, max(5, epochs // 3))  # Увеличим warmup

    # Первый этап: обучение только классификатора
    history = model.fit(
        train_gen,
        epochs=warmup_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tuning
    print("🛠️ ДООбучение (fine-tuning) верхних слоёв EfficientNet...")
    fine_tune_enabled = _enable_fine_tuning(model, trainable_ratio=0.2)  # Уменьшим ratio
    
    if fine_tune_enabled:
        # Компилируем с меньшим learning rate для fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        remaining_epochs = max(5, epochs - warmup_epochs)  # Минимум 5 эпох для fine-tuning
        
        history_ft = model.fit(
            train_gen,
            epochs=remaining_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Объединяем историю
        for k in history_ft.history.keys():
            if k in history.history:
                history.history[k].extend(history_ft.history[k])

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, history, class_names, image_paths, labels_mapped_all

def train_defects_model(characteristiki_folder_path):
    """Обучение модели для классификации характеристик/дефектов"""
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    
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

<<<<<<< HEAD
    # Список имён классов
=======
    # Список имён классов в порядке compact-меток
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    class_names = []
    for orig_label in unique_labels_sorted:
        for path_i, lab in enumerate(labels):
            if lab == orig_label:
                class_names.append(defect_descriptions[path_i])
                break

<<<<<<< HEAD
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
=======
    # Для малого количества данных
    if len(image_paths) <= 5:
        print("⚠️ Мало данных! Используем все данные для обучения")
        train_paths, train_labels = image_paths, labels_mapped_all
        val_paths, val_labels = image_paths[:1], labels_mapped_all[:1]
    else:
        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=labels_mapped_all
            )
        except:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels_mapped_all, test_size=0.2, random_state=42, stratify=None
            )
    
    print(f"📊 Разделение: {len(train_paths)} тренировочных, {len(val_paths)} валидационных")
    
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    batch_size = min(BATCH_SIZE, len(train_paths))
    if batch_size == 0:
        batch_size = 1
    
<<<<<<< HEAD
    train_dataset = create_simple_dataset(train_paths, train_labels, batch_size)
    val_dataset = create_simple_dataset(val_paths, val_labels, batch_size)
    
    num_classes = len(unique_labels_sorted)
    model = create_simple_model(num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
=======
    train_gen = AdvancedDataGenerator(train_paths, train_labels, 
                                    batch_size=batch_size, 
                                    img_size=IMG_SIZE, augmentation=True)
    
    val_gen = AdvancedDataGenerator(val_paths, val_labels, 
                                  batch_size=min(BATCH_SIZE, len(val_paths)), 
                                  img_size=IMG_SIZE, augmentation=False, shuffle=False)
    
    num_classes = len(unique_labels_sorted)
    model = create_regularized_model(num_classes, 'defects')

    # Убедимся, что веса классов корректны
    class_weights = _compute_class_weights(train_labels)
    if class_weights:
        expected_keys = set(range(num_classes))
        actual_keys = set(class_weights.keys())
        if expected_keys != actual_keys:
            print(f"⚠️ Исправляем веса классов: ожидались {expected_keys}, получены {actual_keys}")
            correct_weights = {}
            for i in range(num_classes):
                if i in class_weights:
                    correct_weights[i] = class_weights[i]
                else:
                    correct_weights[i] = 1.0
            class_weights = correct_weights

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

<<<<<<< HEAD
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
=======
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath='best_defects.keras',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6),
        checkpoint_cb,
    ]

    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ МОДЕЛИ ХАРАКТЕРИСТИК (WARMUP)...")

    epochs = min(NUM_EPOCHS, 30)
    warmup_epochs = min(WARMUP_EPOCHS, max(5, epochs // 3))

    history = model.fit(
        train_gen,
        epochs=warmup_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tuning
    print("🛠️ ДООбучение (fine-tuning) верхних слоёв EfficientNet...")
    fine_tune_enabled = _enable_fine_tuning(model, trainable_ratio=0.2)
    
    if fine_tune_enabled:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        remaining_epochs = max(5, epochs - warmup_epochs)
        
        history_ft = model.fit(
            train_gen,
            epochs=remaining_epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        for k in history_ft.history.keys():
            if k in history.history:
                history.history[k].extend(history_ft.history[k])

    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    return model, history, class_names, image_paths, labels_mapped_all

def plot_training_history(history, title):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_model(model, test_image_path, class_names, model_type="породы"):
    """Тестирование модели на одном изображении"""
    try:
        # Загрузка и подготовка изображения
        img = keras.preprocessing.image.load_img(test_image_path, target_size=IMG_SIZE)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        
        # Предсказание
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Получаем название класса
        if predicted_class < len(class_names):
            class_name = class_names[predicted_class]
        else:
            class_name = f"Класс {predicted_class}"
        
<<<<<<< HEAD
=======
        # Топ-3 предсказания
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = []
        for idx in top3_indices:
            if idx < len(class_names):
                name = class_names[idx]
            else:
                name = f"Класс {idx}"
            top3_predictions.append((name, predictions[0][idx]))
        
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        print(f"\n🔍 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ({model_type}):")
        print(f"📸 Изображение: {Path(test_image_path).name}")
        print(f"🎯 Предсказание: {class_name}")
        print(f"📊 Уверенность: {confidence:.2%}")
<<<<<<< HEAD
=======
        print(f"🏆 Топ-3 предсказания:")
        for i, (name, conf) in enumerate(top3_predictions, 1):
            print(f"   {i}. {name}: {conf:.2%}")
        
        # Визуализация
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Тестовое изображение\nПредсказание: {class_name}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Бар-plot предсказаний
        classes_to_show = min(5, len(class_names))
        top_indices = np.argsort(predictions[0])[-classes_to_show:][::-1]
        top_probs = predictions[0][top_indices]
        top_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in top_indices]
        
        plt.barh(range(classes_to_show), top_probs)
        plt.yticks(range(classes_to_show), top_labels)
        plt.xlabel('Вероятность')
        plt.title('Топ-5 предсказаний')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
>>>>>>> e4a4378eac530946868097685580eb82d315742b
        
        return class_name, confidence
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return None, 0

<<<<<<< HEAD
def evaluate_model_on_all_images(model, image_paths, true_labels, class_names, model_type="породы"):
    """Оценка модели на всех изображениях с выводом ✅/❌ для каждого предсказания"""
    print(f"\n📊 ОЦЕНКА МОДЕЛИ НА ВСЕХ ИЗОБРАЖЕНИЯХ ({model_type}):")
    print("=" * 60)
    
=======
def evaluate_model(model, image_paths, labels, class_names, model_type="породы"):
    """Оценка модели на всех тестовых данных"""
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    if len(image_paths) == 0:
        print("❌ Нет данных для оценки")
        return 0
    
<<<<<<< HEAD
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
=======
    print(f"\n📊 ПОЛНАЯ ОЦЕНКА МОДЕЛИ ({model_type}):")
    print("=" * 50)
    
    correct = 0
    total = len(image_paths)
    
    for i, (img_path, true_label) in enumerate(zip(image_paths, labels)):
        try:
            img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
>>>>>>> e4a4378eac530946868097685580eb82d315742b
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
<<<<<<< HEAD
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
=======
            is_correct = (predicted_class == true_label)
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            pred_name = class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
            
            print(f"{status} {i+1:2d}/{total}: {Path(img_path).name:15} | Истина: {true_name:20} | Предсказание: {pred_name:20} | Уверенность: {confidence:.2%}")
            
        except Exception as e:
            print(f"❌ Ошибка при оценке {img_path}: {e}")
    
    accuracy = correct / total
    print(f"\n🎯 ИТОГОВАЯ ТОЧНОСТЬ: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def predict_new_image(model, image_path, class_names, model_type="породы"):
    """Предсказание для нового изображения"""
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    print(f"\n🎯 ПРЕДСКАЗАНИЕ ДЛЯ НОВОГО ИЗОБРАЖЕНИЯ ({model_type}):")
    print("=" * 50)
    
    class_name, confidence = test_model(model, image_path, class_names, model_type)
    
    if confidence > 0.7:
        print(f"✅ Высокая уверенность: {confidence:.2%}")
    elif confidence > 0.3:
        print(f"⚠️ Средняя уверенность: {confidence:.2%}")
    else:
        print(f"❌ Низкая уверенность: {confidence:.2%}")
    
    return class_name, confidence

# ЗАПУСК ПРОГРАММЫ
if __name__ == "__main__":
    print("🌲 ЗАПУСК СИСТЕМЫ КЛАССИФИКАЦИИ ДЕРЕВЬЕВ")
    print("=" * 60)
    
    # Сначала создаем правильные CSV файлы
    print("📝 СОЗДАНИЕ ПРАВИЛЬНЫХ CSV ФАЙЛОВ...")
    create_proper_csv_files()
>>>>>>> e4a4378eac530946868097685580eb82d315742b
    
    porody_path = "data/породы"
    char_path = "data/характеристики"
    
<<<<<<< HEAD
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





=======
    # Обучаем модель для пород
    porody_model, porody_history, species_names, porody_images, porody_labels = train_tree_species_model(porody_path)
    
    print("\n" + "=" * 60)
    
    # Обучаем модель для характеристик
    defects_model, defects_history, defect_descriptions, defects_images, defects_labels = train_defects_model(char_path)
    
    # Визуализация результатов обучения
    if porody_history:
        plot_training_history(porody_history, 'Классификация пород деревьев')
    
    if defects_history:
        plot_training_history(defects_history, 'Классификация характеристик')
    
    print("\n" + "=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
    print("=" * 60)
    
    # Тестирование модели пород
    if porody_model is not None and len(species_names) > 0 and len(porody_images) > 0:
        print("🌳 ТЕСТИРОВАНИЕ МОДЕЛИ ПОРОД:")
        
        # Тестируем на первом изображении
        test_image = porody_images[0]
        test_model(porody_model, test_image, species_names, "породы")
        
        # Полная оценка на всех данных
        porody_accuracy = evaluate_model(porody_model, porody_images, porody_labels, species_names, "породы")
        
        # Сохранение модели
        porody_model.save('model_porody.h5')
        print("✅ Модель пород сохранена как 'model_porody.h5'")
    
    # Тестирование модели характеристик
    if defects_model is not None and len(defect_descriptions) > 0 and len(defects_images) > 0:
        print("\n🔍 ТЕСТИРОВАНИЕ МОДЕЛИ ХАРАКТЕРИСТИК:")
        
        # Тестируем на первом изображении
        test_image = defects_images[0]
        test_model(defects_model, test_image, defect_descriptions, "характеристики")
        
        # Полная оценка на всех данных
        defects_accuracy = evaluate_model(defects_model, defects_images, defects_labels, defect_descriptions, "характеристики")
        
        # Сохранение модели
        defects_model.save('model_defects.h5')
        print("✅ Модель характеристик сохранена как 'model_defects.h5'")
    
    print("\n🎉 ПРОГРАММА ЗАВЕРШЕНА!")
>>>>>>> e4a4378eac530946868097685580eb82d315742b
