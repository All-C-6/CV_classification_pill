import logging
from pathlib import Path
import shutil
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import setup_logging
from torchvision import transforms
from collections import Counter

logger = logging.getLogger(__name__)
setup_logging(log_file_path="logs/dataset_executor.log", level="INFO")


class PillDataset(Dataset):
    def __init__(self, dataset_root_path, split='train', image_size=224):
        """
        Инициализация датасета для загрузки изображений таблеток с их координатами.

        Args:
            dataset_root_path: путь до корневой папки датасета kaggle
            split: 'train', 'valid' или 'test'
            image_size: размер изображения для ресайза
        """
        self.split = split
        self.image_size = image_size

        # Путь к данным: dataset_root/ogyeiv2/ogyeiv2/{split}
        split_path = Path(dataset_root_path) / 'ogyeiv2' / 'ogyeiv2' / split
        self.images_path = split_path / 'images'
        self.labels_path = split_path / 'labels'

        # Получаем список всех изображений
        self.image_files = sorted(list(self.images_path.glob('*.jpg')))

        # Извлекаем классы из имен файлов (все до последнего _номер)
        self.classes = []
        for img_file in self.image_files:
            # Убираем расширение и последний номер
            class_name = '_'.join(img_file.stem.split('_')[:-1])
            self.classes.append(class_name)

        # Уникальные классы
        self.unique_classes = sorted(list(set(self.classes)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}

        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Загружаем координаты из txt
        label_path = self.labels_path / f"{img_path.stem}.txt"
        with open(label_path, 'r') as f:
            coords_line = f.readline().strip().split()
            # Формат: class_id center_x center_y width height (нормализованные 0-1)
            class_id = int(coords_line[0])
            bbox_coords = [float(x) for x in coords_line[1:5]]

        # Применяем трансформации
        image_tensor = self.transform(image)

        # Получаем индекс класса по имени файла
        class_name = '_'.join(img_path.stem.split('_')[:-1])
        label = self.class_to_idx[class_name]

        return {
            'image': image_tensor,
            'label': label,
            'bbox': torch.tensor(bbox_coords, dtype=torch.float32),
            'class_name': class_name
        }

    def get_class_counts(self):
        """Возвращает Counter с количеством изображений для каждого класса"""
        return Counter(self.classes)


def load_cv_data(dataset_root_path, batch_size=32, image_size=224, num_workers=4):
    """
    Загружает train, validation и test DataLoader'ы для датасета таблеток.

    Args:
        dataset_root_path: путь до сохраненного датасета kaggle
        batch_size: размер батча (для 16GB VRAM рекомендуется 32-64)
        image_size: размер изображения
        num_workers: количество воркеров для загрузки данных

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    # Создаем датасеты
    train_dataset = PillDataset(dataset_root_path, split='train', image_size=image_size)
    val_dataset = PillDataset(dataset_root_path, split='valid', image_size=image_size)
    test_dataset = PillDataset(dataset_root_path, split='test', image_size=image_size)

    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    all_classes = (train_dataset.unique_classes, val_dataset.unique_classes, test_dataset.unique_classes)

    return train_loader, val_loader, test_loader, all_classes

def download_kaggle_cv_dataset(kaggle_dataset_name: str) -> Path:
    """
    Загружает датасет из Kaggle в папку /dataset рядом с текущим файлом.

    Проверяет наличие папки и её заполненность перед загрузкой.
    Если папка существует и не пуста, загрузка не выполняется.

    Args:
        kaggle_dataset_name: Название датасета в формате "username/dataset-name"
                            (например, "richardradli/ogyeiv2")

    Returns:
        Path: Путь к папке с загруженным датасетом

    Raises:
        ValueError: Если kaggle_dataset_name имеет некорректный формат
        RuntimeError: Если произошла ошибка при загрузке или копировании
    """
    # Валидация входного параметра
    if not kaggle_dataset_name or "/" not in kaggle_dataset_name:
        error_message = (
            f"Некорректный формат названия датасета: '{kaggle_dataset_name}'. "
            "Ожидается формат 'username/dataset-name'"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    # Получаем путь к текущей папке (где находится скрипт)
    current_script_directory = Path(__file__).parent

    # Извлекаем имя датасета для создания папки
    dataset_short_name = kaggle_dataset_name.split("/")[-1]
    target_dataset_directory = current_script_directory / "dataset" / dataset_short_name

    # Проверяем наличие папки и её содержимое
    if target_dataset_directory.exists():
        # Проверяем, не пуста ли папка
        directory_contents = list(target_dataset_directory.iterdir())
        if directory_contents:
            logger.info(
                f"Датасет уже существует в папке '{target_dataset_directory}' "
                f"и содержит {len(directory_contents)} файл(ов)/папок. Загрузка пропущена."
            )
            return target_dataset_directory
        else:
            logger.info(
                f"Папка '{target_dataset_directory}' существует, но пуста. "
                "Выполняется загрузка датасета."
            )
    else:
        logger.info(
            f"Папка '{target_dataset_directory}' не существует. "
            "Выполняется загрузка датасета."
        )

    try:
        # Загружаем датасет в кеш kagglehub
        logger.info(f"Начинается загрузка датасета '{kaggle_dataset_name}' из Kaggle...")
        path_to_cached_dataset = kagglehub.dataset_download(kaggle_dataset_name)
        logger.info(f"Датасет загружен в кеш: {path_to_cached_dataset}")

        # Создаём целевую папку, если её нет
        target_dataset_directory.parent.mkdir(parents=True, exist_ok=True)

        # Копируем датасет из кеша в целевую папку
        logger.info(
            f"Копирование датасета из кеша в '{target_dataset_directory}'..."
        )
        shutil.copytree(
            path_to_cached_dataset, 
            target_dataset_directory, 
            dirs_exist_ok=True
        )
        logger.info(
            f"Датасет успешно скопирован в '{target_dataset_directory}'"
        )

        return target_dataset_directory

    except Exception as exception_during_download:
        error_message = (
            f"Ошибка при загрузке или копировании датасета '{kaggle_dataset_name}': "
            f"{type(exception_during_download).__name__}: {exception_during_download}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message) from exception_during_download


def debug_dataloader_output(dataloader, dataloader_name="DataLoader"):
    """Отладка: проверяем что возвращает DataLoader"""
    print(f"\n{'='*60}")
    print(f"Отладка {dataloader_name}")
    print(f"{'='*60}")

    # Получаем один батч
    batch_data = next(iter(dataloader))

    print(f"Тип возвращаемых данных: {type(batch_data)}")

    if isinstance(batch_data, (list, tuple)):
        print(f"Количество элементов в батче: {len(batch_data)}")
        for idx, item in enumerate(batch_data):
            if isinstance(item, torch.Tensor):
                print(f"  Элемент {idx}: Tensor, shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  Элемент {idx}: {type(item)}, значение={item}")
    elif isinstance(batch_data, dict):
        print(f"Батч - это словарь с ключами: {batch_data.keys()}")
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  '{key}': Tensor, shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  '{key}': {type(value)}")
    else:
        print(f"Батч имеет неожиданный тип: {type(batch_data)}")

    return batch_data
