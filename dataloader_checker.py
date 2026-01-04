import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def diagnose_dataloader(
    dataloader: torch.utils.data.DataLoader,
    number_of_batches_to_check: int = 5,
    visualize_samples: bool = True,
    dataset_name: str = "train"
):
    """
    Диагностика DataLoader для выявления проблем с данными.

    Args:
        dataloader: DataLoader для проверки
        number_of_batches_to_check: Количество батчей для проверки
        visualize_samples: Визуализировать ли примеры изображений
        dataset_name: Название датасета (для логирования)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ДИАГНОСТИКА DATALOADER: {dataset_name.upper()}")
    logger.info(f"{'='*60}\n")

    all_class_indices = []
    batch_sizes = []
    image_statistics = {
        'min_values': [],
        'max_values': [],
        'mean_values': [],
        'std_values': []
    }

    for batch_index, batch_data in enumerate(dataloader):
        if batch_index >= number_of_batches_to_check:
            break

        images = batch_data['image']
        class_indices = batch_data['class_index']
        bounding_boxes = batch_data['bounding_box']

        batch_size = images.shape[0]
        batch_sizes.append(batch_size)

        logger.info(f"\n--- Батч {batch_index + 1} ---")
        logger.info(f"  Shape изображений: {images.shape}")
        logger.info(f"  Dtype изображений: {images.dtype}")
        logger.info(f"  Device изображений: {images.device}")

        # Проверка диапазона значений пикселей
        batch_min_value = images.min().item()
        batch_max_value = images.max().item()
        batch_mean_value = images.mean().item()
        batch_std_value = images.std().item()

        image_statistics['min_values'].append(batch_min_value)
        image_statistics['max_values'].append(batch_max_value)
        image_statistics['mean_values'].append(batch_mean_value)
        image_statistics['std_values'].append(batch_std_value)

        logger.info(f"  Диапазон значений пикселей: [{batch_min_value:.4f}, {batch_max_value:.4f}]")
        logger.info(f"  Среднее значение пикселей: {batch_mean_value:.4f}")
        logger.info(f"  Std пикселей: {batch_std_value:.4f}")

        # Проверка классов
        if isinstance(class_indices, torch.Tensor):
            class_indices_list = class_indices.tolist()
        else:
            class_indices_list = class_indices

        all_class_indices.extend(class_indices_list)

        logger.info(f"  Type class_index: {type(class_indices)}")
        logger.info(f"  Shape class_index: {class_indices.shape if isinstance(class_indices, torch.Tensor) else len(class_indices)}")
        logger.info(f"  Классы в батче: {class_indices_list}")
        logger.info(f"  Уникальные классы: {len(set(class_indices_list))}")
        logger.info(f"  Min класс: {min(class_indices_list)}, Max класс: {max(class_indices_list)}")

        # Проверка bounding boxes
        logger.info(f"  Type bounding_box: {type(bounding_boxes)}")
        if isinstance(bounding_boxes, torch.Tensor):
            logger.info(f"  Shape bounding_box: {bounding_boxes.shape}")

        # Визуализация первого изображения из батча
        if visualize_samples and batch_index == 0:
            visualize_batch_samples(images, class_indices_list, dataset_name)

    # Общая статистика
    logger.info(f"\n{'='*60}")
    logger.info(f"ОБЩАЯ СТАТИСТИКА ПО {dataset_name.upper()} DATASET")
    logger.info(f"{'='*60}\n")

    logger.info(f"Всего проверено батчей: {len(batch_sizes)}")
    logger.info(f"Всего проверено образцов: {sum(batch_sizes)}")
    logger.info(f"Средний размер батча: {np.mean(batch_sizes):.2f}")

    # Статистика по пикселям
    logger.info(f"\nСтатистика значений пикселей:")
    logger.info(f"  Min across batches: {min(image_statistics['min_values']):.4f}")
    logger.info(f"  Max across batches: {max(image_statistics['max_values']):.4f}")
    logger.info(f"  Среднее Mean: {np.mean(image_statistics['mean_values']):.4f}")
    logger.info(f"  Среднее Std: {np.mean(image_statistics['std_values']):.4f}")

    # Проверка нормализации
    avg_mean = np.mean(image_statistics['mean_values'])
    avg_std = np.mean(image_statistics['std_values'])

    if -0.5 <= avg_mean <= 0.5 and 0.8 <= avg_std <= 1.2:
        logger.info(f"  ✓ Изображения, похоже, нормализованы корректно")
    elif 0 <= avg_mean <= 1 and min(image_statistics['min_values']) >= 0:
        logger.warning(f"  ⚠ Изображения в диапазоне [0, 1], но НЕ нормализованы!")
        logger.warning(f"    Рекомендуется добавить нормализацию ImageNet!")
    else:
        logger.warning(f"  ⚠ Необычные значения пикселей. Проверьте препроцессинг!")

    # Статистика по классам
    logger.info(f"\nСтатистика по классам:")
    logger.info(f"  Всего классов встречено: {len(set(all_class_indices))}")
    logger.info(f"  Диапазон классов: [{min(all_class_indices)}, {max(all_class_indices)}]")

    class_distribution = Counter(all_class_indices)
    most_common_classes = class_distribution.most_common(10)
    least_common_classes = class_distribution.most_common()[-10:]

    logger.info(f"\n  10 самых частых классов:")
    for class_idx, count in most_common_classes:
        logger.info(f"    Класс {class_idx}: {count} образцов")

    logger.info(f"\n  10 самых редких классов:")
    for class_idx, count in least_common_classes:
        logger.info(f"    Класс {class_idx}: {count} образцов")

    # Проверка дисбаланса классов
    max_count = max(class_distribution.values())
    min_count = min(class_distribution.values())
    imbalance_ratio = max_count / min_count

    logger.info(f"\n  Дисбаланс классов:")
    logger.info(f"    Максимум образцов в классе: {max_count}")
    logger.info(f"    Минимум образцов в классе: {min_count}")
    logger.info(f"    Соотношение дисбаланса: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 10:
        logger.warning(f"  ⚠ СИЛЬНЫЙ дисбаланс классов! Рекомендуется:")
        logger.warning(f"    - Использовать weighted loss")
        logger.warning(f"    - Применить oversampling/undersampling")
        logger.warning(f"    - Использовать focal loss")

    # Визуализация распределения классов
    visualize_class_distribution(class_distribution, dataset_name)

    logger.info(f"\n{'='*60}\n")

    return {
        'class_distribution': class_distribution,
        'image_statistics': image_statistics,
        'imbalance_ratio': imbalance_ratio,
        'unique_classes': len(set(all_class_indices))
    }


def visualize_batch_samples(
    images: torch.Tensor,
    class_indices: list,
    dataset_name: str,
    number_of_samples_to_show: int = 8
):
    """
    Визуализация образцов из батча.

    Args:
        images: Тензор изображений [batch_size, 3, 224, 224]
        class_indices: Список индексов классов
        dataset_name: Название датасета
        number_of_samples_to_show: Количество образцов для показа
    """
    number_of_samples_to_show = min(number_of_samples_to_show, images.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Образцы из {dataset_name} dataset', fontsize=16)
    axes = axes.flatten()

    for sample_index in range(number_of_samples_to_show):
        image = images[sample_index].cpu().numpy()
        class_index = class_indices[sample_index]

        # Транспонирование: [C, H, W] -> [H, W, C]
        image = np.transpose(image, (1, 2, 0))

        # Денормализация если изображение нормализовано
        if image.min() < 0:
            # ImageNet нормализация
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean

        # Клиппинг значений
        image = np.clip(image, 0, 1)

        axes[sample_index].imshow(image)
        axes[sample_index].set_title(f'Класс: {class_index}')
        axes[sample_index].axis('off')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_samples_visualization.png', dpi=150, bbox_inches='tight')
    logger.info(f"Визуализация сохранена: {dataset_name}_samples_visualization.png")
    plt.close()


def visualize_class_distribution(
    class_distribution: Counter,
    dataset_name: str
):
    """
    Визуализация распределения классов.

    Args:
        class_distribution: Counter с распределением классов
        dataset_name: Название датасета
    """
    sorted_classes = sorted(class_distribution.items())
    class_indices = [x[0] for x in sorted_classes]
    class_counts = [x[1] for x in sorted_classes]

    plt.figure(figsize=(20, 6))
    plt.bar(class_indices, class_counts, alpha=0.7)
    plt.xlabel('Индекс класса', fontsize=12)
    plt.ylabel('Количество образцов', fontsize=12)
    plt.title(f'Распределение классов в {dataset_name} dataset', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_class_distribution.png', dpi=150, bbox_inches='tight')
    logger.info(f"График распределения сохранен: {dataset_name}_class_distribution.png")
    plt.close()


def check_dataloader_for_issues(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    expected_number_of_classes: int
):
    """
    Комплексная проверка train и validation DataLoader'ов.

    Args:
        train_loader: Train DataLoader
        val_loader: Validation DataLoader
        expected_number_of_classes: Ожидаемое количество классов
    """
    logger.info("\n" + "="*60)
    logger.info("НАЧАЛО КОМПЛЕКСНОЙ ДИАГНОСТИКИ DATALOADERS")
    logger.info("="*60 + "\n")

    # Проверка train loader
    train_stats = diagnose_dataloader(
        train_loader,
        number_of_batches_to_check=10,
        visualize_samples=True,
        dataset_name="train"
    )

    # Проверка validation loader
    val_stats = diagnose_dataloader(
        val_loader,
        number_of_batches_to_check=5,
        visualize_samples=True,
        dataset_name="validation"
    )

    # Сравнительный анализ
    logger.info("\n" + "="*60)
    logger.info("СРАВНИТЕЛЬНЫЙ АНАЛИЗ TRAIN vs VALIDATION")
    logger.info("="*60 + "\n")

    logger.info(f"Ожидаемое количество классов: {expected_number_of_classes}")
    logger.info(f"Train уникальных классов: {train_stats['unique_classes']}")
    logger.info(f"Validation уникальных классов: {val_stats['unique_classes']}")

    if train_stats['unique_classes'] < expected_number_of_classes:
        logger.warning(f"⚠ В train выборке меньше классов чем ожидается!")

    if val_stats['unique_classes'] < expected_number_of_classes:
        logger.warning(f"⚠ В validation выборке меньше классов чем ожидается!")

    # Проверка пересечения классов
    train_classes = set(train_stats['class_distribution'].keys())
    val_classes = set(val_stats['class_distribution'].keys())

    classes_only_in_train = train_classes - val_classes
    classes_only_in_val = val_classes - train_classes

    if classes_only_in_val:
        logger.warning(f"⚠ ПРОБЛЕМА: {len(classes_only_in_val)} классов есть только в validation!")
        logger.warning(f"  Модель не видела эти классы на обучении:")
        logger.warning(f"  {sorted(list(classes_only_in_val))[:20]}")

    if classes_only_in_train:
        logger.info(f"ℹ {len(classes_only_in_train)} классов есть только в train")

    # КРИТИЧЕСКИЕ ПРОБЛЕМЫ
    logger.info("\n" + "="*60)
    logger.info("ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ И РЕКОМЕНДАЦИИ")
    logger.info("="*60 + "\n")

    problems_found = []

    # 1. Проверка нормализации
    avg_train_mean = np.mean(train_stats['image_statistics']['mean_values'])
    if not (-0.5 <= avg_train_mean <= 0.5):
        problems_found.append("Отсутствует нормализация ImageNet")
        logger.error("❌ КРИТИЧНО: Изображения НЕ нормализованы!")
        logger.error("   Добавьте в transforms:")
        logger.error("   transforms.Normalize(mean=[0.485, 0.456, 0.406],")
        logger.error("                        std=[0.229, 0.224, 0.225])")

    # 2. Проверка дисбаланса
    if train_stats['imbalance_ratio'] > 10:
        problems_found.append(f"Сильный дисбаланс классов ({train_stats['imbalance_ratio']:.1f}:1)")
        logger.warning(f"⚠ Дисбаланс классов может ухудшать обучение")

    # 3. Проверка количества классов
    if train_stats['unique_classes'] < expected_number_of_classes * 0.9:
        problems_found.append("Недостаточно классов в датасете")
        logger.error(f"❌ В данных меньше классов чем ожидается")

    if not problems_found:
        logger.info("✓ Критических проблем не обнаружено")

    return train_stats, val_stats

