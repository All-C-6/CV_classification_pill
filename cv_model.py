import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from pathlib import Path
from typing import Optional
import time

from utils import setup_logging


logger = logging.getLogger(__name__)
setup_logging(log_file_path="logs/cv_model.log", level="INFO")


class PillClassifierDenseNet(nn.Module):
    """
    Классификатор таблеток на базе DenseNet с механизмами борьбы с переобучением.

    Используемые техники:
    - Предобученная модель DenseNet121 (легче, чем DenseNet161/201)
    - Dropout для регуляризации
    - Label Smoothing в loss функции
    - Градиентное размораживание слоев
    """

    def __init__(
        self,
        number_of_classes: int = 224,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone_initially: bool = True
    ):
        super(PillClassifierDenseNet, self).__init__()

        # Используем DenseNet121 - он легче и менее склонен к переобучению
        self.backbone = models.densenet121(pretrained=pretrained)

        # Замораживаем backbone изначально для постепенного обучения
        if freeze_backbone_initially:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # Получаем размер признаков из последнего слоя DenseNet
        number_of_features = self.backbone.classifier.in_features

        # Заменяем classifier на свой с дополнительной регуляризацией
        self.backbone.classifier = nn.Identity()

        # Создаем собственный классификатор с dropout и batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(number_of_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(number_of_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, number_of_classes)
        )

        # Инициализация весов классификатора
        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов классификатора"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_tensor)
        output = self.classifier(features)
        return output

    def unfreeze_backbone_gradually(self, stage: int = 1):
        """
        Постепенное размораживание backbone.
        stage=1: размораживаем последний transition блок и denseblock4
        stage=2: размораживаем denseblock3
        stage=3: размораживаем все
        """
        if stage >= 1:
            # Размораживаем denseblock4 и последний transition
            for name, parameter in self.backbone.named_parameters():
                if 'denseblock4' in name or 'transition3' in name:
                    parameter.requires_grad = True

        if stage >= 2:
            # Размораживаем denseblock3
            for name, parameter in self.backbone.named_parameters():
                if 'denseblock3' in name:
                    parameter.requires_grad = True

        if stage >= 3:
            # Размораживаем все
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_epochs: int = 100,
    train_accuracy_threshold: float = 0.75,
    val_accuracy_threshold: float = 0.50,
    min_train_accuracy_for_save: float = 0.50,
    initial_learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    checkpoint_directory: Path = Path('./checkpoints'),
    unfreeze_schedule: Optional[dict[int, int]] = None
) -> dict:
    """
    Функция обучения классификатора таблеток с ранней остановкой.

    Args:
        model: Модель классификатора
        train_loader: DataLoader для обучающей выборки
        val_loader: DataLoader для валидационной выборки
        device: Устройство (cuda/cpu)
        max_epochs: Максимальное количество эпох
        train_accuracy_threshold: Порог точности на train для остановки (75%)
        val_accuracy_threshold: Порог точности на val для остановки (50%)
        min_train_accuracy_for_save: Минимальная точность на train для сохранения (50%)
        initial_learning_rate: Начальная скорость обучения
        weight_decay: L2 регуляризация
        label_smoothing: Label smoothing для борьбы с переобучением
        checkpoint_directory: Директория для сохранения чекпоинтов
        unfreeze_schedule: Расписание размораживания слоев {epoch: stage}

    Returns:
        Словарь с историей обучения
    """

    # Создаем директорию для чекпоинтов
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # Переносим модель на устройство
    model = model.to(device)

    # Optimizer с разными learning rates для backbone и classifier
    optimizer_parameters = [
        {'params': model.classifier.parameters(), 'lr': initial_learning_rate},
        {'params': model.backbone.parameters(), 'lr': initial_learning_rate * 0.1}
    ]
    optimizer = AdamW(optimizer_parameters, weight_decay=weight_decay)

    # Loss функция с label smoothing для борьбы с переобучением
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Scheduler для динамического изменения learning rate
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Расписание размораживания по умолчанию
    if unfreeze_schedule is None:
        unfreeze_schedule = {
            5: 1,   # После 5 эпох размораживаем denseblock4
            15: 2,  # После 15 эпох размораживаем denseblock3
            30: 3   # После 30 эпох размораживаем все
        }

    # История обучения
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': [],
        'epoch_times': []
    }

    best_val_accuracy = 0.0
    best_model_path = None

    logger.info(f"Начало обучения на устройстве: {device}")
    logger.info(f"Количество параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info("-" * 100)

    for epoch in range(1, max_epochs + 1):
        epoch_start_time = time.time()

        # Проверяем расписание размораживания
        if epoch in unfreeze_schedule:
            stage = unfreeze_schedule[epoch]
            logger.debug(f"\n>>> Размораживаем backbone (stage {stage}) <<<")
            model.unfreeze_backbone_gradually(stage)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.debug(f"Обучаемых параметров теперь: {trainable_params:,}\n")

        # === ОБУЧЕНИЕ ===
        model.train()
        train_loss_accumulated = 0.0
        train_correct_predictions = 0
        train_total_samples = 0

        for batch_index, batch_data in enumerate(train_loader):
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Статистика
            train_loss_accumulated += loss.item() * images.size(0)
            _, predicted_classes = torch.max(outputs, 1)
            train_correct_predictions += (predicted_classes == labels).sum().item()
            train_total_samples += labels.size(0)

        train_loss_average = train_loss_accumulated / train_total_samples
        train_accuracy = train_correct_predictions / train_total_samples

        # === ВАЛИДАЦИЯ ===
        model.eval()
        val_loss_accumulated = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_accumulated += loss.item() * images.size(0)
                _, predicted_classes = torch.max(outputs, 1)
                val_correct_predictions += (predicted_classes == labels).sum().item()
                val_total_samples += labels.size(0)

        val_loss_average = val_loss_accumulated / val_total_samples
        val_accuracy = val_correct_predictions / val_total_samples

        # Обновляем scheduler
        scheduler.step()
        current_learning_rate = optimizer.param_groups[0]['lr']

        # Время эпохи
        epoch_time = time.time() - epoch_start_time

        # Сохраняем историю
        training_history['train_loss'].append(train_loss_average)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['val_loss'].append(val_loss_average)
        training_history['val_accuracy'].append(val_accuracy)
        training_history['learning_rates'].append(current_learning_rate)
        training_history['epoch_times'].append(epoch_time)

        # Выводим результаты эпохи
        print(f"Эпоха [{epoch:3d}/{max_epochs}] | "
              f"Время: {epoch_time:6.2f}s | "
              f"LR: {current_learning_rate:.2e}")
        print(f"  Train: Loss={train_loss_average:.4f}, Accuracy={train_accuracy*100:6.2f}%")
        print(f"  Val:   Loss={val_loss_average:.4f}, Accuracy={val_accuracy*100:6.2f}%")

        # Сохраняем лучшую модель (только если train accuracy > 50%)
        if val_accuracy > best_val_accuracy and train_accuracy >= min_train_accuracy_for_save:
            best_val_accuracy = val_accuracy

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss_average,
                'val_loss': val_loss_average,
            }

            best_model_path = checkpoint_directory / f'best_model_epoch{epoch:03d}_valacc{val_accuracy*100:.2f}.pth'
            torch.save(checkpoint_data, best_model_path)
            logger.info(f"  ✓ Сохранена лучшая модель: {best_model_path.name}")

        print("-" * 100)

        # Проверка условий остановки
        if train_accuracy >= train_accuracy_threshold:
            print(f"\nДостигнут порог точности на обучающей выборке: {train_accuracy*100:.2f}% >= {train_accuracy_threshold*100:.2f}%")
            print("Остановка обучения.")
            break

        if val_accuracy >= val_accuracy_threshold:
            print(f"\nДостигнут порог точности на валидационной выборке: {val_accuracy*100:.2f}% >= {val_accuracy_threshold*100:.2f}%")
            print("Остановка обучения.")
            break

    # Финальная статистика
    print("\n" + "=" * 100)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 100)
    print(f"Всего эпох: {epoch}")
    print(f"Лучшая val accuracy: {best_val_accuracy*100:.2f}%")
    if best_model_path:
        print(f"Лучшая модель: {best_model_path}")
    else:
        print("Модель не была сохранена (train accuracy < 50%)")
    print(f"Финальная train accuracy: {train_accuracy*100:.2f}%")
    print(f"Финальная val accuracy: {val_accuracy*100:.2f}%")

    training_history['best_val_accuracy'] = best_val_accuracy
    training_history['best_model_path'] = str(best_model_path) if best_model_path else None
    training_history['total_epochs'] = epoch

    return training_history


def evaluate_model_on_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    checkpoint_path: Optional[Path] = None,
    class_names: Optional[list[str]] = None
) -> dict:
    """
    Оценка модели на тестовой выборке с детальным анализом по классам.

    Args:
        model: Модель классификатора
        test_loader: DataLoader для тестовой выборки
        device: Устройство (cuda/cpu)
        checkpoint_path: Путь к чекпоинту (если нужно загрузить веса)
        class_names: Список имен классов для читаемого вывода

    Returns:
        Словарь с метриками на тестовой выборке
    """
    

    # Загружаем веса если указан путь
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Загружены веса из {checkpoint_path}")
        logger.info(f"  Эпоха чекпоинта: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"  Val accuracy чекпоинта: {checkpoint.get('val_accuracy', 0)*100:.2f}%")

    model = model.to(device)
    model.eval()

    test_correct_predictions = 0
    test_total_samples = 0
    all_predictions = []
    all_true_labels = []

    logger.info("\nОценка на тестовой выборке...")

    with torch.no_grad():
        for batch_data in test_loader:
            images = batch_data['image'].to(device)
            labels = batch_data['label'].to(device)

            outputs = model(images)
            _, predicted_classes = torch.max(outputs, 1)

            test_correct_predictions += (predicted_classes == labels).sum().item()
            test_total_samples += labels.size(0)

            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # Общая accuracy
    test_accuracy = accuracy_score(all_true_labels, all_predictions)

    # Вычисляем метрики для каждого класса
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(
            all_true_labels, 
            all_predictions, 
            average=None,
            zero_division=0
        )

    # Определяем количество уникальных классов
    unique_classes = sorted(set(all_true_labels))
    number_of_classes = len(unique_classes)

    # Создаем список кортежей (класс, f1_score) для сортировки
    class_performance = [
        (class_idx, f1_per_class[class_idx]) 
        for class_idx in unique_classes
    ]

    # Сортируем по F1-score
    sorted_by_f1_descending = sorted(
        class_performance, 
        key=lambda x: x[1], 
        reverse=True
    )
    sorted_by_f1_ascending = sorted(
        class_performance, 
        key=lambda x: x[1], 
        reverse=False
    )

    # Выбираем топ-5 лучших и худших
    top_5_best_classes = sorted_by_f1_descending[:5]
    top_5_worst_classes = sorted_by_f1_ascending[:5]

    # Вывод результатов
    print(f"\n{'='*80}")
    print(f"РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*80}")
    print(f"Общая Accuracy: {test_accuracy*100:.2f}%")
    print(f"Правильно классифицировано: {test_correct_predictions}/{test_total_samples}")
    print(f"Всего классов: {number_of_classes}")

    # Функция для получения имени класса
    def get_class_display_name(class_idx: int) -> str:
        if class_names is not None and class_idx < len(class_names):
            return f"{class_idx} ({class_names[class_idx]})"
        return str(class_idx)

    # Вывод топ-5 лучших классов
    print(f"\n{'='*80}")
    print(f"ТОП-5 ЛУЧШИХ КЛАССОВ (по F1-score)")
    print(f"{'='*80}")
    print(f"{'Класс':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print(f"{'-'*80}")

    for class_idx, f1_score in top_5_best_classes:
        class_display_name = get_class_display_name(class_idx)
        precision = precision_per_class[class_idx]
        recall = recall_per_class[class_idx]
        support = support_per_class[class_idx]

        print(f"{class_display_name:<30} "
              f"{precision*100:>10.2f}%  "
              f"{recall*100:>10.2f}%  "
              f"{f1_score*100:>10.2f}%  "
              f"{support:>10}")

    # Вывод топ-5 худших классов
    print(f"\n{'='*80}")
    print(f"ТОП-5 ХУДШИХ КЛАССОВ (по F1-score)")
    print(f"{'='*80}")
    print(f"{'Класс':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print(f"{'-'*80}")

    for class_idx, f1_score in top_5_worst_classes:
        class_display_name = get_class_display_name(class_idx)
        precision = precision_per_class[class_idx]
        recall = recall_per_class[class_idx]
        support = support_per_class[class_idx]

        print(f"{class_display_name:<30} "
              f"{precision*100:>10.2f}%  "
              f"{recall*100:>10.2f}%  "
              f"{f1_score*100:>10.2f}%  "
              f"{support:>10}")

    print(f"{'='*80}\n")

    return {
        'test_accuracy': test_accuracy,
        'correct_predictions': test_correct_predictions,
        'total_samples': test_total_samples,
        'predictions': all_predictions,
        'true_labels': all_true_labels,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'top_5_best_classes': top_5_best_classes,
        'top_5_worst_classes': top_5_worst_classes
    }

