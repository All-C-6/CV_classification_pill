import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple
import logging

from utils import setup_logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PillClassificationModel(nn.Module):
    """
    Модель для классификации таблеток на изображениях.

    Архитектура: предобученная ResNet50 с модифицированным финальным слоем.
    """

    def __init__(
        self,
        number_of_pill_classes: int,
        pretrained_backbone: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone_layers: bool = False
    ):
        """
        Args:
            number_of_pill_classes: Количество классов таблеток для классификации
            pretrained_backbone: Использовать ли предобученные веса ImageNet
            dropout_rate: Коэффициент dropout для регуляризации
            freeze_backbone_layers: Заморозить ли слои backbone для fine-tuning
        """
        super(PillClassificationModel, self).__init__()

        self.number_of_pill_classes = number_of_pill_classes
        logger.info(f"Инициализация модели для {number_of_pill_classes} классов таблеток")

        # Загрузка предобученной ResNet50
        self.backbone_network = models.resnet50(pretrained=pretrained_backbone)

        # Заморозка слоев backbone при необходимости
        if freeze_backbone_layers:
            logger.info("Замораживание слоев backbone")
            for parameter in self.backbone_network.parameters():
                parameter.requires_grad = False

        # Получение размера выходного слоя backbone
        number_of_backbone_features = self.backbone_network.fc.in_features

        # Замена финального классификационного слоя
        self.backbone_network.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(number_of_backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, number_of_pill_classes)
        )

        logger.info(f"Модель инициализирована успешно")

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            input_images: Тензор изображений размера [batch_size, 3, 224, 224]

        Returns:
            Логиты классификации размера [batch_size, number_of_pill_classes]
        """
        classification_logits = self.backbone_network(input_images)
        return classification_logits

    def unfreeze_backbone_layers(self, number_of_layers_to_unfreeze: Optional[int] = None):
        """
        Разморозка слоев backbone для fine-tuning.

        Args:
            number_of_layers_to_unfreeze: Количество последних слоев для разморозки.
                                          Если None, размораживаются все слои.
        """
        if number_of_layers_to_unfreeze is None:
            logger.info("Размораживание всех слоев backbone")
            for parameter in self.backbone_network.parameters():
                parameter.requires_grad = True
        else:
            logger.info(f"Размораживание последних {number_of_layers_to_unfreeze} слоев")
            # Реализация постепенной разморозки слоев
            all_parameters = list(self.backbone_network.parameters())
            for parameter in all_parameters[-number_of_layers_to_unfreeze:]:
                parameter.requires_grad = True


class PillClassificationTrainer:
    """
    Класс для обучения модели классификации таблеток.
    """

    def __init__(
        self,
        model: PillClassificationModel,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Args:
            model: Модель для обучения
            device: Устройство для вычислений (CPU/GPU)
            learning_rate: Скорость обучения
            weight_decay: Коэффициент L2 регуляризации
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        # Функция потерь для многоклассовой классификации
        self.loss_function = nn.CrossEntropyLoss()

        # Оптимизатор Adam
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler для уменьшения learning rate
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        logger.info(f"Trainer инициализирован на устройстве: {device}")

    def train_single_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        epoch_number: int
    ) -> Tuple[float, float]:
        """
        Обучение модели на одной эпохе.

        Args:
            train_dataloader: DataLoader с обучающими данными
            epoch_number: Номер текущей эпохи

        Returns:
            Кортеж (средние потери, точность) за эпоху
        """
        self.model.train()

        total_epoch_loss = 0.0
        total_correct_predictions = 0
        total_samples_processed = 0

        for batch_index, batch_data in enumerate(train_dataloader):
            # Извлечение данных из батча
            input_images = batch_data['image'].to(self.device)
            target_class_indices = batch_data['class_index'].to(self.device)
            # bounding_boxes можно использовать для augmentation или region-based подходов
            # bounding_boxes = batch_data['bounding_box']

            # Обнуление градиентов
            self.optimizer.zero_grad()

            # Прямой проход
            predicted_logits = self.model(input_images)

            # Вычисление функции потерь
            batch_loss = self.loss_function(predicted_logits, target_class_indices)

            # Обратное распространение
            batch_loss.backward()

            # Обновление весов
            self.optimizer.step()

            # Вычисление метрик
            _, predicted_class_indices = torch.max(predicted_logits, dim=1)
            batch_correct_predictions = (predicted_class_indices == target_class_indices).sum().item()
            batch_size = input_images.size(0)

            total_epoch_loss += batch_loss.item() * batch_size
            total_correct_predictions += batch_correct_predictions
            total_samples_processed += batch_size

            # Логирование прогресса каждые 50 батчей
            if (batch_index + 1) % 50 == 0:
                current_batch_accuracy = 100.0 * batch_correct_predictions / batch_size
                logger.info(
                    f"Эпоха [{epoch_number}] Батч [{batch_index + 1}/{len(train_dataloader)}] "
                    f"Loss: {batch_loss.item():.4f} Accuracy: {current_batch_accuracy:.2f}%"
                )

        average_epoch_loss = total_epoch_loss / total_samples_processed
        epoch_accuracy = 100.0 * total_correct_predictions / total_samples_processed

        logger.info(
            f"Эпоха [{epoch_number}] завершена - "
            f"Средний Loss: {average_epoch_loss:.4f} Accuracy: {epoch_accuracy:.2f}%"
        )

        return average_epoch_loss, epoch_accuracy

    def validate_model(
        self,
        validation_dataloader: torch.utils.data.DataLoader,
        epoch_number: int
    ) -> Tuple[float, float]:
        """
        Валидация модели.

        Args:
            validation_dataloader: DataLoader с валидационными данными
            epoch_number: Номер текущей эпохи

        Returns:
            Кортеж (средние потери, точность) на валидации
        """
        self.model.eval()

        total_validation_loss = 0.0
        total_correct_predictions = 0
        total_samples_processed = 0

        with torch.no_grad():
            for batch_data in validation_dataloader:
                input_images = batch_data['image'].to(self.device)
                target_class_indices = batch_data['class_index'].to(self.device)

                # Прямой проход
                predicted_logits = self.model(input_images)

                # Вычисление функции потерь
                batch_loss = self.loss_function(predicted_logits, target_class_indices)

                # Вычисление метрик
                _, predicted_class_indices = torch.max(predicted_logits, dim=1)
                batch_correct_predictions = (predicted_class_indices == target_class_indices).sum().item()
                batch_size = input_images.size(0)

                total_validation_loss += batch_loss.item() * batch_size
                total_correct_predictions += batch_correct_predictions
                total_samples_processed += batch_size

        average_validation_loss = total_validation_loss / total_samples_processed
        validation_accuracy = 100.0 * total_correct_predictions / total_samples_processed

        logger.info(
            f"Валидация [Эпоха {epoch_number}] - "
            f"Loss: {average_validation_loss:.4f} Accuracy: {validation_accuracy:.2f}%"
        )

        # Обновление learning rate на основе валидационных потерь
        self.learning_rate_scheduler.step(average_validation_loss)

        return average_validation_loss, validation_accuracy

    def save_model_checkpoint(
        self,
        checkpoint_file_path: str,
        epoch_number: int,
        validation_accuracy: float
    ):
        """
        Сохранение чекпойнта модели.

        Args:
            checkpoint_file_path: Путь для сохранения чекпойнта
            epoch_number: Номер эпохи
            validation_accuracy: Точность на валидации
        """
        checkpoint_data = {
            'epoch': epoch_number,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'validation_accuracy': validation_accuracy,
            'number_of_pill_classes': self.model.number_of_pill_classes
        }

        torch.save(checkpoint_data, checkpoint_file_path)
        logger.info(f"Чекпойнт сохранен: {checkpoint_file_path}")

    def load_model_checkpoint(self, checkpoint_file_path: str) -> Dict:
        """
        Загрузка чекпойнта модели.

        Args:
            checkpoint_file_path: Путь к файлу чекпойнта

        Returns:
            Словарь с информацией о чекпойнте
        """
        checkpoint_data = torch.load(checkpoint_file_path, map_location=self.device)

        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        logger.info(
            f"Чекпойнт загружен: {checkpoint_file_path} "
            f"(Эпоха {checkpoint_data['epoch']}, "
            f"Accuracy: {checkpoint_data['validation_accuracy']:.2f}%)"
        )

        return checkpoint_data
