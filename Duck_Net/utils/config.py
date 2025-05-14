from dataclasses import dataclass


@dataclass
class Configs:
    ROOT_DIR: str = '.'
    train_dataset: str = 'CVC-ClinicDB'


    # GPU configs
    gpu_id: int = 3

    # Model configs
    input_channels: int = 3
    num_classes: int = 1
    num_filters: int = 17

    # Dataset configs
    batch_size: int = 2
    num_workers: int = 8

    # Optimizer configs
    lr: float = 0.001
    betas: tuple = (0.9, 0.99)
    weight_decay: float = 0.0001

    # Scheduler configs
    step_size: int = 3
    gamma: float = 0.1

    # Training configs
    num_epochs: int = 30
    early_stopping: int = 3
    early_stop_patience: int = 0
    save_dir: str = 'checkpoints'