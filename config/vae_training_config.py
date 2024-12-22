from dataclasses import dataclass

@dataclass
class VAETrainingConfig:
    epochs: int = 500
    batch_size: int = 16

    log_interval: int = 100
    save_interval: int = 100

    save_dir: str = "saved_models/"
    log_dir: str = "runs/"
    no_concurrent_saves: int = 5


    num_workers: int = 0
    checkpoint_images: int = 4
    pin_memory: bool = True