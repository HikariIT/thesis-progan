from dataclasses import dataclass, field

@dataclass
class GANTrainingConfig:
    img_generation_interval: int = 1000
    save_interval: int = 1000
    log_interval: int = 100

    log_dir: str = "runs/"

    steps_for_depth: list[int] = field(default_factory=lambda: [20000, 20000, 20000, 20000, 20000, 40000])
    transition_steps_for_depth: list[int] = field(default_factory=lambda: [20000, 20000, 20000, 20000, 40000])
    batch_sizes_for_depth: list[int] = field(default_factory=lambda: [16, 16, 16, 16, 16, 8])

    num_workers: int = 0
    checkpoint_images: int = 4

    path_to_saved_model: str | None = None
    save_dir: str = "saved_models/"
    no_concurrent_saves: int = 5

    pin_memory: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
