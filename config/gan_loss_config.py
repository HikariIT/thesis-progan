from dataclasses import dataclass


@dataclass
class GANLossConfig:
    lambda_gp: float = 10.0