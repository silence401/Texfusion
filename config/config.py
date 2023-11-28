from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from typing import Tuple

@dataclass
class ModelConfig:
    sd_path: str = '/workspace/code/baidu/ar/neural_engine/algorithms/text23D/StableDiffusion/epicrealism_pureEvolutionV5'
    depth_control_path: str = '/workspace/code/baidu/ar/neural_engine/algorithms/text23D/StableDiffusion/control_v11f1p_sd15_depth'
    res: Tuple[int, int] = (512, 512)

