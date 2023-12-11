from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger
from typing import Tuple

@dataclass
class ModelConfig:
    sd_path: str = '/home/wujinbo/Downloads/weights/epicrealism_pureEvolutionV5/'
    depth_control_path: str = '/home/wujinbo/Downloads/weights/control_v11f1p_sd15_depth/'
    res: Tuple[int, int] = (512, 512)

