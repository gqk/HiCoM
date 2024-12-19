from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import torch
from absl import logging

from libgs.pipeline import Config as BaseConfig
from libgs.pipeline import Pipeline as BasePipeline
from libgs.renderer import network_gui
from libgs.utils.config import to_yaml
from libgs.utils.dist import is_global_zero
from libgs.utils.general import safe_state

from .data import DataConfig, DataModule
from .module import Module, ModuleConfig
from .trainer import Trainer, TrainerConfig


@dataclass
class Config(BaseConfig):
    output_dir: Path = Path("output")
    experiment_name: str = datetime.now().strftime("%Y-%m-%d")
    mode: Literal["train", "validate"] = "train"
    ckpt_path: Optional[Path] = None
    gui_ip: str = "127.0.0.1"
    gui_port: int = 6009
    detect_anomaly: bool = False
    quiet: bool = False
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    module: ModuleConfig = field(default_factory=ModuleConfig)


class Pipeline(BasePipeline):
    datamodule: DataModule
    module: Module
    trainer: Trainer

    def setup(self):
        safe_state(self.config.quiet)
        network_gui.init(self.config.gui_ip, self.config.gui_port)
        torch.autograd.set_detect_anomaly(self.config.detect_anomaly)
        return super().setup()

    def setup_trainer_and_modules(self):
        output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.datamodule = DataModule(self.config.data)
        self.module = Module(self.config.module, datamodule=self.datamodule)
        self.trainer = Trainer(self.config.trainer, output_dir=output_dir)
