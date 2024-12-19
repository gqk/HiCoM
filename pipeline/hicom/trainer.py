from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from absl import logging
from tqdm import tqdm

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer
from libgs.pipeline import TrainerConfig as BaseTrainerConfig


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    testing_steps: List[int] = field(default_factory=lambda: [5000])
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [5000])
    saving_ckpt_every_n_frames: int = 1000
    num_init_steps: int = 15000
    num_incr_steps: int = 200


class Trainer(BaseTrainer):
    @property
    def num_frames(self) -> int:
        return self.datamodule.num_frames

    @property
    def current_frame(self) -> int:
        return self.datamodule.current_frame

    @property
    def is_first_frame(self) -> bool:
        return self.current_frame == 0  # range from 0

    @property
    def is_last_frame(self) -> bool:
        return self.current_frame == self.num_frames - 1  # range from 0

    @property
    def num_steps(self) -> int:
        if self.current_frame == 0:
            return self.config.num_init_steps
        return self.config.num_incr_steps

    @property
    def is_first_step(self) -> bool:
        return self.global_step == 1  # range from 1

    @property
    def is_last_step(self) -> bool:
        return self.global_step == self.num_steps  # range from 1

    @property
    def ckpt_save_path(self) -> Path:
        filename = f"ckpt-{self.current_frame}-{self.global_step}.pth"
        return self.output_dir / filename

    def training_loop(self):
        next_frame = self.datamodule.current_frame + 1
        if self.global_step > 0:  # restored from a checkpoint
            next_frame = self.datamodule.current_frame
        for current_frame in range(next_frame, self.datamodule.num_frames):
            logging.info(f"\n==> Start learning frame {current_frame} ...")
            self.datamodule.current_frame = current_frame
            self.frame_training_loop()

    def frame_training_loop(self):
        train_dataloader = self.datamodule.train_dataloader()
        logging.info(f"Number of views in training dataset: {len(train_dataloader)}")
        cache = self.config.cache_data_on_device
        move_to_device = lambda ts: ts.to(self.device, True)
        fetcher = Fetcher(train_dataloader, cache, move_to_device)

        metrics_ema = {"loss": 0.0, "psnr": 0.0}

        def update_metric(name, value):
            metrics_ema[name] *= 0.6
            metrics_ema[name] += 0.4 * value

        progress_bar = tqdm(
            range(self.global_step, self.num_steps),
            desc=f"Training frame {self.current_frame}",
        )

        for self.global_step in range(self.global_step + 1, self.num_steps + 1):
            self.module.pre_training_step()

            self.module.timer.clock("data").start()
            args = (fetcher.next(), self.global_step)
            self.module.timer.clock("data").stop()
            loss, metrics, render_pkg = self.module.training_step(*args)
            self.module.timer.clock("backward").start()
            loss.backward()
            self.module.timer.clock("backward").stop()

            self.module.timer.clock("progress updating").start()
            update_metric("loss", loss.item())
            update_metric("psnr", metrics["psnr"].item())
            if self.global_step % 10 == 0:
                metrics_pbar = {k: f"{v:.4f}" for k, v in metrics_ema.items()}
                metrics_pbar["gs"] = metrics["gs"]
                progress_bar.set_postfix(metrics_pbar)
                progress_bar.update(10)
            if self.global_step == self.num_steps:
                progress_bar.close()
            self.module.timer.clock("progress updating").stop()

            self.module.post_training_step(render_pkg)

            saving_ckpt_steps = self.config.saving_ckpt_steps + [self.num_steps]
            if self.global_step in saving_ckpt_steps:
                self.save_checkpoint()

            if self.global_step in self.config.testing_steps + [self.num_steps]:
                self.validation_loop()
        self.global_step = 0  # !!!Important

    @torch.no_grad()
    def validation_loop(self):
        torch.cuda.empty_cache()

        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]

        results_list = []
        for loader_idx, loader in enumerate(loaders):
            logging.info(
                f"Number of views in testing dataset {loader_idx}: {len(loader)}"
            )
            results = []
            for idx, viewpoint in enumerate(loader):
                viewpoint = viewpoint.to(self.device)
                metrics = self.module.validation_step(viewpoint, idx, loader_idx)
                results.append(metrics)
            results_list.append(results)

        num_loaders = len(loaders)
        if num_loaders == 1:
            results_list = results_list[0]
        self.module.validation_end(results_list, num_loaders)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def save_checkpoint(self):
        interval = self.config.saving_ckpt_every_n_frames
        if self.is_last_frame or self.current_frame % interval == 0:
            return super().save_checkpoint()
