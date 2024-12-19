import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from libgs.data import Dataset, SceneInfo, load_scene
from libgs.pipeline import DataConfig as BaseDataConfig
from libgs.pipeline import DataModule as BaseDataModule


@dataclass
class DataConfig(BaseDataConfig):
    root: str = ""
    resolution: int = -1
    white_background: bool = False
    split_train_test: bool = True
    shuffle: bool = True
    eval_train: bool = True
    eval_test: bool = True
    extra_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)


class DataModule(BaseDataModule):
    scene: SceneInfo
    eval_names: List[str]
    current_frame: int = -1

    def __init__(self, config: DataConfig):
        super().__init__(config)

        args = (Path(config.root), config.resolution)
        kwargs = {
            "split_train_test": config.split_train_test,
            "white_background": config.white_background,
            **config.extra_dataset_kwargs,
        }
        self.scene = load_scene(*args, **kwargs)

    @property
    def cameras_extent(self) -> float:
        return self.scene.nerf_normalization["radius"]

    @property
    def num_frames(self) -> int:
        return self.scene.train_dataset.num_frames

    def setup(self, save_dir: Optional[Path] = None):
        self.scene.train_dataset.setup()
        if self.scene.test_dataset is not None:
            self.scene.test_dataset.setup()
        if self.config.shuffle:  # Multi-res consistent random shuffling
            self.scene.train_dataset.shuffle()
            if self.scene.test_dataset is not None:
                self.scene.test_dataset.shuffle()
        if save_dir:
            self.save_scene_info(save_dir)

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["current_frame"] = self.current_frame

    def on_load_checkpoint(self, ckpt: dict):
        if "current_frame" in ckpt:
            self.current_frame = ckpt["current_frame"]

    def save_scene_info(self, save_dir: Path):
        if not self.scene.ply_path.exists():
            print(f"[WARNING] ply file not exists: {self.scene.ply_path}")
            return

        input_ply_path = save_dir / "input.ply"
        input_ply_path.write_bytes(self.scene.ply_path.read_bytes())

        json_cams, camlist = [], []
        if self.scene.test_dataset:
            camlist.extend(self.scene.test_dataset.items)
        if self.scene.train_dataset:
            camlist.extend(self.scene.train_dataset.items)

        json_cams = [camera.to_json(id=id) for id, camera in enumerate(camlist)]
        with (save_dir / "cameras.json").open("w") as file:
            json.dump(json_cams, file)

    def get_train_dataset(self, scale: float = 1.0) -> Dataset:
        train_dataset = self.scene.train_dataset
        if self.current_frame >= 0:
            train_dataset = train_dataset.get_frame_dataset(self.current_frame)

        if scale != 1:
            train_dataset = deepcopy(train_dataset)
        train_dataset.resolution_scale = scale
        return train_dataset

    def get_test_dataset(self, scale: float = 1.0) -> Optional[Dataset]:
        test_dataset = self.scene.test_dataset
        if self.current_frame >= 0:
            test_dataset = test_dataset.get_frame_dataset(self.current_frame)
        if test_dataset:
            if scale != 1:
                test_dataset = deepcopy(test_dataset)
            test_dataset.resolution_scale = scale
        return test_dataset

    def train_dataloader(
        self,
        scale: float = 1.0,
        random: bool = True,
        indices: Optional[Sequence[int]] = None,
    ) -> DataLoader:
        dataset = self.get_train_dataset(scale)
        if indices:
            dataset = Subset(dataset, indices)
        collate_fn = lambda batch: batch[0]
        kwargs = dict(num_workers=8, pin_memory=True, collate_fn=collate_fn)
        if random:
            kwargs["sampler"] = RandomSampler(dataset)
        else:
            kwargs["sampler"] = SequentialSampler(dataset)
        kwargs = {**kwargs, **self.config.extra_dataloader_kwargs}
        return DataLoader(dataset, **kwargs)

    def test_dataloader(self, scale: float = 1.0) -> Optional[DataLoader]:
        if dataset := self.get_test_dataset(scale):
            collate_fn = lambda batch: batch[0]
            kwargs = dict(num_workers=8, pin_memory=True, collate_fn=collate_fn)
            kwargs = {**kwargs, **self.config.extra_dataloader_kwargs}
            return DataLoader(dataset, **kwargs)

    def val_dataloader(self, scale: float = 1.0) -> List[DataLoader]:
        datasets, eval_names = [], []
        if self.config.eval_train:
            datasets.append(self.train_dataloader(scale, random=False))
            eval_names.append("train")
        if self.config.eval_test and (dataset := self.test_dataloader(scale)):
            datasets.append(dataset)
            eval_names.append("test")
        self.eval_names = eval_names
        return datasets
