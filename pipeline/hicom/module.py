from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
from absl import logging
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.data.types import TensorSpace
from libgs.metric import psnr, ssim
from libgs.metric.lpips import LPIPS
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig
from libgs.renderer.network_gui import interact_with_gui
from libgs.utils.time import Timer

from .data import DataModule
from .model.deformation import Deformation, DeformConfig
from .model.gaussian import GaussianModel


@dataclass
class GaussianConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 15000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01


@dataclass
class DensifyConfig:
    from_step: int = 500
    until_step: int = 5000
    interval: int = 100
    grad_threshold: float = 2e-4
    opacity_reset_interval: int = 30000
    max_gaussians: int = 10000000


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 3
    random_background: bool = False  # for training
    lambda_dssim: float = 0.2
    lambda_deform: float = 1.0
    noise_scale: Optional[float] = 0.01
    saving_gs_steps: List[int] = field(default_factory=lambda: [5000])
    saving_gs_every_n_frames: int = 1000
    num_saving_images: int = 5
    full_eval: bool = False
    merge_to_base: bool = True
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    deformation: DeformConfig = field(default_factory=DeformConfig)


class Module(BaseModule):
    datamodule: DataModule
    trainer: "Trainer"
    gaussians: GaussianModel
    background: torch.Tensor  # TODO remove
    timer: Timer

    @property
    def current_frame(self) -> int:
        return self.datamodule.current_frame

    @property
    def num_steps(self) -> int:
        return self.trainer.num_steps

    def setup(self):
        white_background = self.datamodule.config.white_background
        bg_color_fn = torch.ones if white_background else torch.zeros
        self.register_buffer("background", bg_color_fn(3, device=self.device))

        if self.config.full_eval:
            self.lpips_fn = LPIPS("vgg").to(self.device)

        self.gaussians = GaussianModel(self.config.sh_degree)
        self.gaussians.create_from_pcd(
            self.datamodule.scene.point_cloud, self.datamodule.cameras_extent
        )
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.renderer = Renderer(self.config.renderer, self.gaussians)
        self.deform = Deformation(self.config.deformation)
        self.timer = Timer()

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["gaussians"] = self.gaussians.capture()
        if not self.trainer.is_first_frame:
            ckpt["gaussians_incr"] = self.gaussians_incr.capture()
        ckpt["deformation"] = self.deform.capture()

    def on_load_checkpoint(self, ckpt: dict):
        self.gaussians.restore(ckpt["gaussians"], self.config.gaussian)
        if not self.trainer.is_first_frame:
            self.gaussians_incr.restore(
                ckpt["gaussians_incr"], self.config.gaussian_stage2
            )
        self.deform.restore(ckpt["deformation"])

    def forward(self, viewpoint: TensorSpace, training: bool = False, **kwargs) -> dict:
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)

        results = {}
        if not self.trainer.is_first_frame:
            num_stage1_steps = self.deform.config.num_stage1_steps
            enable_deform = False
            if self.global_step < num_stage1_steps:
                enable_deform = True
            elif self.global_step == num_stage1_steps and training:
                enable_deform = True

            if enable_deform:
                delta_xyz, delta_rot = self.deform(self.gaussians.get_xyz_ori)
                results["delta_xyz"] = delta_xyz
                results["delta_rotation"] = delta_rot
                self.gaussians.delta_xyz = delta_xyz
                self.gaussians.delta_rot = delta_rot
        elif training:
            self.gaussians.noise_scale = self.config.noise_scale

        results.update(self.renderer(viewpoint, bg_color, **kwargs))
        for attr in ["delta_xyz", "delta_rot", "noise_scale"]:
            setattr(self.gaussians, attr, None)
        return results

    def pre_training_step(self):
        interact_with_gui(
            self.global_step,
            self.renderer.config,  # can be modified
            self,
            self.datamodule.config.root,
            self.num_steps,
        )

        if self.trainer.is_first_step:
            self.timer.clock("frame").start()

        if not self.trainer.is_first_frame and self.trainer.is_first_step:
            self.timer.clock("frame setup").start()
            num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
            if self.config.merge_to_base and num_incr_gs > 0:
                logging.info(f"Merge and prune {num_incr_gs} gaussians")
                self.gaussians.extend(self.gaussians_incr)
                indices = self.gaussians.get_opacity.flatten().sort().indices
                self.gaussians.prune_points(indices[:num_incr_gs])

            reset_grid = self.current_frame == 1
            self.deform.setup(self.gaussians.get_xyz.detach(), reset_grid)
            self.deform.to(self.device)
            self.gaussians_incr = GaussianModel(self.config.sh_degree)
            logging.info("Reset gaussians of renderer")
            self.renderer.gaussians = self.gaussians
            self.timer.clock("frame setup").stop()

        num_stage2_steps = self.global_step - self.deform.config.num_stage1_steps
        if not self.trainer.is_first_frame and num_stage2_steps == 1:
            self.gaussians_incr.training_setup(self.config.gaussian_stage2)
            self.renderer.gaussians = MergedGaussianModel(
                [self.gaussians, self.gaussians_incr]
            )

    def training_step(
        self, viewpoint: TensorSpace, current_step: int
    ) -> Tuple[Tensor, dict, dict]:
        self.timer.clock("training step").start()

        num_stage1_steps = self.deform.config.num_stage1_steps
        if self.trainer.is_first_frame:
            self.gaussians.update_learning_rate(current_step)
        elif (current_stage2_step := current_step - num_stage1_steps) > 0:
            self.gaussians_incr.update_learning_rate(current_stage2_step)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if current_step % 1000 == 0 and self.trainer.is_first_frame:
            self.gaussians.oneupSHdegree()

        # Render
        if (current_step - 1) == self.renderer.config.debug_from:
            self.renderer.config.debug = True

        render_pkg = self.forward(viewpoint, training=True)
        image = render_pkg["render"]
        gt_image = viewpoint.image
        loss_rgb = l1_loss(image, gt_image)
        loss_dssim = 1.0 - ssim(image, gt_image)

        lambda_dssim = self.config.lambda_dssim
        loss = (1.0 - lambda_dssim) * loss_rgb + lambda_dssim * loss_dssim

        metrics = {
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "psnr": psnr(image, gt_image).mean().double(),
            "gs": self.renderer.gaussians.get_xyz.shape[0],
        }

        if "delta_xyz" in render_pkg and self.config.lambda_deform > 0:
            loss_reg = self.deform.reg_loss()
            loss = loss + self.config.lambda_deform * loss_reg
            metrics["loss/reg"] = loss_reg

        self.timer.clock("training step").stop()

        self.timer.clock("tensorboard logging").start()
        self.log_dict(metrics)
        self.timer.clock("tensorboard logging").stop()

        return loss, metrics, render_pkg

    @torch.no_grad()
    def post_training_step_init(self, render_pkg):
        if self.global_step < self.config.densify.until_step:
            # Densification, depend on grad
            self.densify_gaussians(
                self.global_step,
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step_incr(self, render_pkg):
        if self.global_step < self.config.densify.until_step:
            self.adaptive_densify_gaussians(
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        if self.global_step > self.deform.config.num_stage1_steps:
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)
        elif render_pkg.get("delta_xyz", None) is not None:
            self.deform.optimizer.step()
        self.deform.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step(self, render_pkg: dict):
        self.timer.clock("post training step").start()
        if self.trainer.is_first_frame:
            self.post_training_step_init(render_pkg)
        else:
            self.post_training_step_incr(render_pkg)
        self.timer.clock("post training step").stop()

        self.save_point_cloud()

        if self.trainer.is_last_step:
            self.timer.clock("frame").stop()
            if not self.trainer.is_first_frame:
                num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
                logging.info(f"Numer of increment gaussians: {num_incr_gs}")
            time_stats = self.timer.display()
            logging.info(f"Time stats at {self.global_step} step:\n{time_stats}\n")
            self.timer.reset()

    def validation_step(
        self, viewpoint: TensorSpace, idx: int, loader_idx: int
    ) -> dict:
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        gt_image = viewpoint.image
        if idx <= self.config.num_saving_images:
            root = self.datamodule.config.root
            path = self.output_dir / f"image" / viewpoint.path.relative_to(root)
            save_dir, name, ext = path.parent, path.stem, path.suffix
            save_dir.mkdir(parents=True, exist_ok=True)
            save_image(image, save_dir / f"{name}_step{self.global_step}{ext}")

            testing_steps = self.trainer.config.testing_steps + [self.num_steps]
            if self.global_step == testing_steps[0]:
                save_image(gt_image, save_dir / f"{name}_gt{ext}")

        metrics = dict(l1=l1_loss(image, gt_image), psnr=psnr(image, gt_image))
        if self.config.full_eval:
            metrics["ssim"] = ssim(image, gt_image)
            metrics["lpips"] = self.lpips_fn(image, gt_image).mean()
        return metrics

    def validation_end(self, results: Union[dict, List[dict]], num_loaders: int = 1):
        def fn(metrics_list, idx):
            mean = lambda xs: sum(xs) / len(xs)
            metrics = {k: mean([ms[k] for ms in metrics_list]) for k in metrics_list[0]}
            name = self.datamodule.eval_names[idx]
            self.log_dict({f"eval-{name}/{k}": v for k, v in metrics.items()})
            logging.info(
                f"Evaluate {name} dataset at step {self.global_step}:\n\t"
                + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            )

        results = [results] if num_loaders == 1 else results
        for idx, result in enumerate(results):
            fn(result, idx)

        self.log_histogram("scene/opacity_histogram", self.gaussians.get_opacity)
        self.log("total_points", self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        skiping = self.current_frame % self.config.saving_gs_every_n_frames > 0
        if not self.trainer.is_last_frame and skiping:
            return

        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.current_frame}-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        logging.info(f"\nSave gaussians to {save_path}")
        if self.trainer.is_first_frame:
            self.gaussians.save_ply(save_path)
        else:
            self.gaussians_incr.save_ply(save_path)

        if not self.trainer.is_first_frame and self.trainer.is_last_step:
            filename = f"initial-gaussians-{self.current_frame}.ply"
            save_path = self.output_dir / "point_cloud" / filename
            logging.info(f"Save deformed initial gaussians to {save_path}")

    def densify_gaussians(
        self,
        current_step: int,
        visibility_filter: Float[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(
            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter
        )

        opacity_reset_interval = self.config.densify.opacity_reset_interval
        max_gaussians = self.config.densify.max_gaussians

        densify_started = current_step > self.config.densify.from_step
        at_interval = current_step % self.config.densify.interval == 0
        within_limit = self.gaussians.get_xyz.shape[0] < max_gaussians
        if densify_started and at_interval and within_limit:
            exceed_reset_inverval = current_step > opacity_reset_interval
            size_threshold = 20 if exceed_reset_inverval else None
            self.gaussians.densify_and_prune(
                self.config.densify.grad_threshold,
                0.005,
                self.datamodule.cameras_extent,
                size_threshold,
            )

        white_background = self.datamodule.config.white_background
        at_reset_interval = current_step % opacity_reset_interval == 0
        at_densify_start = current_step == self.config.densify.from_step
        if at_reset_interval or (white_background and at_densify_start):
            self.gaussians.reset_opacity()

    def adaptive_densify_gaussians(
        self,
        visibility_filter: Bool[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        num_init_gs = self.gaussians.get_xyz.shape[0]
        num_stage1_steps = self.deform.config.num_stage1_steps
        current_stage2_step = self.global_step - num_stage1_steps

        # Keep track of max radii in image-space for pruning
        visibility_filter_init = visibility_filter[:num_init_gs]
        radii_init = radii[:num_init_gs]
        self.gaussians.max_radii2D[visibility_filter_init] = torch.max(
            self.gaussians.max_radii2D[visibility_filter_init],
            radii_init[visibility_filter_init],
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter_init, align="left"
        )

        if current_stage2_step > 0:
            visibility_filter_incr = visibility_filter[num_init_gs:]
            radii_incr = radii[num_init_gs:]
            self.gaussians_incr.max_radii2D[visibility_filter_incr] = torch.max(
                self.gaussians_incr.max_radii2D[visibility_filter_incr],
                radii_incr[visibility_filter_incr],
            )
            self.gaussians_incr.add_densification_stats(
                viewspace_point_tensor, visibility_filter_incr, align="right"
            )

        if self.global_step == num_stage1_steps:
            logging.info("Apply deformation to previous gaussians")
            delta_xyz, delta_rotation = self.deform(self.gaussians.get_xyz.detach())
            self.gaussians.apply_deformation(delta_xyz, delta_rotation)
            self.gaussians.adaptive_densify_and_clone(
                self.deform.config.densify_grad_threshold,
                self.datamodule.cameras_extent,
            )
            self.gaussians_incr.restore(
                self.gaussians.capture(), self.config.gaussian_stage2
            )
            mask = torch.zeros_like(self.gaussians.get_xyz[:, 0], dtype=torch.bool)
            mask[num_init_gs:] = True
            self.gaussians.prune_points(mask)
            self.gaussians_incr.prune_points(~mask)

        densify_interval = self.deform.config.densify_interval
        if current_stage2_step > 0 and current_stage2_step % densify_interval == 0:
            self.gaussians_incr.adaptive_densify_and_prune(
                self.deform.config.densify_grad_threshold,
                self.deform.config.opacity_threshold,
                self.datamodule.cameras_extent,
                20,  # TODO check
            )
