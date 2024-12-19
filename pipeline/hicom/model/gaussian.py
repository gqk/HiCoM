from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from libgs.model.gaussian import GaussianModel as BaseGaussianModel
from libgs.utils.general import build_rotation, quaternion_multiply


class GaussianModel(BaseGaussianModel):
    delta_xyz: Optional[Tensor]
    delta_rot: Optional[Tensor]
    noise_scale: Optional[float]

    @property
    def get_xyz(self) -> Float[Tensor, "n 3"]:
        xyz = super().get_xyz
        if (delta_xyz := getattr(self, "delta_xyz", None)) is not None:
            return xyz + delta_xyz
        if (noise_scale := getattr(self, "noise_scale", None)) is not None:
            return xyz + torch.randn_like(xyz.detach()) * noise_scale
        return xyz

    @property
    def get_rotation(self) -> Float[Tensor, "n 3"]:
        if (delta_rot := getattr(self, "delta_rot", None)) is not None:
            return quaternion_multiply(super().get_rotation, delta_rot)
        return super().get_rotation

    @property
    def get_xyz_ori(self) -> Float[Tensor, "n 3"]:
        return super().get_xyz

    @property
    def get_rotation_ori(self) -> Float[Tensor, "n 3"]:
        return super().get_rotation

    def add_densification_stats(
        self,
        viewspace_point_tensor: Float[Tensor, "n 2"],
        update_filter: Bool[Tensor, "n"],
        align: Literal["left", "right"] = "left",
    ):
        grad = viewspace_point_tensor.grad
        if grad.shape[0] != update_filter.shape[0]:
            if align == "left":
                grad = grad[: update_filter.shape[0]]
            else:
                grad = grad[-update_filter.shape[0] :]

        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def adaptive_densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        scene_extent: float,
        max_screen_size: Optional[float] = None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.adaptive_densify_and_split(grads, max_grad, scene_extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def adaptive_densify_and_clone(
        self,
        grad_threshold: float,
        scene_extent: float,
        sigma_scale: float = 2.0,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        means = self._xyz[selected_pts_mask]
        stds = self.get_scaling[selected_pts_mask]
        new_xyz = torch.normal(means, stds * sigma_scale)
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        # Set opacity to 0.1
        new_opacities = torch.zeros_like(self._opacity[selected_pts_mask]) + 0.1
        new_scaling = self._scaling[selected_pts_mask]
        # Set rotation to identity quaternion
        new_rotation = torch.zeros_like(self._rotation[selected_pts_mask])
        new_rotation[:, 0] = 1

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def adaptive_densify_and_split(
        self,
        grads: Float[Tensor, "n 2"],
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        options = dict(device="cuda", dtype=bool)
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), **options),
            )
        )
        self.prune_points(prune_filter)

    @torch.no_grad()
    def apply_deformation(
        self,
        delta_xyz: Float[Tensor, "n 3"],
        delta_rotation: Float[Tensor, "n 4"],
    ):
        xyz = self.get_xyz + delta_xyz
        rotation = quaternion_multiply(self.get_rotation, delta_rotation)
        try:
            self._xyz = self.replace_tensor_to_optimizer(xyz, "xyz")["xyz"]
            tensors = self.replace_tensor_to_optimizer(rotation, "rotation")
            self._rotation = tensors["rotation"]
        except TypeError:  # state stored in optimizer is None
            self._xyz, self._rotation = xyz, rotation

    @torch.no_grad()
    def extend(self, gaussians: BaseGaussianModel) -> "GaussianModel":
        self.densification_postfix(
            gaussians._xyz.clone(),
            gaussians._features_dc.clone(),
            gaussians._features_rest.clone(),
            gaussians._opacity.clone(),
            gaussians._scaling.clone(),
            gaussians._rotation.clone(),
        )
        return self

    def prune_points(self, mask_or_indices: Tensor):
        if mask_or_indices.dtype is torch.bool:
            return super().prune_points(mask_or_indices)
        mask = torch.zeros_like(self._xyz[:, 0], dtype=torch.bool)
        mask[mask_or_indices] = True
        return super().prune_points(mask)
