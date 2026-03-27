"""
Pure-Python / NumPy implementation of the Aria Fisheye624 camera model
(FisheyeRadTanThinPrism<numK=6, useTangential=true, useThinPrism=true,
 useSingleFocalLength=true>).

Parameter vector layout (15 values):
  [0]     f          – single focal length (fu == fv)
  [1]     cu         – principal point col  (x)
  [2]     cv         – principal point row  (y)
  [3-8]   k0..k5     – radial distortion coefficients
  [9]     p0         – tangential distortion
  [10]    p1         – tangential distortion
  [11-14] s0..s3     – thin-prism distortion

Projection formula (from FisheyeRadTanThinPrism.h):

  a, b     = x/z, y/z
  r        = sqrt(a²+b²)
  θ        = atan(r)
  th_rad   = 1 + k0·θ² + k1·θ⁴ + k2·θ⁶ + k3·θ⁸ + k4·θ¹⁰ + k5·θ¹²
  th_divr  = θ/r   (→ 1 as r→0)
  xr, yr   = th_rad·th_divr·a,  th_rad·th_divr·b
  rd²      = xr² + yr²

  tangential:
    u += (2·xr·p0 + 2·yr·p1)·xr + rd²·p0
    v += (2·xr·p0 + 2·yr·p1)·yr + rd²·p1

  thin-prism:
    u += s0·rd² + s1·rd⁴
    v += s2·rd² + s3·rd⁴

  pixel = f · [u, v] + [cu, cv]

The Aria RGB camera is physically mounted rotated 90° CW relative to the
displayed image.  After projecting with the raw model, apply rotate_aria_pixels
to get display-frame coordinates.
"""

import math
import numpy as np
import torch


# ---------------------------------------------------------------------------
# NumPy (CPU) implementation
# ---------------------------------------------------------------------------

def project_fisheye624_np(points_cam: np.ndarray, params: np.ndarray) -> tuple:
    """
    Project 3-D camera-frame points with Fisheye624.

    Args:
        points_cam : (..., 3)   float64 points in the camera frame
        params     : (15,)      Fisheye624 parameter vector

    Returns:
        pixels : (..., 2)   projected pixel coords [u (col), v (row)] in the
                            *original* (non-rotated) sensor frame
        valid  : (...,)     bool – True when the point is within FOV (solid
                            angle ≤ π/2) AND in front of the camera (z > 0)
    """
    params = np.asarray(params, dtype=np.float64)
    points_cam = np.asarray(points_cam, dtype=np.float64)

    f  = params[0]
    cu = params[1];  cv = params[2]
    k  = params[3:9]            # k0..k5
    p0 = params[9];  p1 = params[10]
    s  = params[11:15]          # s0..s3

    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]

    eps = np.finfo(np.float64).eps

    inv_z = 1.0 / z
    a = x * inv_z
    b = y * inv_z

    r_sq = a * a + b * b
    r    = np.sqrt(r_sq)
    th   = np.arctan(r)
    th_sq = th * th

    # radial polynomial: 1 + Σ kᵢ · θ^(2i+2)
    th_radial = np.ones_like(r)
    th2i = th_sq.copy()
    for ki in k:
        th_radial += ki * th2i
        th2i = th2i * th_sq

    # θ/r, l'Hôpital limit → 1 at r=0
    th_divr = np.where(r < eps, 1.0, th / r)

    xr = th_radial * th_divr * a
    yr = th_radial * th_divr * b
    rd_sq = xr * xr + yr * yr
    rd_4  = rd_sq * rd_sq

    u = xr.copy()
    v = yr.copy()

    # tangential distortion
    tmp = 2.0 * (xr * p0 + yr * p1)
    u += tmp * xr + rd_sq * p0
    v += tmp * yr + rd_sq * p1

    # thin-prism distortion
    u += s[0] * rd_sq + s[1] * rd_4
    v += s[2] * rd_sq + s[3] * rd_4

    # focal length + principal point
    u = f * u + cu
    v = f * v + cv

    pixels = np.stack([u, v], axis=-1)

    # validity: z > 0 and solid angle ≤ π/2
    solid_angle = np.arctan2(np.sqrt(x * x + y * y), z)
    valid = (z > 0) & (solid_angle <= (math.pi / 2))

    return pixels, valid


def rotate_aria_pixels_np(
    pixels: np.ndarray,
    image_size: int = 224,
    orig_size: int = 2880,
) -> np.ndarray:
    """
    Convert pixel coords from the raw Aria RGB sensor frame to the 90°-CW-
    rotated display frame, then scale to `image_size`.

    Raw → rotated:  rotated_u = (orig_size-1) - v_raw
                    rotated_v = u_raw
    Then scale by image_size / orig_size.

    Args:
        pixels    : (..., 2)  [u (col), v (row)] in raw sensor frame
        image_size: target square image resolution (default 224)
        orig_size : raw sensor square resolution (default 2880 for Aria RGB)

    Returns:
        rotated : (..., 2)  [col, row] in scaled display frame
    """
    u = pixels[..., 0]
    v = pixels[..., 1]
    rotated = np.stack([(orig_size - 1) - v, u], axis=-1)
    return rotated * (image_size / orig_size)


# ---------------------------------------------------------------------------
# Torch (GPU-friendly) implementation
# ---------------------------------------------------------------------------

def project_fisheye624_torch(
    points_cam: torch.Tensor,
    params: torch.Tensor,
) -> tuple:
    """
    Differentiable Fisheye624 projection in PyTorch.

    Args:
        points_cam : (..., 3)   float tensor in camera frame
        params     : (15,)      Fisheye624 parameter tensor

    Returns:
        pixels : (..., 2)   [u, v] in raw sensor frame
        valid  : (...,)     bool mask (solid angle ≤ π/2 and z > 0)
    """
    f  = params[0]
    cu = params[1];  cv = params[2]
    k  = params[3:9]
    p0 = params[9];  p1 = params[10]
    s  = params[11:15]

    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]

    eps = torch.finfo(points_cam.dtype).eps

    inv_z = 1.0 / z
    a = x * inv_z
    b = y * inv_z

    r_sq  = a * a + b * b
    r     = torch.sqrt(r_sq)
    th    = torch.atan(r)
    th_sq = th * th

    th_radial = torch.ones_like(r)
    th2i = th_sq.clone()
    for ki in k:
        th_radial = th_radial + ki * th2i
        th2i = th2i * th_sq

    th_divr = torch.where(r < eps, torch.ones_like(r), th / r)

    xr = th_radial * th_divr * a
    yr = th_radial * th_divr * b
    rd_sq = xr * xr + yr * yr
    rd_4  = rd_sq * rd_sq

    u = xr.clone()
    v = yr.clone()

    tmp = 2.0 * (xr * p0 + yr * p1)
    u = u + tmp * xr + rd_sq * p0
    v = v + tmp * yr + rd_sq * p1

    u = u + s[0] * rd_sq + s[1] * rd_4
    v = v + s[2] * rd_sq + s[3] * rd_4

    u = f * u + cu
    v = f * v + cv

    pixels = torch.stack([u, v], dim=-1)

    solid_angle = torch.atan2(torch.sqrt(x * x + y * y), z)
    valid = (z > 0) & (solid_angle <= (math.pi / 2))

    return pixels, valid


def rotate_aria_pixels_torch(
    pixels: torch.Tensor,
    image_size: int = 224,
    orig_size: int = 2880,
) -> torch.Tensor:
    """Torch equivalent of rotate_aria_pixels_np."""
    u = pixels[..., 0]
    v = pixels[..., 1]
    rotated = torch.stack([(orig_size - 1) - v, u], dim=-1)
    return rotated * (image_size / orig_size)


