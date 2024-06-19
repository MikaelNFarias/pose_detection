import pytorch3d.renderer.cameras
import torch
import numpy as np
import torchvision
import sys
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Sequence, List, Any, Tuple
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesUV,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    MeshRenderer,
    OrthographicCameras,
    FoVPerspectiveCameras,
)


def render_mesh_textured(
        verts: np.ndarray | torch.Tensor,
        textures: np.ndarray | torch.Tensor,
        verts_uvs: np.ndarray | torch.Tensor,
        faces_uvs: np.ndarray | torch.Tensor,
        faces_vertices: np.ndarray | torch.Tensor,
        image_size: Tuple[int, int] = (512, 512),
        background: np.ndarray | torch.Tensor = None,
        output_path: str = None,
        output_filename: str = None,
        up: Sequence = None,
        at: np.ndarray = None,
        background_image: np.ndarray | torch.Tensor = None,
        eye_position: Sequence[float] = None,
        fov: float = 60.0,
        landmarks_idx=None,
        draw: bool = False,
) -> Any:
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        at_position = at.tolist()
    except:
        at_position = at

    if background_image is not None:
        background_image = cv.imread(background_image)

    if up is None:
        up = [0, 0, 1]

    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError("image_size must be a tuple of (height, width)")

    height, width = image_size

    # default camera position


    tex = torch.from_numpy(textures / 255.0)[None].to(device)
    textures_rgb = TexturesUV(
        maps=tex, faces_uvs=faces_uvs.to(device), verts_uvs=verts_uvs.to(device)
    )

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_vertices.to(device)],
        textures=textures_rgb,
    )
    try:
        lights = PointLights(
            device=device,
            location=[eye_position],
            ambient_color=[[1., 1., 1.]],
            diffuse_color=[[0.002, 0.002, 0.002]],
            specular_color=[[0., 0., 0.]],
        )
    except Exception as e:
        print(eye_position)
        sys.exit(0)

    try:
        R, T = look_at_view_transform(eye=[eye_position], at=[at_position], up=[up])
    except Exception as e:
        print("Rotation and translation could not be computed")
        sys.exit(1)

    try:
        cameras = FoVPerspectiveCameras(device=device, fov=fov, degrees=True, R=R, T=T)
    except Exception as e:
        print(e)

    if background is None:
        background_color = (1.0, 1.0, 1.0)
        blend_params = BlendParams(background_color=background_color)
    else:
        background_color = background
        blend_params = BlendParams(background_color=background_color / 255.0)

    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    images = renderer(mesh)
    R_channel = images[0, :, :, 0].detach().cpu().numpy()
    G_channel = images[0, :, :, 1].detach().cpu().numpy()
    B_channel = images[0, :, :, 2].detach().cpu().numpy()
    rgbArray = np.zeros((height, width, 3), "uint8")
    rgbArray[..., 0] = (R_channel * 255).astype(int)
    rgbArray[..., 1] = (G_channel * 255).astype(int)
    rgbArray[..., 2] = (B_channel * 255).astype(int)
    img = Image.fromarray(rgbArray)

    if landmarks_idx is not None:
        if draw:
            draw = ImageDraw.Draw(img)
            try:
                for name, idx in landmarks_idx.items():
                    point = verts[idx]
                    x_proj, y_proj = project_point(point, (height, width), device, cameras)
                    draw.ellipse((x_proj - 2, y_proj - 2, x_proj + 2, y_proj + 2), fill='red')
                    draw.text((x_proj + 5, y_proj), name, fill='red')
            except Exception as e:
                print(e)

    if output_filename is not None:
        print("Saving ", os.path.join(output_path, output_filename), "\n")
        img.save(os.path.join(output_path, output_filename))
        return eye_position, at
    else:
        return img


def project_point(point: torch.Tensor,
                  image_size: Tuple[int, int],
                  device: torch.device,
                  cameras: pytorch3d.renderer.cameras.FoVPerspectiveCameras):
    point = point[None, None, :].to(device)  # [1, 1, 3]

    # Transform the point using the camera
    print(f"point is  {point}")
    transformed_point = cameras.transform_points_screen(point,image_size=image_size)
    print(f"Transformed point: {transformed_point}")  # Debugging output

    # Convert the point from world space to NDC space
    test_points = transformed_point.squeeze()
    print(test_points)

    # NDC to screen coordinates
    x_proj = (test_points[..., 0])
    y_proj = (test_points[..., 1])

    print(f"Screen coordinates: ({x_proj}, {y_proj})")  # Debugging output

    return x_proj.item(), y_proj.item()

# Sample usage:
# verts, textures, verts_uvs, faces_uvs, faces_vertices should be defined
# at = torch.tensor([0, 0, 0])
# eye_position = [2, 2, 2]
# landmarks_idx = {"HEAD_TOP": 412}
# render_mesh_textured(verts, textures, verts_uvs, faces_uvs, faces_vertices, at=at, eye_position=eye_position, landmarks_idx=landmarks_idx)
