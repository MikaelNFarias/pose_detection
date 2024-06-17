import torch
import numpy as np
import torchvision
import sys
import cv2 as cv
from PIL import Image
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
        image_size: int = None,
        cam_pos: np.ndarray | torch.Tensor = None,
        background: np.ndarray | torch.Tensor = None,
        output_path: str = None,
        output_filename: str = None,
        up: Sequence = None,
        at: np.ndarray = None,
        background_image: np.ndarray | torch.Tensor = None,
        eye_position: Sequence[float] = None,
        fov: float = 60.0,
) -> Any:
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # default image size
    try:
        at_position = at.tolist()
    except:
        at_position = at
    #mean_position = mean_position.tolist()

    if background_image is not None:
        background_image = cv.imread(background_image)

    if up is None:
        up = [0, 0, 1]

    if image_size is None:
        image_size = 512

    # default camera position
    if cam_pos is None:
        cam_pos = torch.tensor([.5, 3, 1.2])

    # default background color


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
            location=[eye_position],  # [at_position],
            ambient_color=[[1., 1., 1.]],
            diffuse_color=[[0.002, 0.002, 0.002]],
            specular_color=[[0., 0., 0.]],
        )
    except Exception as e:
        print(eye_position)
        sys.exit(0)

    try:
        if True:
            R, T = look_at_view_transform(eye=[eye_position],
                                          at=[at_position],
                                          up=[up])
    except Exception as e:
        print("Rotation and translation could not be computed")
        sys.exit(1)

    try:
        cameras = FoVPerspectiveCameras(device=device,
                                        fov=fov,
                                        degrees=True,
                                        R=R,
                                        T=T)
    except Exception as e:
        print(e)



    if background is None:
        background_color = (1.0, 1.0, 1.0)
        blend_params = BlendParams(background_color=background_color)
    else:
        background_color = background
        blend_params = BlendParams(background_color=background_color / 255.0)

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model

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
    rgbArray = np.zeros((image_size, image_size, 3), "uint8")
    rgbArray[..., 0] = (R_channel * 255).astype(int)
    rgbArray[..., 1] = (G_channel * 255).astype(int)
    rgbArray[..., 2] = (B_channel * 255).astype(int)
    img = Image.fromarray(rgbArray)

    if output_filename is not None:
        print("Saving ", os.path.join(output_path, output_filename), "\n")
        img.save(os.path.join(output_path, output_filename))
        return eye_position, at
    else:
        return img
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.pause(1)
