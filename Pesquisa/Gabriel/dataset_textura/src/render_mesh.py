import torch
import numpy as np
import torchvision
import sys
import cv2 as cv
from PIL import Image
import os
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
    textures : np.ndarray | torch.Tensor,
    verts_uvs : np.ndarray | torch.Tensor,
    faces_uvs : np.ndarray | torch.Tensor,
    faces_vertices : np.ndarray | torch.Tensor,
    image_size :int = None,
    cam_pos : np.ndarray | torch.Tensor = None,
    azimut : int = 0,
    mesh_rot: int = None,
    background : np.ndarray | torch.Tensor = None,
    output_path: str = None,
    output_filename: str = None,
    up = None,
    at = None,
    cam_dist: float = 1.0,
    x_axis_weight: float = 1.0,
    y_axis_weight: float = 1.0,
    z_axis_weight: float = 1.0,
    background_image = None,
):
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # default image size
    at_position = at.tolist()
    #mean_position = mean_position.tolist()
    eye_position = [
        at_position[0] + x_axis_weight * (cam_dist * np.cos(np.deg2rad(azimut))),
        at_position[1] + y_axis_weight * (cam_dist * np.sin(np.deg2rad(azimut))),
        at_position[2],
    ]


    if background_image is not None:
        background_image = cv.imread(background_image)

    if up is None:
        up = ((0, 0, 1),)

    if image_size is None:
        image_size = 512

    # default camera position
    if cam_pos is None:
        cam_pos = torch.tensor([.5, 3, 1.2])

    # default mesh rotation
    if mesh_rot is None:
        mesh_rot = 0

    # default background color

    
    up = ((0, 0, 1),)

    tex = torch.from_numpy(textures / 255.0)[None].to(device)
    textures_rgb = TexturesUV(
        maps=tex, faces_uvs=faces_uvs.to(device), verts_uvs=verts_uvs.to(device)
    )

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_vertices.to(device)],
        textures=textures_rgb,
    )

    lights = PointLights(
        device=device,
        location = [at_position],
        ambient_color=[[1, 1, 1]],
        diffuse_color=[[0, 0, 0]],
        specular_color=[[0, 0, 0]],
    )
    # if orientation == 'frontal':
        # frontal = True
        # side = False
    # else:
        # side = True
        # frontal = False

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the mesh is facing the +Z direction.
    # So we move the camera by mesh_rotation in the azimuth direction.

    #R,T = look_at_view_transform(at,
    if True:
        R, T = look_at_view_transform(eye = [eye_position],
                                      at = [at_position],
                                      up = [[0, 0, 1]])
    ##if side:
    ##    R, T = look_at_view_transform(eye = ((2.5,0.8,1.0),),
    ##                                  at = ((0.5,0.8,1.2),),
    ##                                  up = ((0, 0, 1),))
    #T[0, 1] += cam_pos[1]
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    #cameras = OrthographicCameras(device=device, T=T, R=R)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # image_size_inximage_size_in. As we are rendering images for visualization purposes only we
    # will set faces_per_pixel=1 and blur_radius=0.0. We also set bin_size and max_faces_per_bin to
    # None which ensure that the faster coarse-to-fine rasterization method is used. Refer to
    # rasterize_meshes.py for explanations of these parameters. Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.

    if background is None:
        background_color = (1.0, 1.0, 1.0)
        blend_params = BlendParams(background_color=background_color)
    else:
        background_color = background
        blend_params = BlendParams(background_color=background_color/255.0)

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
    img = Image.fromarray(rgbArray, mode="RGB")

    if output_filename is not None:
        print("Saving ", os.path.join(output_path, output_filename), "\n")
        img.save(os.path.join(output_path, output_filename))
    else:
        return img
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())
    # plt.axis("off")
    # plt.pause(1)
