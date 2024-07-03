import os
import torch
import argparse
import numpy as np
import smplx as SMPL
from PIL import Image
from pytorch3d.io import load_obj
import sys

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)


def render_mesh_textured(
    device,
    verts,
    textures,
    verts_uvs,
    faces_uvs,
    faces_vertices,
    image_size=None,
    cam_pos=None,
    azimut=0,
    mesh_rot=None,
    background_color=None,
    output_path=None,
    output_filename=None,
):
    batch_size = 1

    # default image size
    if image_size is None:
        image_size = 512

    # default camera position
    if cam_pos is None:
        cam_pos = torch.tensor([2.0, 0.35, 0])

    # default mesh rotation
    if mesh_rot is None:
        mesh_rot = 0

    # default background color
    if background_color is None:
        background_color = (1.0, 1.0, 1.0)

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
        ambient_color=[[1, 1, 1]],
        diffuse_color=[[0, 0, 0]],
        specular_color=[[0, 0, 0]],
    )

    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the mesh is facing the +Z direction.
    # So we move the camera by mesh_rotation in the azimuth direction.
    R, T = look_at_view_transform(cam_pos[0], azimut, mesh_rot)
    T[0, 1] += cam_pos[1]
    # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    cameras = OrthographicCameras(device=device, T=T, R=R)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # image_size_inximage_size_in. As we are rendering images for visualization purposes only we
    # will set faces_per_pixel=1 and blur_radius=0.0. We also set bin_size and max_faces_per_bin to
    # None which ensure that the faster coarse-to-fine rasterization method is used. Refer to
    # rasterize_meshes.py for explanations of these parameters. Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    blend_params = BlendParams(background_color=background_color)

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


def render_360_gif(
    device,
    verts,
    current_image_np,
    verts_uvs,
    faces_uvs,
    verts_idx,
    output_filename_path,
):
    """
    renders a 360 video in T pose using the texture from current_image_np
    """

    rotation_offset = 10
    images_for_gif = []

    for mesh_rot in np.arange(0, 360, rotation_offset):
        current_im = render_mesh_textured(
            device,
            verts,
            current_image_np,
            verts_uvs,
            faces_uvs,
            verts_idx,
            image_size=512,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=mesh_rot,  # mesh rotation in Y axis in degrees
        )
        images_for_gif.append(current_im)

    images_for_gif[0].save(
        output_filename_path,
        save_all=True,
        append_images=images_for_gif[1:],
        optimize=False,
        duration=90,
        loop=0,
    )


class Render360:

    def __init__(self) -> None:

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else: 
            self.device = "cpu"

        #   creates a SMPL instance, and samples it T pose
        smpl_path = '../sample_data/SMPL/models/'
        print(smpl_path)
        self.smpl = SMPL.create(smpl_path,model_type='smpl' ,gender='female')

        self.body_pose = torch.zeros(1,69)
        self.betas = torch.zeros(1,10)
        self.body_pose[0,47] = -1.35   # for A pose
        self.body_pose[0,50] =  1.30   # for A pose

        self.smpl_output = self.smpl( betas=self.betas, 
                            body_pose=self.body_pose, 
                            return_verts=True)

        self.verts = self.smpl_output.vertices[0]
        self.faces = self.smpl.faces_tensor

        #   loads the SMPL template mesh to extract UV coordinates for the textures
        mesh_filename = "../sample_data/smpl_uv.obj"
        _, self.faces_verts, aux = load_obj(mesh_filename)
        self.verts_uvs = aux.verts_uvs[None, ...]        # (1, F, 3)
        self.faces_uvs = self.faces_verts.textures_idx[None, ...]   # (1, F, 3)

    def render_textures(self, textures_folder):

        #   extract list of texture files
        if os.path.exists(textures_folder):
            files = os.listdir(textures_folder)
        else:
            print("ERROR: ", textures_folder, " does not exit")

        for idx, current_file in enumerate(files):
            if idx > 1:
                break
            current_texture_path = os.path.join(textures_folder, current_file)
            print('\nProcessing image ', current_texture_path)

            if ".jpg" in current_texture_path or ".png" in current_texture_path:
                with Image.open(current_texture_path) as image:
                    current_image_np = np.asarray(image.convert("RGB")).astype(np.float32)

                print(current_texture_path)
                render_360_gif(self.device, self.verts, 
                        current_image_np, self.verts_uvs, 
                        self.faces_uvs, self.faces_verts.verts_idx, 
                        current_texture_path.replace(".png", "-360.gif"))

parser = argparse.ArgumentParser(description= 'Renders SMPL 360 gifs given input textures')
parser.add_argument('--textures', type=str, help='Folder with textures', required=False)

args = parser.parse_args()

INPUT_FOLDER = args.textures

render = Render360()
render.render_textures('../data/textures/train/')