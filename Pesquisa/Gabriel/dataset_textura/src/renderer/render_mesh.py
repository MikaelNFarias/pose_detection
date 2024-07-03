import pytorch3d.renderer.cameras
import torch
import numpy as np
import torchvision
import sys
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import os
from typing import Sequence, List, Any, Tuple, Union
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



SMPL_IND2JOINT = {
    0: 'pelvis',
    1: 'left_hip',
    2: 'right_hip',
    3: 'spine1',
    4: 'left_knee',
    5: 'right_knee',
    6: 'spine2',
    7: 'left_ankle',
    8: 'right_ankle',
    9: 'spine3',
    10: 'left_foot',
    11: 'right_foot',
    12: 'neck',
    13: 'left_collar',
    14: 'right_collar',
    15: 'head',
    16: 'left_shoulder',
    17: 'right_shoulder',
    18: 'left_elbow',
    19: 'right_elbow',
    20: 'left_wrist',
    21: 'right_wrist',
    22: 'left_hand',
    23: 'right_hand'
}


def render_mesh_textured(
        verts: Union[np.ndarray , torch.Tensor],
        textures: Union[np.ndarray , torch.Tensor],
        verts_uvs: Union[np.ndarray , torch.Tensor],
        faces_uvs: Union[np.ndarray , torch.Tensor],
        faces_vertices: Union[np.ndarray , torch.Tensor],
        image_size: Tuple[int, int] = (512, 512),
        background: Union[np.ndarray , torch.Tensor] = None,
        output_path: str = None,
        output_filename: str = None,
        up: Sequence = None,
        at: np.ndarray = None,
        background_image: Union[np.ndarray , torch.Tensor] = None,
        eye_position: Sequence[float] = None,
        fov: float = 60.0,
        landmarks_idx=None,
        joints=None,
        draw_landmarks: bool = False,
        draw_joints: bool = False,
        extreme_points=None
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

    if joints is not None:
        try:
            joints = torch.tensor(joints)
        except:
            joints = joints

    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError("image_size must be a tuple of (height, width)")

    height, width = image_size



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

    try:
        images = renderer(mesh)
        R_channel = images[0, :, :, 0].detach().cpu().numpy()
        G_channel = images[0, :, :, 1].detach().cpu().numpy()
        B_channel = images[0, :, :, 2].detach().cpu().numpy()
        rgbArray = np.zeros((height, width, 3), "uint8")
        rgbArray[..., 0] = (R_channel * 255).astype(int)
        rgbArray[..., 1] = (G_channel * 255).astype(int)
        rgbArray[..., 2] = (B_channel * 255).astype(int)
        img = Image.fromarray(rgbArray)
    
    except Exception as e:
        print(e)
        sys.exit()
    projections_landmarks = {}
    projections_joints = {}
    projections_extremes = {}
    projections = {}
    if landmarks_idx is not None:
        for name,idx in landmarks_idx.items():
            if name == "HEELS":
                continue
            name = name.lower()
            point = verts[idx]
            x_proj, y_proj = project_point(point, (height, width), device, cameras)
            projections_landmarks[name] = [x_proj, y_proj]

        projections["landmarks"] = projections_landmarks
        if draw_landmarks:
            draw_image = ImageDraw.Draw(img)
            for proj,values in projections_landmarks.items():
                draw_image.ellipse((values[0] - 1, values[1] - 1, values[0] + 1, values[1] + 1),fill='red')
                draw_image.text((values[0] + 5, values[1]), proj, fill='red')

    if joints is not None:
        for idx, name in SMPL_IND2JOINT.items():
            joint_point = joints[idx]
            x_proj,y_proj = project_point(joint_point, (height, width), device, cameras)
            projections_joints[name] = [x_proj, y_proj]

        projections["joints"] = projections_joints

        if draw_joints:
            draw_image = ImageDraw.Draw(img)
            for proj, values in projections_joints.items():
                draw_image.ellipse((values[0] - 2, values[1] - 2, values[0] + 2, values[1] + 2), fill='blue')
                draw_image.text((values[0] + 5, values[1]), proj, fill='blue')


    if extreme_points is not None:
        for landmark_name in extreme_points:
            left = torch.from_numpy(extreme_points[landmark_name]["left_extreme"].astype(np.float32))
            right = torch.from_numpy(extreme_points[landmark_name]["right_extreme"].astype(np.float32))

            left = project_point(left,(height,width),device,cameras)
            right = project_point(right,(height,width),device,cameras)


            #draw_image = ImageDraw.Draw(img)
            #draw_image.ellipse((left[0] - 2, left[1] - 2, left[0] + 2, left[1] + 2), fill='blue')
            ##draw_image.text((left[0] + 5, left[1]), proj, fill='blue')


            #draw_image.ellipse((right[0] - 2, right[1] - 2, right[0] + 2, right[1] + 2), fill='blue')
            projections_extremes[landmark_name] = {"leftmost":left,"rightmost":right}
            projections["extremes"] = projections_extremes

            #draw_image.text((left[0] + 5, left[1]), proj, fill='blue')



    if output_filename is not None:
        print("Saving ", os.path.join(output_path, output_filename), "\n")
        img.save(os.path.join(output_path, output_filename))
        return projections
    else:
        return img


def project_point(point: torch.Tensor,
                  image_size: Tuple[int, int],
                  device: torch.device,
                  cameras: pytorch3d.renderer.cameras.FoVPerspectiveCameras):
    
    try:
        point = point[None, None, :].to(device)  # [1, 1, 3]
        transformed_point = cameras.transform_points_screen(point,image_size=image_size)
        test_points = transformed_point.squeeze()

        # NDC to screen coordinates
        x_proj = (test_points[..., 0])
        y_proj = (test_points[..., 1])
    except Exception as e:
        print("Erro na hora de projetar pontos em 2D")
        print(f"[ERRO] {e}")
        sys.exit(1)
    #print(f"Screen coordinates: ({x_proj}, {y_proj})")  # Debugging output

    return x_proj.item(), y_proj.item()

# Sample usage:
# verts, textures, verts_uvs, faces_uvs, faces_vertices should be defined
# at = torch.tensor([0, 0, 0])
# eye_position = [2, 2, 2]
# landmarks_idx = {"HEAD_TOP": 412}
# render_mesh_textured(verts, textures, verts_uvs, faces_uvs, faces_vertices, at=at, eye_position=eye_position, landmarks_idx=landmarks_idx)
