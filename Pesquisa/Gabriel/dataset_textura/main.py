import trimesh
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import vedo
import os
import torch
from render_mesh import render_mesh_textured
import smplx as SMPL
from pytorch3d.io import load_obj
# Load the OBJ file

def test1() -> None:
    mesh = trimesh.load('teste.obj',process=False)
    texture_image = Image.open('TEXTURE.png')
    tex = trimesh.visual.TextureVisuals(image=texture_image)
    mesh.visual.texture = tex
    mesh.show()

def test2() -> None:
    mesh = vedo.load('teste.obj').texture('TEXTURE.png')
    mesh.lighting('glossy')
    mesh.show()


def test3() -> None :
    ...


def main(debug=False) -> None:
    with Image.open("TEXTURE.png") as texture_image:
        np_image = np.asarray(texture_image.convert("RGB")).astype(np.float32)

    smpl = SMPL.create(
        model_path="sample_data/SMPL/models/",
        gender = "male",
        model_type="smpl",
    )

    betas, expression = None, None

    smpl_out = smpl(betas=betas, expression=expression,return_verts=True)
    

    #load mesh

    mesh = load_obj("sample_data/SMPL_male_default_resolution.obj")
    _,facets_verts,aux = mesh

    verts_uvs = aux.verts_uvs[None,...]
    faces_uvs = facets_verts.textures_idx[None,...]

    verts = smpl_out.vertices[0]
    faces = smpl.faces_tensor


    if not os.path.isfile("sample_data/front_render.png") or debug:  
        print("[INFO] Rendering front image")  
        render_mesh_textured(
                "cpu",
                verts,
                np_image,
                verts_uvs,
                faces_uvs,
                facets_verts.verts_idx,
                image_size=1024,  # image resolution
                cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
                mesh_rot=0,  # mesh rotation in Y axis in degrees
                output_path="sample_data/",
                output_filename="front_render.png",
            )
    
    if not os.path.isfile("sample_data/side_render.png") or debug:
        print("[INFO] Rendering side image")
        render_mesh_textured(
            "cpu",
            verts,
            np_image,
            verts_uvs,
            faces_uvs,
            facets_verts.verts_idx,
            image_size=1024,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=90,  # mesh rotation in Y axis in degrees
            output_path="sample_data/",
            output_filename="side_render.png",
        )
    

if  __name__ == "__main__":
    main(debug=True)
