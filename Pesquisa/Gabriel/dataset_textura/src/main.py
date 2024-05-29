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


def load_texture_image(texture_image_path):
    with Image.open(texture_image_path) as texture_image:
        return np.asarray(texture_image.convert("RGB")).astype(np.float32)

def initialize_smpl(smpl_model_path, gender='male', model_type='smpl'):
    return SMPL.create(
        model_path=smpl_model_path,
        gender=gender,
        model_type=model_type
    )

def test1() -> None:
    mesh = trimesh.load('teste.obj',process=False)
    texture_image = Image.open('../TEXTURE.png')
    tex = trimesh.visual.TextureVisuals(image=texture_image)
    mesh.visual.texture = tex
    mesh.show()

def test2() -> None:
    mesh = vedo.load('teste.obj').texture('TEXTURE.png')
    mesh.lighting('glossy')
    mesh.show()


def test3(texture_image_path: str,
          obj_mesh_path: str,
          smpl_model_path: str,
          gender: str = 'male',
          model_type: str = 'smpl') -> None :
    
    with Image.open(texture_image_path) as texture_image:
        np_image = np.asarray(texture_image.convert("RGB")).astype(np.float32)
    
    smpl = SMPL.create(
        model_path=smpl_model_path,
        gender = gender,
        model_type = model_type
    )

    betas, expression = (None, None)

    smpl_out = smpl(betas=betas, expression=expression,return_verts=True)

    
    ...
    


def main(debug=False) -> None:
    with Image.open("../TEXTURE.png") as texture_image:
        np_image = np.asarray(texture_image.convert("RGB")).astype(np.float32)

    smpl = SMPL.create(
        model_path="../sample_data/SMPL/models",
        gender = "male",
        model_type="smpl",
    )

    smpl_out = smpl(
        betas=None,
        expression=None,
        return_verts=True,
    )
    #smpl_out.vertices
    

    mesh_filename = "../sample_data/smpl_uv.obj"
    verts_obj, faces_verts, aux = load_obj(mesh_filename)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, F, 3)
    faces_uvs = faces_verts.textures_idx[None, ...]  # (1, F, 3)
    #print(faces_verts.verts_idx.shape)

    print(verts_uvs.shape, faces_uvs.shape,faces_verts.verts_idx.shape) 
    #load skeletex obj
    skeletex_mesh_filename = "../sample_data/skeletex_shape.obj"
    verts_sk, faces_sk, aux_sk = load_obj(skeletex_mesh_filename)
    print(verts_obj.shape,verts_sk.shape)
    #exit()
    

        #   debug: saves sampled SMPL mesh
    # save_obj('/tmp/hello_smpl3.obj', smpl_output.vertices[0], smpl.faces_tensor)
    # verts, faces_idx, _ = load_obj('/tmp/hello_smpl3.obj')
    verts = verts_sk
    #print(f"{verts[0:5]}\n\n{verts_sk[0:5]}")
    print(f"{verts.mean(dim=0)}\n\n {verts_sk.mean(dim=0)}")
    print(f"{verts.min(dim=0)}\n\n {verts_sk.min(dim=0)}")
    print(f"{verts.max(dim=0)}\n\n {verts_sk.max(dim=0)}")
    #exit()
    faces = smpl.faces_tensor
    verts_T_pose = verts


    # x mexe para esquerda e direita
    # y mexe para frente e para tras
    # z mexe para cima e para baixo
    if not os.path.isfile("../sample_data/front_render.png") or debug:
        print("[INFO] Rendering front image")  
        render_mesh_textured(
                "cpu",
                verts,
                np_image,
                verts_uvs,
                faces_uvs,
                faces_verts.verts_idx,
                image_size=1024,  # image resolution
                cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position (x,y,z)
                mesh_rot=0,  # mesh rotation in Y axis in degrees
                output_path="../sample_data/",
                output_filename="front_render.png",
                orientation='frontal',
            )
    
    if not os.path.isfile("../sample_data/side_render.png") or debug:
        print("[INFO] Rendering side image")
        render_mesh_textured(
            "cpu",
            verts,
            np_image,
            verts_uvs,
            faces_uvs,
            faces_verts.verts_idx,
            image_size=1024,  # image resolution
            cam_pos=torch.tensor([2.0, 0.35, 0]),  # camera position
            mesh_rot=90,  # mesh rotation in Y axis in degrees
            output_path="../sample_data/",
            output_filename="side_render.png",
            orientation='side',
        )
    

if  __name__ == "__main__":
    main(debug=True)

#TODO :
    # Testar em diversas condições de luz (Quantidade de luz, posição das luzes e parametros de cor)
    # Testar em diversas condições de câmera (paramêtros de câmera, posição e orientação da câmera)
    # Adicionar background
    #
