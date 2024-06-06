import render as rd


def main() -> None:
    texture_image_path = "../TEXTURE.png"
    smpl_model_path = "../sample_data/SMPL/models"
    model_type = "smpl"
    smpl_uv_map_path = "../sample_data/smpl_uv.obj"
    obj_mesh_path = "../sample_data/skeletex_shape.obj"
    output_path = "../sample_data/"

    rd.render(texture_image_path,
           smpl_model_path,
           model_type,
           smpl_uv_map_path,
           obj_mesh_path,
           output_path,
           debug=True,
           cam_dist=2,
           background_image_path= "../Untitled.jpeg",
           image_size=1024,
           mode=['frontal','side'],
           anti_aliasing=True,)

if __name__ == "__main__":
    main()

# No dataset skeletex:
#    distancia da camera: [2,5], com images em 1024x1024
    # Anti - aliasing na imagem final https://github.com/facebookresearch/pytorch3d/issues/399 