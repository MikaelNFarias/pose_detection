import render as rd


def main() -> None:

    rd.render("../TEXTURE.png",
           "../sample_data/SMPL/models",
           "smpl",
           "../sample_data/smpl_uv.obj",
           "../sample_data/skeletex_shape.obj",
           "../sample_data/",
           debug=True,
           cam_dist=2,
           background_image_path= "../Untitled.jpeg",
           image_size=512,
           )

if __name__ == "__main__":
    main()

# No dataset skeletex:
#    distancia da camera: [2,5], com images em 1024x1024