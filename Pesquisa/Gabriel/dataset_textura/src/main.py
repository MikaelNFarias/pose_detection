import render as rd


def main() -> None:

    rd.render("../TEXTURE.png",
           "../sample_data/SMPL/models",
           "smpl",
           "../sample_data/smpl_uv.obj",
           "../sample_data/skeletex_shape.obj",
           "../sample_data/",
           debug=True,
           )

if __name__ == "__main__":
    main()