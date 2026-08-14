import argparse

from orchestrator import transform_images


def str_to_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        "Expected true/false"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply visual data transformations."
    )
    parser.add_argument(
        "-t", "--transformation",
        choices=["embedding","description","hashing","face_obstruction",],
        help="Transformation to apply."
    )
    parser.add_argument(
        "-i", "--input_file",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="outputs"
    )

    # Face obstruction arguments
    parser.add_argument(
        "--obstruction_n_neighbors",
        type=int,
        default=4
    )

    parser.add_argument(
        "--obstruction_scale_factor",
        type=float,
        default=1.3
    )

    parser.add_argument(
        "--obstruction_sigma",
        type=float,
        default=30
    )

    parser.add_argument(
        "--obstruction_pixel_size",
        type=int,
        default=10
    )

    parser.add_argument(
        "--obstruction_face_h",
        type=float,
        default=0.48
    )

    parser.add_argument(
        "--obstruction_face_w",
        type=float,
        default=0.42
    )

    parser.add_argument(
        "--obstruction_face_angle",
        type=float,
        default=0
    )
    parser.add_argument(
        "--obstruction_method",
        default="blur",
        choices=["blur", "pixelate", "block"],
        help="Face obstruction method to use."
    )
    # Hashing arguments
    parser.add_argument(
        "--hashing_method",
        default="phash",
        choices=["average","phash","phash_simple","dhash","dhash_vertical","whash","colorhash","crop_resistant"],
        help="Hashing method to use."
    )
    # Embedding arguments
    parser.add_argument(
        "--embedding_model",
        default="openai/clip-vit-base-patch32"
    )
    parser.add_argument(
        "--embedding_normalization",
        type=str_to_bool,
        default=True
    )
    parser.add_argument(
        "--embedding_method",
        default="clip",
        choices=["clip", "dino"],
        help="Image embedding method to use."
    )
    # Description arguments
    parser.add_argument(
        "--description_model",
        default=None
    )
    parser.add_argument(
        "--description_method",
        default="descriptive",
        choices=["descriptive", "narrative"],
        help="Image description method to use."
    )

    args = parser.parse_args()
    kwargs = vars(args)
    transformation = kwargs.pop("transformation")
    input_file = kwargs.pop("input_file")
    output_dir = kwargs.pop("output_dir")

    try:
        transform_images(input_file,output_dir,transformation,**kwargs)
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()