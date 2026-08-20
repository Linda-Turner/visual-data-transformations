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
        choices=["embedding","description","hashing","face_obstruction","regeneration"],
        help="Transformation to apply.",
        required=True,
    )
    parser.add_argument(
        "-i", "--input_file",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="outputs"
    )

    # Face obstruction arguments
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
        default= "google/gemma-4-E2B-it"
    )
    parser.add_argument(
        "--description_method",
        default="descriptive",
        choices=["descriptive", "narrative"],
        help="Image description method to use."
    )
    #Regeneration
    parser.add_argument(
        "--regeneration_descriptions",
        help="Image descriptions to base the regeneration on."
    )
    parser.add_argument(
        "--regeneration_model",
        default="sd2-community/stable-diffusion-2-1",
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