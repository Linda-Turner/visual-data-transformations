import argparse
import os, csv
from tqdm import tqdm
import time

from PIL import Image
import pandas as pd

import transformations.hashing as image_hash  
import transformations.embedding as image_embedding  
import transformations.face_obstruction as face_obstruction  
import transformations.description as image_description  
import transformations.regeneration as image_regeneration  


TRANSFORMATIONS = {
    "hashing": image_hash,
    "embedding": image_embedding,
    "face_obstruction": face_obstruction,
    "description": image_description,
    "regeneration": image_regeneration,
}


def transform_images(input_file, output_dir, transformation, **kwargs):
    start_time = time.perf_counter()

    try:
        if transformation not in TRANSFORMATIONS:
            raise ValueError(
                f"Unknown transformation: {transformation}"
            )
        if not os.path.exists(input_file):
            raise ValueError("Input file does not exist")
        
        transformer = TRANSFORMATIONS[transformation]
        try:
            transformer.check_args(**kwargs)
        except ValueError as e:
            raise ValueError(
                f"Invalid arguments for transformation '{transformation}':\n{e}"
            ) from None                        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)    

        # Get setup information
        transformation_file, transformation_columns, context, transformation_dir = transformer.setup(
            transformation, output_dir, **kwargs
        )

        input_df = pd.read_csv(input_file)

        processed = set()
        if os.path.exists(transformation_file):
            with open(transformation_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    processed.add(os.path.join(row['Dir'], row['ImageID']))
            writer_file = open(transformation_file, "a", newline="", encoding="utf-8")
            print('Images already processed: ' + str(len(processed)))
        else:
            writer_file = open(transformation_file, "w", newline="", encoding="utf-8")
            print('Images already processed: 0')

        writer = csv.DictWriter(
            writer_file,
            fieldnames=transformation_columns
        )

        if os.stat(transformation_file).st_size == 0:
            writer.writeheader()

        print(f"\n{'='*60}")
        print(f"Transforming data using {transformation}...")

        for row in tqdm(input_df.itertuples(), total=len(input_df)):
            image_path = os.path.join(row.Dir, row.ImageID)

            if image_path in processed:
                continue

            # Transform the image
            result = transformer.transform(image_path, context)

            # Write the result to csv
            output_row = transformer.format_output(
                result, row, transformation_dir
            )
            writer.writerow(output_row)
            writer_file.flush()

        writer_file.close()

    finally:
        elapsed = time.perf_counter() - start_time

        print(f"\n{'='*60}")
        print(f"Total transformation time: {elapsed:.2f} seconds")
        print(f"Total transformation time: {elapsed / 60:.2f} minutes")
        print(f"{'='*60}")


def str_to_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(
        "Expected true/false"
    )


def parse_args():
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kwargs = vars(args)
    transformation = kwargs.pop("transformation")
    input_file = kwargs.pop("input_file")
    output_dir = kwargs.pop("output_dir")

    try:
        transform_images(input_file,output_dir,transformation,**kwargs)
    except Exception as e:
        print(f"\nError: {e}")