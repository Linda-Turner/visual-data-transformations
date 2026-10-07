import subprocess
import sys
import os
import pandas as pd

MAIN_SCRIPT = os.path.join("src", "main.py")

INPUT_FILE = "Annotations/data/data_file.csv"
OUTPUT_DIR = "Annotations/data"


transformations =  [
    # face obstruction
    [
        "-t", "face_obstruction",
        "--obstruction_method", "blur"
    ],
    # regeneration
    [
        "-t", "regeneration",
        "--regeneration_method", "description_based"
    ],
    [
        "-t", "regeneration",
        "--regeneration_method", "image_based"
    ]
]


def main():
    failed = []

    for i, test_args in enumerate(transformations, start=1):
        command = [
            sys.executable,
            MAIN_SCRIPT,
            "-i", INPUT_FILE,
            "-o", OUTPUT_DIR,
            *test_args,
        ]

        print("\n" + "=" * 70)
        print(f"TEST {i}/{len(transformations)}")
        print(" ".join(command))
        print("=" * 70)

        result = subprocess.run(command)

        if result.returncode != 0:
            print(f"FAILED: test {i}")
            failed.append(test_args)
        else:
            print(f"PASSED: test {i}")

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Total:  {len(transformations)}")
    print(f"Passed: {len(transformations) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for test in failed:
            print(" ".join(test))

    original_df = pd.read_csv(os.path.join(OUTPUT_DIR,'data_file.csv'))
    original_df['image_type'] = 'original'

    regeneration_desc_df = pd.read_csv(os.path.join(OUTPUT_DIR,'regeneration_from_descriptions.csv'))
    image_regeneration_desc_df = regeneration_desc_df[['regeneration_Dir', 'regeneration_ImageID']]
    image_regeneration_desc_df.columns = ['Dir', 'ImageID']
    image_regeneration_desc_df['image_type'] = 'regeneration_description'   

    regeneration_img_df = pd.read_csv(os.path.join(OUTPUT_DIR,'regeneration_from_images.csv'))
    image_regeneration_img_df = regeneration_img_df[['regeneration_Dir', 'regeneration_ImageID']]
    image_regeneration_img_df.columns = ['Dir', 'ImageID']
    image_regeneration_img_df['image_type'] = 'regeneration_image'   

    face_blur_df = pd.read_csv(os.path.join(OUTPUT_DIR,'face_obstruction_blur.csv'))
    image_face_blur_df = face_blur_df[['obstruction_Dir', 'obstruction_imageID']]
    image_face_blur_df.columns = ['Dir', 'ImageID']
    image_face_blur_df['image_type'] = 'face_blur'

    combined_df = pd.concat([original_df, image_regeneration_desc_df, image_regeneration_img_df, image_face_blur_df], axis=0, ignore_index=True)
    combined_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_data.csv'), index=False)


if __name__ == "__main__":
    main()