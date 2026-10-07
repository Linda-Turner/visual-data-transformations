import subprocess
import sys
import os

MAIN_SCRIPT = os.path.join("src", "main.py")

INPUT_FILE = "Annotations/data/combined_data.csv"
OUTPUT_DIR = "Annotations/transformations"

transformations =  [
    # Embeddings
    [
        "-t", "embedding",
        "--embedding_method", "clip",
        "--embedding_model", "openai/clip-vit-base-patch32"
    ],
    [
        "-t", "embedding",
        "--embedding_method", "dino",
        "--embedding_model", "facebook/dinov2-base"
    ],
    [
        "-t", "embedding",
        "--embedding_method", "vgg"
    ],
    # Hashing
    [
        "-t", "hashing",
        "--hashing_method", "phash",
    ],
    # Descriptions
    [
        "-t", "description",
        "--description_method", "descriptive"
    ],
    # Bert
    [
        "-t", "embedding",
        "--embedding_method", "bert",
        "--embedding_descriptions", "Annotations/transformations/description_descriptive.csv"
    ],
    # Object recognition
    [
        "-t", "object_recognition"
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


if __name__ == "__main__":
    main()