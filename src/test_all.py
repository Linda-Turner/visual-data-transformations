import subprocess
import sys
import os


INPUT_FILE = "data/data_file.csv"
OUTPUT_DIR = "outputs/test_all_gpu_2"


tests = [
    #Descriptions
   [
       "-t", "description",
       "--description_method", "descriptive"
   ],
   [
       "-t", "description",
       "--description_method", "narrative"
   ],
    #Generation
   [
       "-t", "regeneration",
   ],
    [
        "-t", "regeneration",
        "--regeneration_descriptions", "outputs/test_all_cpu/description_descriptive.csv"
    ],
    # Face obstruction
    [
        "-t", "face_obstruction",
        "--obstruction_method", "blur",
    ],
    [
        "-t", "face_obstruction",
        "--obstruction_method", "pixelate",
    ],
    [
        "-t", "face_obstruction",
        "--obstruction_method", "block",
    ],

    # Hashing
    [
        "-t", "hashing",
        "--hashing_method", "average",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "phash",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "phash_simple",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "dhash",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "dhash_vertical",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "whash",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "colorhash",
    ],
    [
        "-t", "hashing",
        "--hashing_method", "crop_resistant",
    ],

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
]


def main():
    failed = []

    for i, test_args in enumerate(tests, start=1):
        command = [
            sys.executable,
            os.path.join('src', 'main.py'),
            "-i", INPUT_FILE,
            "-o", OUTPUT_DIR,
            *test_args,
        ]

        print("\n" + "=" * 70)
        print(f"TEST {i}/{len(tests)}")
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

    print(f"Total:  {len(tests)}")
    print(f"Passed: {len(tests) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for test in failed:
            print(" ".join(test))


if __name__ == "__main__":
    main()
