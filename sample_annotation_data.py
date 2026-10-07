import os
import random
import shutil

DATA_PATH = r"/nobackup/proj/disk/it-gov-data/shared/XRFFF/"
OUTPUT_DIR = r"Annotations/data/original"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

folders = [
    os.path.join(DATA_PATH, folder)
    for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder))
]

for folder in folders:
    images = [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if (
            os.path.isfile(os.path.join(folder, file))
            and not file.startswith("._")
            and os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS
        )
    ]
    print(folder)
    print(len(images))
    sampled_images = random.sample(images, 12)
    print([os.path.basename(x) for x in sampled_images])
    folder_name = os.path.basename(folder)
    output_folder = os.path.join(OUTPUT_DIR, folder_name)

    os.makedirs(output_folder, exist_ok=True)

    for image_path in sampled_images:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, filename)

        shutil.copy2(image_path, output_path)