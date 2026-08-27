# Visual Data Transformations

Transformations of visual data that preserve different types of information from the original image.

### Usage
**1. Create virtual environment with Python 3.12**

python3.12 -m venv .venv

**2. Activate it**

source .venv/bin/activate

**3. Install all packages with pip in the same env**

python -m pip install -r requirements.txt

**4. Run code**

python src/main.py -i <input_file> -t <transformation>

Additional optional arguments can be provided depending on the selected transformation. 
These transformation-specific arguments allow you to configure the method and its parameters.
If no arguments are provided the default will be used. 

For example:

python src/main.py -i <input_file> -t description --description_method descriptive --description_model google/gemma-4-E2B-it

python src/main.py -i <input_file> -t face_obstruction --obstruction_method blur --obstruction_sigma 30

python src/main.py -i <input_file> -t regeneration --regeneration_descriptions <description_file> --regeneration_model sd2-community/stable-diffusion-2-1

python src/main.py -i <input_file> -t embedding --embedding_method clip --embedding_model openai/clip-vit-base-patch32 --embedding_normalization True

python src/main.py -i <input_file> -t hashing --hashing_method phash


# Available transformations

1. Textual description of the image
Create a textual description of the image. This can be a summation of what is visually depicted in the image, 
or a message of the meaning or narrative that is conveyed by the image.  

Arguments:
    description_method: The type of description to generate. (Defaults to 'descriptive')
        Available option include:
        'descriptive' to get the descriptive description of the image
        'narrative' to get the narrative of the image.
    description_model: The Gemma model used to generate the description. (Defaults to 'google/gemma-4-E2B-it')


2. Face Obstruction
Create the same image with detected faces obscured.
    Possible obstruction methods include blurring, pixelating and blacking-out.

Arguments:
    obstruction_method: How the detected faces should be obscured. (Defaults to 'blur')
        Available options include: 'blur', 'pixelate', 'block'
    obstruction_sigma: Controls the strength of the blur when using obstruction_method="blur". (Defaults to 30)
    obstruction_pixel_size: Controls the pixel size when using obstruction_method="pixelate". (Defaults to 10)


3. Redraw
Create a new image that preserves the general content of the original image without directly reproducing the original image.

Arguments:
    regeneration_descriptions: CSV-File containing textual descriptions of the images, containing columns 'Dir', 'ImageID' and 'description'.
        The descriptions given will be shortened to comply with the Stable Diffuser token limit.
        If no file is given, descriptions will be generated.
    regeneration_model: The Stable Diffusion model used to regenerate the images. (Defaults to 'sd2-community/stable-diffusion-2-1')


4. Embedding
Convert the image into a numerical representation using a pretrained vision model.
    Possible embedding methods include CLIP and DINO.

Arguments:
    embedding_method: The embedding method to use. (Defaults to 'clip')
        Available options include: 'clip', 'dino'
    embedding_model: The specific pretrained model to use. (Defaults to 'openai/clip-vit-base-patch32'; example for dino 'facebook/dinov2-base')
    embedding_normalization: Whether the resulting embedding should be normalized. (Defaults to True)


5. Hashing
Generate perceptual hashes of the image.
    Possible hashing methods include Average Hash, Perceptual Hash, Simple Perceptual Hash, Difference Hash, Vertical Difference Hash, Wavelet Hash, Color Hash, Crop-Resistant Hash.

Arguments:
    hashing_methods: A list of hashing methods to generate. Multiple methods can be selected at the same time. (Defaults to 'phash')
        Available option include: 'average', 'phash', 'phash_simple', 'dhash', 'dhash_vertical', 'whash', 'colorhash', 'crop_resistant'