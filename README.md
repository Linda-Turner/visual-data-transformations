# Visual Data Transformations

Transformations of visual data that preserve different types of information from the original image.

### Usage
**1. Create the conda environment with Python 3.12**

conda create -n vistrans_env python=3.12 -y

**2. Activate it**

conda activate vistrans_env

**3. Install all packages with pip in the same env**

python -m pip install -r requirements.txt

**4. Run code**

python main.py -i <input_file> -t <transformation>

Additional optional arguments can be provided depending on the selected transformation. 
These transformation-specific arguments allow you to configure the method and its parameters.

For example:

python main.py -i <input_file> -t embedding --embedding_method clip --embedding_model openai/clip-vit-base-patch32

python main.py -i <input_file> -t hashing --hashing_method phash

python main.py -i <input_file> -t face_obstruction --obstruction_method blur --obstruction_sigma 30


# Available transformations

1. Textual description of the image
Create a textual description of the image. This can be a summation of what is visually depicted in the image, 
or a message of the meaning or narrative that is conveyed by the image.  

Arguments:
    description_method: The type of description to generate. (Defaults to 'descriptive')
        Available option include:
        'descriptive' to get the descriptive description of the image
        'narrative' to get the narrative of the image.
    description_model: The LLaVa model used to generate the description. (Defaults to 'llava-hf/llava-1.5-7b-hf')


2. Face Obstruction
Create the same image with detected faces obscured.
    Possible obstruction methods include blurring, pixelating and blacking-out.

Arguments:
    obstruction_method: How the detected faces should be obscured. (Defaults to 'blur')
        Available options include: 'blur', 'pixelate', 'block'
    obstruction_n_neighbors: Controls the face detection sensitivity.
    obstruction_scale_factor: Controls the scale factor used by the face detector.
    obstruction_face_h: Controls the height of the area around the detected face that is obscured.
    obstruction_face_w: Controls the width of the area around the detected face that is obscured.
    obstruction_face_angle: Controls the rotation angle of the obstruction area.
    obstruction_sigma: Controls the strength of the blur when using obstruction_method="blur".
    obstruction_pixel_size: Controls the pixel size when using obstruction_method="pixelate".


3. Redraw
Create a new image that preserves the general content of the original image without directly reproducing the original image.

NOT IMPLEMENTED.


4. Embedding
Convert the image into a numerical representation using a pretrained vision model.
    Possible embedding methods include CLIP and DINO.

Arguments:
    embedding_method: The embedding method to use. (Defaults to 'clip')
        Available options include: 'clip', 'dino'
    embedding_model: The specific pretrained model to use. (Defaults to 'openai/clip-vit-base-patch32')
    embedding_normalization: Whether the resulting embedding should be normalized. (Defaults to True)


5. Hashing
Generate perceptual hashes of the image.
    Possible hashing methods include Average Hash, Perceptual Hash, Simple Perceptual Hash, Difference Hash, Vertical Difference Hash, Wavelet Hash, Color Hash, Crop-Resistant Hash.

Arguments:
    hashing_methods: A list of hashing methods to generate. Multiple methods can be selected at the same time. (Defaults to 'phash')
        Available option include: 'average', 'phash', 'phash_simple', 'dhash', 'dhash_vertical', 'whash', 'colorhash', 'crop_resistant'