import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
from diffusers import StableDiffusionPipeline
import filetype
import base64

import transformations.description as image_description  


def check_args(**kwargs):
    model_name = kwargs.get("regeneration_model","sd2-community/stable-diffusion-2-1")
    descriptions = kwargs.get("regeneration_descriptions")
    errors = []
    if descriptions:
        if not os.path.exists(descriptions):
            errors.append(f"regeneration_descriptions does not exist, got {descriptions}")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("regeneration_model must be a non-empty string")

    if errors:
        raise ValueError("\n".join(errors))

    
def setup(transformation, output_dir, **kwargs):
    """
    Prepare the output file and directory for the transformation and define the transformation context.

    Returns:
        tuple[str, list[str], dict, str]: A tuple containing:
            - Path to the CSV file where transformation metadata will be saved.
            - Column names for the transformation metadata CSV file.
            - Transformation parameters used during processing.
            - Directory where transformed images will be saved.
    """
    print(f"\n{'='*60}")
    print("Setting up description generation context...")
    model_name = kwargs.get("regeneration_model","sd2-community/stable-diffusion-2-1")
    regeneration_descriptions = kwargs.get("regeneration_descriptions")

    transformation_dir = os.path.join(
        output_dir,
        "regeneration")
    os.makedirs(transformation_dir, exist_ok=True)
    print(f"Transformations will be saved to: {transformation_dir}")

    pipeline = load_generation_model(model_name)

    if regeneration_descriptions:
        gemma_model, gemma_processor, gemma_device = (image_description.load_gemma_model("google/gemma-4-E2B-it"))
        descriptions = pd.read_csv(regeneration_descriptions)
        transformation_file = os.path.join(output_dir,f"{transformation}_from_descriptions.csv")
    else:
        gemma_model, gemma_processor, gemma_device = (image_description.load_gemma_model("google/gemma-4-E2B-it"))
        descriptions = None
        transformation_file = os.path.join(output_dir,f"{transformation}.csv")
        
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    context = {
        "pipeline": pipeline,
        "regeneration_descriptions": descriptions,
        "gemma_model" : gemma_model,
        "gemma_processor" : gemma_processor,
        "gemma_device" : gemma_device,
    }
    return transformation_file, ["Dir", "ImageID", "regeneration_Dir", "regeneration_ImageID", 'description'], context, transformation_dir


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    image, description = result
    image_path = row.Dir
    filename = row.ImageID
    regeneration_path = os.path.join(transformation_dir,image_path)
    os.makedirs(regeneration_path, exist_ok=True)
    regeneration_file = os.path.join(regeneration_path,filename)
    if description is None:
        description = "No description available."
    if image is not None:
        image.save(regeneration_file)
    output_row = {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "regeneration_Dir":regeneration_path,
        "regeneration_ImageID": filename,
        'description' : description
    }
    return output_row


def transform(image_file, context):
    """
    Apply the selected description methods to an image.
    """
    # image = Image.open(image_file)
    return regenerate_images(
        image_file,
        context["pipeline"],
        context["regeneration_descriptions"], 
        context['gemma_model'],
        context['gemma_processor'],
        context['gemma_device'],
    )


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generation_model(generation_model):
    """
    Load a pre-trained stable diffusion pipeline.

    Args:
        generation_model (str): Hugging Face model identifier for the stable diffusion model.

    Returns:
        tuple: Loaded stable diffusion pipeline.
    """
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline.from_pretrained(generation_model, torch_dtype=dtype)
    pipeline = pipeline.to(device)
    return pipeline


def regenerate_images(image_path, pipeline, regeneration_descriptions,gemma_model, gemma_processor, gemma_device):
    """
    redraw images based on textual description of the original

    Args:
        image_path (str): Path to the original image.
        pipeline: Stable Diffusion pipeline used to generate the new image.
        regeneration_descriptions (pd.DataFrame | None): Optional DataFrame
            containing pre-generated descriptions.
        gemma_model: Gemma model used to generate a description when one is
            not available in regeneration_descriptions.
        gemma_processor: Processor corresponding to the Gemma model.
        gemma_device: Device on which the Gemma model is running.

    Return: 
        Tuple[image, str]: the redrawn image and the prompt used to redraw the original image. 
    """
    prompt = get_prompt(image_path, regeneration_descriptions,gemma_model, gemma_processor, gemma_device)
    if prompt:
        image = pipeline(prompt=prompt).images[0]
        return (image, prompt)
    return (None, None)



def get_prompt(image_path, regeneration_descriptions,gemma_model, gemma_processor, gemma_device):
    """
        If no description is available, generate descriptive prompt for the image. Otherwise return the ready prompt.
    """
    if regeneration_descriptions is not None:
        image_dir, image = os.path.split(image_path)
        image_dir = os.path.join(image_dir, '')
        match = regeneration_descriptions[(regeneration_descriptions["Dir"] == image_dir) & (regeneration_descriptions["ImageID"] == image)]
        if not match.empty:
            prompt = match["description"].iloc[0]
            shortend_prompt = image_description.shorten_text(prompt, gemma_processor, gemma_model, gemma_device)
            return shortend_prompt
        else:
            return None
    return generate_prompt(image_path,gemma_model,gemma_processor,gemma_device)


def generate_prompt(image_path,gemma_model, gemma_processor, gemma_device):
    """
        Generate descriptive prompt for the image.
    """
    prompt = image_description.describe_image(image_path, gemma_processor, gemma_model, 'descriptive', gemma_device)
    shortend_prompt = image_description.shorten_text(prompt, gemma_processor, gemma_model, gemma_device)
    return shortend_prompt
