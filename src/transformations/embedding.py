import os
import numpy as np

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel


def check_args(**kwargs):
    embedding_method = kwargs.get("embedding_method", "clip")
    model_name = kwargs.get("embedding_model","openai/clip-vit-base-patch32")
    normalization = kwargs.get("embedding_normalization",True)
    errors = []
    if embedding_method not in {"clip", "dino"}:errors.append(f"embedding_method must be 'clip' or 'dino', got '{embedding_method}'")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("embedding_model must be a non-empty string")
    if not isinstance(normalization, bool):
        errors.append("embedding_normalization must be True or False")

    if errors:
        raise ValueError("\n".join(errors))


def setup(transformation, output_dir, **kwargs):
    print(f"\n{'='*60}")
    print("Setting up embedding context...")
    embedding_method = kwargs.get("embedding_method", "clip")
    model_name = kwargs.get("embedding_model",'openai/clip-vit-base-patch32')
    normalization = kwargs.get("embedding_normalization", True)

    transformation_file = os.path.join(output_dir,f"{transformation}_{embedding_method}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    transformation_dir = os.path.join(
        output_dir,
        "embedding",
        embedding_method
    )
    os.makedirs(transformation_dir, exist_ok=True)

    if embedding_method == "clip":
        embedding_model, processor, device = load_clip_model(model_name)
    elif embedding_method == "dino":
        embedding_model, processor, device = load_dino_model(model_name)

    context = {
        "embedding_model": embedding_model,
        "processor": processor,
        "embedding_method": embedding_method,
        "device": device,
        "embedding_normalization": normalization,
    }
    print(f"Transformations will be saved to: {transformation_dir}")
    return transformation_file, ["Dir", "ImageID", "embedding_Dir", "embedding_ImageID"], context, transformation_dir


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    filename = f"{row.ImageID}.npy"
    embedding_path = os.path.join(transformation_dir,filename)
    np.save(embedding_path, result)
    return {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "embedding_Dir": transformation_dir, 
        "embedding_ImageID" : filename
    }


def transform(image, context):
    if context["embedding_method"] == "clip":
        return clip_embedding(
            image,
            context["processor"],
            context["embedding_model"],
            context["embedding_normalization"],
            context["device"]
        )
    elif context["embedding_method"] == "dino":
        return dino_embedding(
            image,
            context["processor"],
            context["embedding_model"],
            context["embedding_normalization"],
            context["device"]
        )


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model(clip_model):
    """
    Load a pre-trained CLIP model and processor.

    Args:
        clip_model (str): Hugging Face model identifier for the CLIP model.

    Returns:
        tuple: Loaded CLIP model and processor.
    """
    print("Loading CLIP model...")
    device=get_device()
    model = CLIPModel.from_pretrained(clip_model)
    model.to(device) 
    model.eval()
    processor = CLIPProcessor.from_pretrained(clip_model)
    return model, processor, device


def clip_embedding(
    image,
    image_processor,
    clip_model,
    normalization,
    device
):
    """
    Generate a CLIP image embedding for an image.

    Args:
        image_file (str): Path to the input image.
        image_processor: Pre-trained CLIP image processor used to prepare the image for the model.
        clip_model: Pre-trained CLIP model used to generate the image embedding.
        normalization (bool): If True the image embedding is normalized 
        device: Device on which the model and input tensors are processed, e.g. "cuda" or "cpu". Defaults "cpu"

    Returns:
        numpy.ndarray: CLIP image embedding as a one-dimensional NumPy array.
    """
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        embedding = clip_model.visual_projection(vision_outputs.pooler_output)
    if normalization: 
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding.cpu().numpy()


def load_dino_model(dino_model):
    """
    Load a pre-trained DINO model and processor.

    Args:
        blip_model (str): Hugging Face model identifier for the DINO model.

    Returns:
        tuple: Loaded DINO model and processor.
    """
    print("Loading DINO model...")
    device=get_device()
    model = AutoModel.from_pretrained(dino_model)
    model.to(device) 
    model.eval()
    processor = AutoImageProcessor.from_pretrained(dino_model)
    return model, processor, device


def dino_embedding(
    image,
    image_processor,
    dino_model,
    normalization,
    device
):
    """
    Generate a DINO image embedding for an image.

    Args:
        image_file (str): Path to the input image.
        image_processor: Pre-trained DINO image processor used to prepare the image for the model.
        dino_model: Pre-trained DINO model used to generate the image embedding.
        normalization (bool): If True the image embedding is normalized 
        device: Device on which the model and input tensors are processed, e.g. "cuda" or "cpu". Defaults "cpu"

    Returns:
        numpy.ndarray: DINO image embedding as a one-dimensional NumPy array.
    """
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dino_model(**inputs)
    embedding = outputs.pooler_output
    if normalization:
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding.cpu().numpy()