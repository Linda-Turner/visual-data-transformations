import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
import filetype
import base64


def check_args(**kwargs):
    description_method = kwargs.get("description_method", "descriptive")
    model_name = kwargs.get("description_model","google/gemma-4-E2B-it")
    errors = []
    if description_method not in {"descriptive", "narrative"}:errors.append(f"description_method must be 'descriptive' or 'narrative', got '{description_method}'")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("description_model must be a non-empty string")

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
    description_method = kwargs.get("description_method", "general")
    model_name = kwargs.get("description_model","google/gemma-4-E2B-it")

    transformation_file = os.path.join(output_dir,f"{transformation}_{description_method}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    model, processor, device = load_gemma_model(model_name)

    context = {
        "description_model": model,
        "processor": processor,
        "description_method": description_method,
        "device" : device
    }
    return transformation_file, ["Dir", "ImageID", "description"], context, output_dir


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    output_row = {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "description":result
    }
    return output_row


def transform(image_file, context):
    """
    Apply the selected description methods to an image.
    """
    # image = Image.open(image_file)
    return describe_image(
        image_file,
        context["processor"],
        context["description_model"],
        context['description_method'], 
        context['device']
    )


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_gemma_model(gemma_description_model):
    """
    Load a pre-trained GEMMA model and processor.

    Args:
        llava_model (str): Hugging Face model identifier.

    Returns:
        tuple: Loaded LLaVA model and processor.
    """
    print("Loading GEMMA model...")

    device = get_device()
    model = AutoModelForMultimodalLM.from_pretrained(gemma_description_model, dtype="auto")
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(gemma_description_model)
    return model, processor, device


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    
def describe_image(image_path, processor, model, description_method, device):
    """
    Generate either a descriptive or narrative image description.

    Args:
        image_path (str): Path to the image.
        processor: Gemma processor.
        model: Gemma model.
        description_method (str): "descriptive" or "narrative".
        device: Model device.
    
    Return:
        str: A response from the LLM.
    """
    if description_method == 'descriptive':
        PROMPT = "Describe what is shown in the image in a few sentences. Try to include as much detail as possible."
    elif description_method == 'narrative':
        PROMPT = "Describe what this image is trying to communicate. Discuss its purpose, meaning and narrative."

    kind = filetype.guess(image_path)

    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user", "content": [
                {"type": "image", "url": f"data:{kind.mime};base64,{base64_image}"},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=300)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    response_oneline = response.replace('\n', ' ').replace('\r', '')
    return response_oneline.strip()


def shorten_text(text, processor, model, device):
    """
    Shorten text given by the user.

    Args:
        text (str): Text to shorten.
        processor: Gemma processor.
        model: Gemma model.
        device: Model device.
    
    Return:
        str: A response from the LLM.
    """
    messages = [
        {
            "role": "user", "content": [
                {"type": "text", "text": f"Shorten the following description to a few sentences, in maximum 75 tokens. The description is: {text}"}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(**inputs, max_new_tokens=300)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    response_oneline = response.replace('\n', ' ').replace('\r', '')
    return response_oneline.strip()