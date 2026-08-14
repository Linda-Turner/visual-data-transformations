import os

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def check_args(**kwargs):
    description_method = kwargs.get("description_method", "descriptive")
    model_name = kwargs.get("description_model","llava-hf/llava-1.5-7b-hf")
    errors = []
    if description_method not in {"descriptive", "narrative"}:errors.append(f"description_method must be 'descriptive' or 'narrative', got '{description_method}'")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("description_model must be a non-empty string")

    if errors:
        raise ValueError("\n".join(errors))

    
def setup(transformation, output_dir, **kwargs):
    print(f"\n{'='*60}")
    print("Setting up description generation context...")
    description_method = kwargs.get("description_method", "general")
    model_name = kwargs.get("description_model",'llava-hf/llava-1.5-7b-hf')

    transformation_file = os.path.join(output_dir,f"{transformation}_{description_method}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    model, processor = load_llava_model(model_name)

    context = {
        "description_model": model,
        "processor": processor,
        "description_method": description_method,
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


def transform(image, context):
    """
    Apply the selected description description_methods to an image.
    """
    return describe_image(
        image,
        context["processor"],
        context["embedding_model"],
        context['description_method']
    )


def load_llava_model(llava_model):
    """
    Load a pre-trained LLaVA model and processor.

    Args:
        llava_model (str): Hugging Face model identifier.

    Returns:
        tuple: Loaded LLaVA model and processor.
    """
    print("Loading LLaVA model...")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = LlavaForConditionalGeneration.from_pretrained(
        llava_model,
        torch_dtype=dtype,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(llava_model)
    return model, processor


def describe_image(image, processor, model, description_method):
    if description_method == 'general':
        PROMPT = "Describe this image. Describe only what is visually observable, including people, objects, text, symbols," \
        "and the overall scene. Do not discuss its purpose, meaning, narrative, or make assumptions about what the image communicates."
    elif description_method == 'narrative':
        PROMPT = "Describe this image. Describe only what is visually observable, including people, objects, text, symbols," \
        "and the overall scene. Do not discuss its purpose, meaning, narrative, or make assumptions about what the image communicates."

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (PROMPT)
                }
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
    inputs = processor(images=image,text=prompt,return_tensors="pt")
    inputs = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    generate_ids = model.generate(**inputs,max_new_tokens=100,do_sample=False)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = generate_ids[:, input_length:]
    result = processor.batch_decode(generated_ids,skip_special_tokens=True)[0].strip()
    return result