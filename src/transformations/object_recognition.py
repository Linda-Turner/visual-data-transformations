import os
import numpy as np

import cv2
from PIL import Image
from retinaface import RetinaFace
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.transforms.functional import to_tensor, to_pil_image

def check_args(**kwargs):
    errors = []
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
    print("Setting up object recognition context...")
    model, preprocess, weights = setup_object_recognition_model()

    transformation_file = os.path.join(output_dir,f"{transformation}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    transformation_dir = os.path.join(output_dir,"object_recognition")
    os.makedirs(transformation_dir, exist_ok=True)

    context = {
        "recognition_model" : model,
        "recognition_preprocess" : preprocess,
        "recognition_weights" : weights
    }
    print(f"Transformations will be saved to: {transformation_dir}")
    return transformation_file, ["Dir", "ImageID", "obstruction_Dir","obstruction_imageID", "#detected_objects","objects"], context, transformation_dir
     

def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    if "object_recognition" in transformation_dir:
        column_title = "#detected_objects"
    else:
        column_title = "#detected_faces"
    image, number_detected_faces = result
    image_path = row.Dir
    filename = f"{row.ImageID}"
    obstruction_path = os.path.join(transformation_dir,image_path)
    os.makedirs(obstruction_path, exist_ok=True)
    obstruction_file = os.path.join(obstruction_path,filename)
    image.save(obstruction_file)
    return {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "obstruction_Dir": obstruction_path,
        "obstruction_imageID": image,
        f"{column_title}": number_detected_faces,
    }
# def format_output(result, row, transformation_dir):
#     """
#     Format the transformation result as one CSV row.
#     """
#     image, number_detected_faces, labels = result
#     image_path = row.Dir
#     filename = f"{row.ImageID}"
#     relative_dir = os.path.relpath(image_path, start="Annotations/data")
#     obstruction_path = os.path.join(transformation_dir,relative_dir)
#     os.makedirs(obstruction_path, exist_ok=True)
#     object_file = os.path.join(obstruction_path,filename)
#     image.save(object_file)
#     return {
#         "Dir": image_path,
#         "ImageID": filename,
#         "obstruction_Dir": object_file,
#         "obstruction_imageID": filename,
#         "#detected_objects": number_detected_faces,
#         "objects": str(labels)
#     }


def transform(image_file, context):
    """
    Apply the selected object_recognition method to an image.
    """
    image, number_detected, labels = object_recognition(image_file,
            context['recognition_model'], context['recognition_preprocess'], context['recognition_weights'])
    return (image, number_detected, labels)


def setup_object_recognition_model():
    """
    Set up the object recognition model for face obstruction.
    Returns:
        model: The object recognition model.
        preprocess: The preprocessing function for the model's weights.
    """
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(weights=weights, score_thresh=0.35)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess, weights

    
def object_recognition(image_path, model, preprocess, weights):
    image = Image.open(image_path)
    preproc_image = preprocess(image)
    with torch.no_grad():
        prediction = model([preproc_image])[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    tensor = to_tensor(image)
    image_boxes = draw_bounding_boxes(tensor, boxes=prediction["boxes"],labels=labels, width=5, colors="cyan", font_size=40)
    image = to_pil_image(image_boxes)
    return image, len(prediction["boxes"]), labels