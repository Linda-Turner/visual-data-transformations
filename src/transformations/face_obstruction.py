import os
import numpy as np

import cv2
from PIL import Image
from retinaface import RetinaFace


def check_args(**kwargs):
    obstruction_method = kwargs.get("obstruction_obstruction_method", "blur")
    sigma = kwargs.get("obstruction_sigma", 30)
    pixel_size = kwargs.get("obstruction_pixel_size", 10)
    errors = []
    if obstruction_method not in {"blur", "pixelate", "block"}:
        errors.append(f"obstruction_obstruction_method must be 'blur', 'pixelate', or 'block', got '{obstruction_method}'")
    if obstruction_method == "blur":
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            errors.append("obstruction_sigma must be a positive number")
    if obstruction_method == "pixelate":
        if not isinstance(pixel_size, int) or pixel_size < 1:
            errors.append("obstruction_pixel_size must be a positive integer")

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
    print("Setting up face obstruction context...")
    obstruction_method = kwargs.get("obstruction_method", "blur")
    obstruction_sigma = kwargs.get("obstruction_sigma", 30)
    obstruction_pixel_size = kwargs.get("obstruction_pixel_size", 10)

    transformation_file = os.path.join(output_dir,f"{transformation}_{obstruction_method}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    transformation_dir = os.path.join(
        output_dir,
        "face_obstruction",
        obstruction_method
    )
    os.makedirs(transformation_dir, exist_ok=True)

    context = {
        "obstruction_method": obstruction_method,
        "obstruction_sigma": obstruction_sigma,
        "obstruction_pixel_size": obstruction_pixel_size,
    }
    print(f"Transformations will be saved to: {transformation_dir}")
    return transformation_file, ["Dir", "ImageID", "obstruction_Dir","obstruction_imageID", "#detected_faces"], context, transformation_dir
     


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
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
        "#detected_faces" : number_detected_faces,
    }


def transform(image_file, context):
    """
    Apply the selected face_obstruction method to an image.
    """
    image, number_detected_faces = obstruct_faces(image_file, context['obstruction_method'], 
                   context['obstruction_sigma'],
                   context['obstruction_pixel_size'],)
    return (image, number_detected_faces)


def obstruct_faces(image_file, obstruction_type, sigma, pixel_size):
    """
    Detect and obstruct faces in an image.

    Args:
        image_file (str): path to the image file.
        obstruction_type (str): Method used to obstruct detected faces.
            Supported methods:
                "blur"     : Gaussian blur
                "pixelate" : Pixelation
                "block"    : Black out the face
        sigma (float, optional): Standard deviation of the Gaussian kernel. Larger values
            produce stronger blur. Defaults to 30.
        pixel_size (int, optional): Width and height of the reduced representation used during pixelation.
            Smaller values produce stronger pixelation. Defaults to 10.

    Returns:
        Tuple[PIL.Image, int:] Tuple of the processed image and the number of detected faces.
    """
    image = cv2.imread(image_file)

    faces = RetinaFace.detect_faces(image_file)

    if obstruction_type == "blur":
        image = blur(faces,image,sigma=sigma)
    elif obstruction_type == "pixelate":
        image = pixelate(faces,image,pixel_size=pixel_size)
    elif obstruction_type == "block":
        image = block_out(faces,image)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image, len(faces)


def blur(faces, image, sigma):
    """
    A Gaussian blur is applied to the detected mask for each detected face. 

    Args:
        faces (dict): Face detections returned by the face detector. Containing at least:
            facial_area (list[int]): Face bounding box in the format [x1, y1, x2, y2].
            landmarks (dict): RetinaFace facial landmarks containing 'right_eye' and 'left_eye' coordinates.
        image (numpy.ndarray): Image containing the detected faces.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Larger values
            produce stronger blur. Defaults to 30.

    Returns:
        numpy.ndarray: The input image with the detected face regions blurred.

    Notes:
        The kernel size is automatically determined by OpenCV because (0, 0) is provided as the kernel size.
    """
    for face in range(len(faces)):
        identity = faces[f"face_{face + 1}"]
        facial_area = identity["facial_area"]
        x1, y1, x2, y2 = facial_area
        w, h = x2 - x1, y2 - y1
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi,(0, 0),sigma)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        roi[mask == 255] = blurred_roi[mask == 255]
        image[y1:y2, x1:x2] = roi
    return image


def pixelate(faces, image, pixel_size):
    """
    Pixelate is applied to the detected mask for each detected face. 

    Args:
        faces (dict): Face detections returned by the face detector. Containing at least:
            facial_area (list[int]): Face bounding box in the format [x1, y1, x2, y2].
            landmarks (dict): RetinaFace facial landmarks containing 'right_eye' and 'left_eye' coordinates.
        image (numpy.ndarray): Image containing the detected faces.
        pixel_size (int, optional): Width and height of the reduced representation used during pixelation.
            Smaller values produce stronger pixelation. Defaults to 10.

    Returns:
        numpy.ndarray: The input image with the detected face regions pixelated.
    """
    for face in range(len(faces)):
        identity = faces[f"face_{face + 1}"]
        facial_area = identity["facial_area"]
        x1, y1, x2, y2 = facial_area
        w, h = x2 - x1, y2 - y1
        roi = image[y1:y2, x1:x2]
        small = cv2.resize(roi,(pixel_size, pixel_size),interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small,(w, h),interpolation=cv2.INTER_NEAREST)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        roi[mask == 255] = pixelated_roi[mask == 255]
        image[y1:y2, x1:x2] = roi
    return image


def block_out(faces, image):
    """
    Black out is applied to the detected area mask for each detected face. 

    Args:
        faces (dict): Face detections returned by the face detector. Containing at least:
            facial_area (list[int]): Face bounding box in the format [x1, y1, x2, y2].
            landmarks (dict): RetinaFace facial landmarks containing 'right_eye' and 'left_eye' coordinates.
        image (numpy.ndarray): Image containing the detected faces.

    Returns:
        numpy.ndarray: The input image with the detected face regions replaced by black pixels.
    """
    for face in range(len(faces)):
        identity = faces[f"face_{face + 1}"]
        facial_area = identity["facial_area"]
        x1, y1, x2, y2 = facial_area
        w, h = x2 - x1, y2 - y1
        roi = image[y1:y2, x1:x2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        roi[mask == 255] = 0
        image[y1:y2, x1:x2] = roi
    return image


# def face_mask(facial_area,landmarks,face_h=FACE_H,face_w=FACE_W):
#     """
#     Create an elliptical mask approximating a detected face, rotated
#     according to the angle of the eyes.

#     Args:
#         facial_area (list[int]): Face bounding box in the format
#             [x1, y1, x2, y2].
#         landmarks (dict): RetinaFace facial landmarks containing
#             'right_eye' and 'left_eye' coordinates.
#         face_h (float, optional): Vertical radius of the ellipse as a
#             proportion of the bounding-box height.
#         face_w (float, optional): Horizontal radius of the ellipse as a
#             proportion of the bounding-box width.

#     Returns:
#         numpy.ndarray: Binary uint8 mask of the detected face region.
#     """
#     x1, y1, x2, y2 = facial_area
#     w = x2 - x1
#     h = y2 - y1
#     mask = np.zeros((h, w), dtype=np.uint8)
#     center = (w // 2, h // 2)
#     axes = (int(w * face_w),int(h * face_h))
#     right_eye = landmarks["right_eye"]
#     left_eye = landmarks["left_eye"]
#     dx = left_eye[0] - right_eye[0]
#     dy = left_eye[1] - right_eye[1]
#     angle = np.degrees(np.arctan2(dy, dx))
#     cv2.rectangle(mask, (x2, y2), (x1, y1), (255, 255, 255), 1)
#     # cv2.ellipse(mask,center,axes,angle,0,360,255,-1)
#     return mask