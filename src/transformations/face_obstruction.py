import os
import numpy as np

import cv2
from PIL import Image


def check_args(**kwargs):
    obstruction_method = kwargs.get("obstruction_obstruction_method", "blur")
    n_neighbors = kwargs.get("obstruction_n_neighbors", 4)
    scale_factor = kwargs.get("obstruction_scale_factor", 1.3)
    sigma = kwargs.get("obstruction_sigma", 30)
    pixel_size = kwargs.get("obstruction_pixel_size", 10)
    face_h = kwargs.get("obstruction_face_h", 0.48)
    face_w = kwargs.get("obstruction_face_w", 0.43)
    face_angle = kwargs.get("obstruction_face_angle", 0)
    errors = []
    if obstruction_method not in {"blur", "pixelate", "block"}:
        errors.append(f"obstruction_obstruction_method must be 'blur', 'pixelate', or 'block', got '{obstruction_method}'")
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        errors.append("obstruction_n_neighbors must be a positive integer")
    if not isinstance(scale_factor, (int, float)) or scale_factor <= 1:
        errors.append("obstruction_scale_factor must be a number greater than 1")
    if not isinstance(face_h, (int, float)) or not 0 < face_h <= 1:
        errors.append("obstruction_face_h must be between 0 and 1")
    if not isinstance(face_w, (int, float)) or not 0 < face_w <= 1:
        errors.append("obstruction_face_w must be between 0 and 1")
    if not isinstance(face_angle, (int, float)):
        errors.append( "obstruction_face_angle must be a number")
    if obstruction_method == "blur":
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            errors.append("obstruction_sigma must be a positive number")
    if obstruction_method == "pixelate":
        if not isinstance(pixel_size, int) or pixel_size < 1:
            errors.append("obstruction_pixel_size must be a positive integer")

    if errors:
        raise ValueError("\n".join(errors))


def setup(transformation, output_dir, **kwargs):
    print(f"\n{'='*60}")
    print("Setting up face obstruction context...")
    obstruction_method = kwargs.get("obstruction_method", "blur")
    obstruction_n_neighbors = kwargs.get("obstruction_n_neighbors", 4)
    obstruction_scale_factor = kwargs.get("obstruction_scale_factor", 1.3)
    obstruction_sigma = kwargs.get("obstruction_sigma", 30)
    obstruction_pixel_size = kwargs.get("obstruction_pixel_size", 10)
    obstruction_face_h = kwargs.get("obstruction_face_h", 0.48)
    obstruction_face_w = kwargs.get("obstruction_face_w", 0.43)
    obstruction_face_angle = kwargs.get("obstruction_face_angle", 0)

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
        "obstruction_n_neighbors": obstruction_n_neighbors,
        "obstruction_scale_factor": obstruction_scale_factor,
        "obstruction_sigma": obstruction_sigma,
        "obstruction_pixel_size": obstruction_pixel_size,
        "obstruction_face_h": obstruction_face_h,
        "obstruction_face_w": obstruction_face_w,
        "obstruction_face_angle": obstruction_face_angle,
    }
    print(f"Transformations will be saved to: {transformation_dir}")
    return transformation_file, ["Dir", "ImageID", "obstruction_Dir","obstruction_imageID"], context, transformation_dir
     


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    filename = f"{row.ImageID}"
    image_path = os.path.join(transformation_dir, filename)
    result.save(image_path)
    return {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "obstruction_Dir": transformation_dir,
        "obstruction_imageID": filename
    }


def transform(image, context):
    return obstruct_faces(image, context['obstruction_method'], 
                   context['obstruction_n_neighbors'],
                   context['obstruction_scale_factor'],
                   context['obstruction_sigma'],
                   context['obstruction_pixel_size'],
                   context['obstruction_face_h'],
                   context['obstruction_face_w'],
                   context['obstruction_face_angle'])


def obstruct_faces(image, obstruction_type, n_neighbors, scale_factor, sigma, pixel_size, face_h, face_w, angle):
    """
    Detect and obstruct faces in an image.

    Args:
        image (str): Path to the input image.
        obstruction_type (str): obstruction_method used to obstruct detected faces.
            Supported obstruction_methods:
                "blur"     : Gaussian blur
                "pixelate" : Pixelation
                "block"    : Black out the face
        n_neighbors (int, optional): Defaults to 4.
        scale_factor (int, optional): Defaults to 1.3
        sigma (float, optional): Standard deviation of the Gaussian kernel. Larger values
            produce stronger blur. Defaults to 30.
        pixel_size (int, optional): Width and height of the reduced representation used during pixelation.
            Smaller values produce stronger pixelation. Defaults to 10.
        face_h (float, optional): Vertical radius of the face mask as a proportion of the bounding-box height.
            Larger values produce larger box borders. Defaults to 0.48.
        face_w (float, optional): Horizontal radius of the face mask as a proportion of the bounding-box width.
            Larger values produce larger box borders. Defaults to 0.42.
        angle (float, optional): Rotation angle of the elliptical mask in degrees. Defaults to 0.

    Returns:
        numpy.ndarray: The processed image in RGB format.
    """
    image = np.array(image.convert("RGB"))

    face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    face_data = face_detect.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=n_neighbors)

    # print(f"Detected {len(face_data)} face(s)")

    if obstruction_type == "blur":
        image = blur(face_data,image,sigma=sigma, face_h=face_h, face_w=face_w, angle=angle)
    elif obstruction_type == "pixelate":
        image = pixelate(face_data,image,pixel_size=pixel_size, face_h=face_h, face_w=face_w, angle=angle)
    elif obstruction_type == "block":
        image = block_out(face_data,image, face_h=face_h, face_w=face_w, angle=angle)
    return Image.fromarray(image)


def face_mask(h, w, face_h, face_w, angle):
    """
    Create an elliptical mask approximating the face region.

    Args:
        h (int): Height of the detected face bounding box in pixels.
        w (int): Width of the detected face bounding box in pixels.
        face_h (float, optional): Vertical radius of the face mask as a proportion of the bounding-box height.
            Larger values produce larger box borders. Defaults to 0.48.
        face_w (float, optional): Horizontal radius of the face mask as a proportion of the bounding-box width.
            Larger values produce larger box borders. Defaults to 0.42.
        angle (float, optional): Rotation angle of the elliptical mask in degrees. Defaults to 0.

    Returns:
        numpy.ndarray: A binary uint8 mask of shape (h, w). Pixels inside the
            elliptical face region have value 255; pixels outside
            have value 0.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * face_w), int(h * face_h))
    cv2.ellipse(mask,center,axes,angle=angle,startAngle=0,endAngle=360,color=255,thickness=-1)
    return mask


def blur(face_detection, image, sigma, face_h, face_w, angle):
    """
    A Gaussian blur is applied to an elliptical mask for each detected face. 

    Args:
        face_detection (array-like): Face detections returned by the face detector.
            Each detection must contain four values: (x, y, w, h)
            where x and y are the top-left coordinates of the
            bounding box, and w and h are its width and height.
        image (numpy.ndarray): Image containing the detected faces.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Larger values
            produce stronger blur. Defaults to 30.
        face_h (float, optional): Vertical radius of the face mask as a proportion of the bounding-box height.
            Larger values produce larger box borders. Defaults to 0.48.
        face_w (float, optional): Horizontal radius of the face mask as a proportion of the bounding-box width.
            Larger values produce larger box borders. Defaults to 0.42.
        angle (float, optional): Rotation angle of the elliptical mask in degrees. Defaults to 0.

    Returns:
        numpy.ndarray: The input image with the detected face regions blurred.

    Notes:
        The kernel size is automatically determined by OpenCV because (0, 0) is provided as the kernel size.
    """
    for (x, y, w, h) in face_detection:
        roi = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi,(0, 0),sigma)
        mask = face_mask(h, w, face_h=face_h, face_w=face_w, angle=angle)
        roi[mask == 255] = blurred_roi[mask == 255]
        image[y:y+h, x:x+w] = roi
    return image


def pixelate(face_data, image, pixel_size, face_h, face_w, angle):
    """
    Pixelate is applied to an elliptical mask for each detected face. 

    Args:
        face_detection (array-like): Face detections returned by the face detector.
            Each detection must contain four values: (x, y, w, h)
            where x and y are the top-left coordinates of the
            bounding box, and w and h are its width and height.
        image (numpy.ndarray): Image containing the detected faces.
        pixel_size (int, optional): Width and height of the reduced representation used during pixelation.
            Smaller values produce stronger pixelation. Defaults to 10.
        face_h (float, optional): Vertical radius of the face mask as a proportion of the bounding-box height.
            Larger values produce larger box borders. Defaults to 0.48.
        face_w (float, optional): Horizontal radius of the face mask as a proportion of the bounding-box width.
            Larger values produce larger box borders. Defaults to 0.42.
        angle (float, optional): Rotation angle of the elliptical mask in degrees. Defaults to 0.

    Returns:
        numpy.ndarray: The input image with the detected face regions pixelated.
    """
    for (x, y, w, h) in face_data:
        roi = image[y:y+h, x:x+w]
        small = cv2.resize(roi,(pixel_size, pixel_size),interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small,(w, h),interpolation=cv2.INTER_NEAREST)
        mask = face_mask(h, w, face_h=face_h, face_w=face_w, angle=angle)
        roi[mask == 255] = pixelated_roi[mask == 255]
        image[y:y+h, x:x+w] = roi
    return image


def block_out(face_data, image, face_h, face_w, angle):
    """
    Black out is applied to an elliptical mask for each detected face. 

    Args:
        face_detection (array-like): Face detections returned by the face detector.
            Each detection must contain four values: (x, y, w, h)
            where x and y are the top-left coordinates of the
            bounding box, and w and h are its width and height.
        image (numpy.ndarray): Image containing the detected faces.
        face_h (float, optional): Vertical radius of the face mask as a proportion of the bounding-box height.
            Larger values produce larger box borders. Defaults to 0.48.
        face_w (float, optional): Horizontal radius of the face mask as a proportion of the bounding-box width.
            Larger values produce larger box borders. Defaults to 0.42.
        angle (float, optional): Rotation angle of the elliptical mask in degrees. Defaults to 0.

    Returns:
        numpy.ndarray: The input image with the detected face regions replaced by black pixels.
    """
    for (x, y, w, h) in face_data:
        roi = image[y:y+h, x:x+w]
        mask = face_mask(h, w, face_h=face_h, face_w=face_w, angle=angle)
        roi[mask == 255] = 0
        image[y:y+h, x:x+w] = roi
    return image

