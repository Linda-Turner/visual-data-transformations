import os
from imagehash import average_hash, phash, phash_simple, dhash, dhash_vertical, whash, colorhash, crop_resistant_hash, hex_to_hash, hex_to_multihash

HASH_hashing_methodS = {
    "average": average_hash,
    "phash": phash,
    "phash_simple": phash_simple,
    "dhash": dhash,
    "dhash_vertical": dhash_vertical,
    "whash": whash,
    "colorhash": colorhash,
    "crop_resistant": crop_resistant_hash,
}

def check_args(**kwargs):
    hashing_method = kwargs.get("hashing_method", "phash")
    errors=[]
    if hashing_method not in HASH_hashing_methodS :errors.append(f"embedding_hashing_method must be {list(HASH_hashing_methodS.keys())}, got '{hashing_method}'")
    if errors:
        raise ValueError("\n".join(errors))

def setup(transformation, output_dir, **kwargs):
    """
    Prepare the transformation directory if necessary and define the output columns.

    Returns:
        Tuple[str, list[str]]: a tuple of the transformation directory and a list of the columns names
    """
    print(f"\n{'='*60}")
    print("Setting up hashing context...")
    hashing_method = kwargs.get("hashing_method", "phash")
    context = {'hashing_method' : hashing_method}

    transformation_file = os.path.join(output_dir,f"{transformation}_{hashing_method}.csv")
    print(f"Metadata on transformations will be saved at: {transformation_file}")

    return transformation_file, ["Dir", "ImageID", "hash"], context, output_dir


def format_output(result, row, transformation_dir):
    """
    Format the transformation result as one CSV row.
    """
    output_row = {
        "Dir": row.Dir,
        "ImageID": row.ImageID,
        "hash" : result
    }
    return output_row


def transform(image, context):
    """
    Apply the selected hashing hashing_methods to an image.
    """
    hashing_method = context['hashing_method']
    return image_hash(image, hashing_method)


def image_hash(image, hashing_method : (str) = "phash"):
    """
    Generate one or more perceptual hashes for an image.

    Args:
        image (PIL.Image): PIL Image.
        hashing_method (str, optional): A list of hashing hashing_methods to use. Defaults to ["phash"]. Supported hashing_methods are:
                "average"
                "phash"
                "phash_simple"
                "dhash"
                "dhash_vertical"
                "whash"
                "colorhash"
                "crop_resistant"

    Returns:
        Dictionary mapping hashing_method names to hexadecimal hash strings.
    """
    return str(HASH_hashing_methodS[hashing_method](image))
    

def restore_hash(hash_value):
    """
    Convert a hexadecimal (multi-)hash back into ImageHash objects.

    Args:
        hash_value (str): Hexadecimal representation of a (multi-)hash.

    Returns:
        list: A list of ImageHash objects reconstructed from the hexadecimal (multi-)hash.
    """
    return hex_to_multihash(hash_value)