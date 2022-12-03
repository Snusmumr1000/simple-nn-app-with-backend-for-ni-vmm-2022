from pathlib import Path

from PIL import Image

import src.image_feature_extractor as image_feature_extractor
from src.schemas import ImageInfo

STORAGE: dict[str, ImageInfo] = {}


def load_images_to_storage():
    image_paths = list(Path("static/images").glob('*'))

    for path in image_paths:
        h = path.stem
        if path.suffix != ".jpg":
            continue
        image = Image.open(path)
        image_info = ImageInfo(image.tobytes(), image_feature_extractor.instance.extract(image).tolist(), h)
        STORAGE[h] = image_info


image_feature_extractor.instance.recalculate_action = load_images_to_storage


load_images_to_storage()
