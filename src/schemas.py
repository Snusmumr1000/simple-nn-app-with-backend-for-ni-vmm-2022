import hashlib


class ImageInfo:
    def __init__(self, image: bytes, feature_vector: list[float] = None, h: str = None):
        self.image = image
        self.feature_vector = feature_vector
        self.h = h or hashlib.sha256(image).hexdigest()


class ImageInfoOutDTO:
    def __init__(self, h: str, feature_vector: list[float] = None):
        self.h = h
        self.feature_vector = feature_vector

    @staticmethod
    def from_image_info(image_info: ImageInfo):
        return ImageInfoOutDTO(image_info.h)
