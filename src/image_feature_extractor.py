from torch import tensor
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class ImageFeatureExtractor:
    def __init__(self):
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        self.preprocess = weights.transforms()

        return_nodes = {
            "avgpool": "features"
        }
        self.features_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    def extract(self, image):
        image = self.preprocess(image).unsqueeze(0)
        image_vector = self.features_extractor(image)["features"].squeeze()
        return image_vector

    def compare(self, image_vector, other_image_vectors):
        sim_matrix = pairwise_cosine_similarity(tensor([image_vector]), tensor(other_image_vectors)).squeeze()
        return sim_matrix


instance = ImageFeatureExtractor()


def dependency():
    return instance
