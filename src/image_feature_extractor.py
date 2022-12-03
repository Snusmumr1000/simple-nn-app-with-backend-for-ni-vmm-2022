import importlib
from pprint import pprint

from torch import tensor
from torchmetrics.functional import pairwise_cosine_similarity
# from torchvision.models import resnet50 as _model, ResNet50_Weights as _weights
# from torchvision.models import regnet_y_3_2gf as _model, RegNet_Y_3_2GF_Weights as _weights
# from torchvision.models import efficientnet_v2_s as _model, EfficientNet_V2_S_Weights as _weights
from torchvision.models.feature_extraction import create_feature_extractor


class ImageFeatureExtractor:

    def __init__(self):
        self._initialize_torchvision_meta()
        self._set_model()

    def _initialize_torchvision_meta(self):
        self._torchvision_model = importlib.import_module("torchvision.models").__dict__
        _weights_postfix = "_Weights"
        _weight_keys = [e for e in self._torchvision_model.keys() if e.endswith(_weights_postfix)]
        self.model_names = [
            e if 'resnet' in e
            or 'regnet' in e
            or 'efficientnet' in e
            else None
            for e in [e[:-len(_weights_postfix)].lower() for e in _weight_keys]
        ]
        self._model2weights = {k: v for k, v in zip(self.model_names, _weight_keys)}

        self.model_names = [e for e in self.model_names if e is not None]
        del self._model2weights[None]

    def _set_model(self, model_name="resnet50"):
        self.model_name = model_name

        if model_name not in self._model2weights:
            raise ValueError(f"Model {model_name} is not supported")

        _model = self._torchvision_model[model_name]
        _weights = self._torchvision_model[self._model2weights[model_name]]

        weights = _weights.DEFAULT
        model = _model(weights=weights)
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

    def set_model(self, model_name):
        self._set_model(model_name)


instance = ImageFeatureExtractor()


def dependency():
    return instance
