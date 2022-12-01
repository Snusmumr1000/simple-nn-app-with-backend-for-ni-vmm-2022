from io import BytesIO

import uvicorn as uvicorn
from fastapi import FastAPI, File, Depends

from PIL import Image


# https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

import src.image_feature_extractor as image_feature_extractor
from src.persistence import STORAGE
from src.schemas import ImageInfo, ImageInfoOutDTO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def calculate_image_vector(
        file: bytes = File(),
        extractor: image_feature_extractor.ImageFeatureExtractor = Depends(image_feature_extractor.dependency),
):
    image = Image.open(BytesIO(file))
    return {"image_vector": extractor.extract(image).tolist()}


@app.post("/images")
async def create_image_vector(
        file: bytes = File(),
        extractor: image_feature_extractor.ImageFeatureExtractor = Depends(image_feature_extractor.dependency),
):
    image = Image.open(BytesIO(file)).convert('RGB')
    image_info = ImageInfo(file, extractor.extract(image).tolist())
    h = image_info.h
    if h not in STORAGE:
        image.save(f"static/images/{h}.jpg")
        STORAGE[h] = image_info


@app.get("/images")
async def get_image_ids():
    return [*map(ImageInfoOutDTO.from_image_info, STORAGE.values())]


@app.get("/images/{h}/comparison")
async def get_image_comparison(
        h: str,
        extractor: image_feature_extractor.ImageFeatureExtractor = Depends(image_feature_extractor.dependency),
):
    image_info = STORAGE[h]
    other_images_info = [e for e in STORAGE.values() if e != image_info]
    image_vector = image_info.feature_vector
    other_feature_vectors = [other_image.feature_vector for other_image in other_images_info]

    sim_matrix = extractor.compare(image_vector, other_feature_vectors)
    return {i.h: p for i, p in zip(other_images_info, sim_matrix.tolist())}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
