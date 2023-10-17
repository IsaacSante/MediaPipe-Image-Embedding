import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# https://developers.google.com/mediapipe/solutions/vision/image_embedder/python

# find the model
model_name = "mobilenet_v3_small_075_224_embedder.tflite"
current_dir_path = os.path.dirname(os.path.abspath(__file__))
model_path = str(Path(current_dir_path) / "models" / model_name)

# define the options for image embedding 
BaseOptions = mp.tasks.BaseOptions
ImageEmbedder = mp.tasks.vision.ImageEmbedder
ImageEmbedderOptions = mp.tasks.vision.ImageEmbedderOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ImageEmbedderOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    quantize=True,
    running_mode=VisionRunningMode.IMAGE)

with ImageEmbedder.create_from_options(options) as embedder:
  # The embedder is initialized. Use it here.
  # ...
  # Load the input image from an image file.
    mp_image = mp.Image.create_from_file('/path/to/image')

    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

    # Perform image embedding on the provided single image.
    embedding_result = embedder.embed(mp_image)


# declare images to compare
# inital_image_path = str(Path(current_dir_path) / "images" / "lips1.png")
# comparison_image_path = str(Path(current_dir_path) / "images" / "lips2.png")


#logic 
# for loop 1 goes through images 
# for loop 2 goes through every other image that isnt in the parent loop 
# spits out a comparison 

# final obj looks like this 
#{image: [array of all the other images organized from most similar to least]} ???
