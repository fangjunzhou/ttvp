from os import listdir
from PIL import Image
import cv2

IMAGE_DIR = "/home/jeff/TTVP/data/thumnailUS"
OUTPUT_DIR = "/home/jeff/TTVP/data/resizedThumbnails"

images = listdir(IMAGE_DIR)
for image in images:
    i = Image.open(IMAGE_DIR + "/" + image).convert("RGB")
    i = i.crop((0, 45, 480, 315))
    i = i.resize((224, 224), Image.BILINEAR)
    i.save(OUTPUT_DIR + "/" + image)

# i = Image.open(IMAGE_DIR + "/" + images[0]).convert("RGB")
# i = i.crop((0, 45, 480, 315))
# i = i.resize((224, 224), Image.BILINEAR)
# i.save(OUTPUT_DIR + "/" + images[0])
