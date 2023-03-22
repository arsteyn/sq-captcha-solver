import imutils
from fastapi import FastAPI, UploadFile, File
import cv2
from keras.models import load_model
import pickle
import numpy as np
from pathlib import Path
import concurrent.futures
from motor.motor_asyncio import AsyncIOMotorClient
from motor.aiohttp import AIOHTTPGridFS

app = FastAPI()

script_location = Path(__file__).absolute().parent

MODEL_FILENAME = script_location / "captcha_model.hdf5"
MODEL_LABELS_FILENAME = script_location / "model_labels.dat"

model = None
lb = None

db_client = None
grid_fs_bucket = None

async def load_model_and_labels():
    global model
    global lb

    # Load up the model labels (so we can translate model predictions to actual letters)
    with open(MODEL_LABELS_FILENAME, "rb") as f:
        lb = pickle.load(f)

    # Load the trained neural network
    model = load_model(MODEL_FILENAME)


async def connect_db():
    global db_client
    global grid_fs_bucket

    db_client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = db_client["your_database_name"]
    grid_fs_bucket = AIOHTTPGridFS(db)


@app.on_event("startup")
async def startup_event():
    await load_model_and_labels()
    await connect_db()


@app.get("/")
def hello_world():
    return {"message": "OK"}


@app.post("/solve-captcha/")
async def create_upload_file(file: UploadFile = File()):
    try:
        # Load the image and convert it to grayscale
        file_bytes = np.asarray(bytearray(await file.file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        result_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        result_image = cv2.copyMakeBorder(result_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        thresh = cv2.threshold(result_image, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        img_crop = thresh[10:-10, 10:-10]

        result_image = cv2.copyMakeBorder(img_crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT)

        contours = cv2.findContours(result_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours[1] if imutils.is_cv3() else contours[0]

        letter_image_regions = []

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if w / h > 1.25:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 4:
            return {"message": "WRONG LETTERS COUNT"}

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        def process_letter(letter_bounding_box):
            x, y, w, h = letter_bounding_box

            letter_image = result_image[y - 2:y + h + 2, x - 2:x + w + 2]

            letter_image = resize_to_fit(letter_image, 20, 20)

            letter_image = np.expand_dims(letter_image, axis=2)
            letter_image = np.expand_dims(letter_image, axis=0)

            prediction = model.predict(letter_image)

            letter = lb.inverse_transform(prediction)[0]
            return letter

        with concurrent.futures.ThreadPoolExecutor() as executor:
            predictions = list(executor.map(process_letter, letter_image_regions))

        captcha_text = "".join(predictions)
        return {"message": captcha_text}
    except Exception as e:
        return {"message": str(e)}

    finally:
        file.file.close()


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image
