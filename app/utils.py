from io import BytesIO
from PIL import Image
from typing import Literal
import numpy as np
import soundfile


def audio_array_to_buffer(audio_array: np.array, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    soundfile.write(buffer, audio_array, sample_rate, format="wav")
    buffer.seek(0)
    return buffer


def img_to_bytes(image: Image.Image, img_format: Literal["PNG", "JPEG"] = "PNG") -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=img_format)
    return buffer.getvalue()