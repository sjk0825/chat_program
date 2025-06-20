import base64
import io
from PIL import Image

def to_base64(image):
    if image is None:
        return ""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")