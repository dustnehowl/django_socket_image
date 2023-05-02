import base64
from io import BytesIO
from PIL import Image
import json

with open('data.json', 'r') as f:
    data = json.load(f)

# Convert the JSON data to a string
json_string = json.dumps(data["message"])

image_data = base64.b64decode(json_string)

with open('image.jpg', 'wb') as f:
    f.write(image_data)