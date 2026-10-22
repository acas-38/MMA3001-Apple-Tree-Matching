
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Find all PNG images in the Data folder
data_folder = Path(__file__).resolve().parent.parent / "Data"
files = sorted(data_folder.rglob("*.png"))

print("Total images found:", len(files))

if not files:
    raise FileNotFoundError("No PNG images found in the Data folder")

# Display the first five images
for file in files[:5]:
    image = Image.open(file)
    data = np.array(image)

    print("\nFile:", file.name)
    print("Image type:", image.mode)
    print("Image dimensions:", image.size)
    print("Array shape:", data.shape)
    print("Data type:", data.dtype)
    print("Minimum value:", data.min())
    print("Maximum value:", data.max())

    plt.figure(figsize=(7, 4))
    plt.imshow(image, cmap="gray" if image.mode in ("L", "I", "I;16") else None)
    plt.title(file.name)
    plt.axis("off")
    plt.show()
