import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

# ---------------- CLASS NAMES ----------------

class_names = [
    'Abyssinian', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay',
    'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau',
    'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
    'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland',
    'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue',
    'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
    'Siamese', 'Sphynx', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

# ---------------- TRANSFORM ----------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ---------------- CNN MODEL ----------------

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 37)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)

        return x

# ---------------- LOAD MODEL ----------------

device = torch.device("cpu")

model = CNN()

model.load_state_dict(
    torch.load(
        "pet_classifier.pth",
        map_location=device
    )
)

model.eval()

# ---------------- STREAMLIT UI ----------------

st.title("🐶 Pet Breed Classifier")

st.write("Upload a cat or dog image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    input_image = transform(image)

    input_image = input_image.unsqueeze(0)

    # Prediction
    with torch.no_grad():

        outputs = model(input_image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]

    st.subheader(f"Prediction: {predicted_class}")

    st.write(f"Confidence: {confidence.item()*100:.2f}%")