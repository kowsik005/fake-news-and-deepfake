import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

st.title(" Deepfake Detector")

# -------- Model --------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real/fake
model.eval()

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------- Upload Image --------
uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    label = " Real" if pred == 0 else " Deepfake"
    st.subheader(f"Prediction: {label}")
