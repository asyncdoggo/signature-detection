import torch
import streamlit as st
import torchvision
import PIL
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource()
def load_model(device):
    ## load classnames
    with open("class_names.pt","rb") as f:
        class_names = pickle.load(f)



    ## load model Resnet and apply its weights from local file
    weights = torch.load('signature_model.pt', map_location=device)
    
    model = torchvision.models.resnet50(weights=None).to(device)
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False
    num_classes = 48 # Number of signature classes

    new_fc = torch.nn.Linear(2048, num_classes).to(device)
    model.fc = new_fc
    
    model.load_state_dict(weights)
    return model, class_names


def main():
    # Title
    st.title('Signature Classifier')
    st.write('This app uses a Resnet50 model to classify signatures')
    file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    model,classes = load_model(device)

    if file is not None:
        # Convert the file to an RGB image using OpenCV
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        sample = Image.fromarray(opencv_image)


        # sample = Image.open(file)
        sample = transforms.Compose([
        # Greyscale image
        # transforms.Grayscale(),
        # Resize
        transforms.Resize(size=(362, 512)), # size=(362, 512)
        # transforms.RandAugment(),
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
        ])(sample)

        with torch.inference_mode():
            sample = torch.unsqueeze(sample,dim=0).to(device)

            model.eval()
            pred_logit = model(sample)

            pred_prob = torch.softmax(pred_logit.squeeze(),dim=0)

            pred_item = torch.argmax(pred_prob).item()

            pred_class = classes[str(pred_item)]

            st.write(f'Prediction: {pred_class}')

            ## display image and prediction
            st.image(opencv_image, caption=f'Prediction: {pred_class}', use_column_width=True)




if __name__ == '__main__':
    main()


