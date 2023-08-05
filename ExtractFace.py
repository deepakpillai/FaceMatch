import torch
import cv2
from torchvision import transforms
from PIL import Image

class Extractface():
    def __init__(self):
        pass

    def extract_face_from_image(self, image_name, outpt_name):
        # Read the input image
        img = cv2.imread(image_name)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces and crop the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            faces = img[y:y + h, x:x + w]
            out_name = f"{outpt_name}.jpg"
            cv2.imwrite(out_name, faces)

    def get_image_tensor(self, image_name, outpt_name):
        self.extract_face_from_image(image_name, outpt_name)
        out_name = f"{outpt_name}.jpg"
        self.resize_image(out_name)
        image = Image.open(out_name)
        # image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        image_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        pre_processed_data = image_transforms(image)
        return pre_processed_data.unsqueeze(0)

    def resize_image(self, image_name):
        width = 512
        height = 512
        dim = (width, height)

        image = cv2.imread(image_name)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_name, resized)


# obj = Extractface()
# obj.get_image_tensor('1.jpg')
