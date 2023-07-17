import torch
import cv2
from torchvision import transforms


class Extractface():
    def __init__(self):
        pass

    def extract_face_from_image(self, image_name):
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
            cv2.imwrite('face.jpg', faces)

    def get_image_tensor(self, image_name):
        self.extract_face_from_image(image_name)
        self.resize_image("face.jpg")
        image = cv2.imread("face.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # cv2.imshow("face", image)
        # cv2.waitKey()
        tensor = transforms.ToTensor()
        image_tensor = tensor(image)

        # print(image_tensor.shape)
        return image_tensor

    def resize_image(self, image_name):
        width = 512
        height = 512
        dim = (width, height)

        image = cv2.imread(image_name)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite('face.jpg', resized)


# obj = Extractface()
# obj.get_image_tensor('1.jpg')
