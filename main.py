import Model as m
import Match as mth
import ExtractFace as ef


def check_face_match():
    image_one = ef.Extractface().get_image_tensor('1.jpg')
    image_two = ef.Extractface().get_image_tensor('4.jpg')
    image_one_tensor = m.FaceEmbedding().get_face_embedding(image_one)
    image_two_tensor = m.FaceEmbedding().get_face_embedding(image_two)
    match = mth.Facematch().get_confidante_score(image_one_tensor, image_two_tensor)
    return match

if __name__ == '__main__':
    match = check_face_match()