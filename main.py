import Model as m
import Match as mth
import ExtractFace as ef
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="web/")

@app.route('/', defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/get_two_images", methods=['POST'])
def get_two_images():
    if request.method == "POST":
        image1 = request.files["image1"]
        image2 = request.files["image2"]

        image_one_name = "Images/up_image1.jpg"
        image_two_name = "Images/up_image2.jpg"
        # Save the images to disk
        image1.save(image_one_name)
        image2.save(image_two_name)

        match = check_face_match(image_one_name, image_two_name)
        match_str = ""
        if match > 0.70:
            match_str = "Its a match"
        else:
            match_str = "Its not a match"
        data = {
            "message": match_str
        }
        return jsonify(data)

def check_face_match(image_one_name, image_two_name):
    image_one = ef.Extractface().get_image_tensor(image_one_name, "out_1")
    image_two = ef.Extractface().get_image_tensor(image_two_name, "out_2")
    # image_one = ef.Extractface().get_image_tensor('./Images/1.jpg', "out_1")
    # image_two = ef.Extractface().get_image_tensor('./Images/5.jpg', "out_2")
    image_one_tensor = m.FaceEmbedding().get_face_embedding(image_one)
    image_two_tensor = m.FaceEmbedding().get_face_embedding(image_two)
    match = mth.Facematch().get_confidante_score(image_one_tensor, image_two_tensor)
    print(match)
    return match

if __name__ == '__main__':
    app.run(debug=True)