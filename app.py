from flask import Flask, request, jsonify
import cv2
import numpy as np
import face_recognition as fr

path = 'ImageDataset'
images = []
ClassNames = []
KnownEncodeList=[]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Hello World"})

@app.route('/upload_image_and_user', methods=['POST'])
def upload_image_and_user():
    user_id = request.form.get('user_id')
    img = request.files['image_data'].read()
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encode = fr.face_encodings(img)[0]
    KnownEncodeList.append(encode)
    ClassNames.append(user_id)

    return f"Image data uploaded successfully for user ID: {user_id}"

@app.route('/recognize_user_id', methods=['POST'])
def recognize_user_id():
    img = request.files['image_data'].read()
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    CurrentFrame = fr.face_locations(img)
    EncodeCurrentFrame = fr.face_encodings(img, CurrentFrame)
    names = []
    resultId=''
    for EncodeFace, FaceLoc in zip(EncodeCurrentFrame, CurrentFrame):
        matches = fr.compare_faces(KnownEncodeList, EncodeFace)
        Facedis = fr.face_distance(KnownEncodeList, EncodeFace)
        matchIndex = np.argmin(Facedis)

        if matches[matchIndex]:
            names.append(ClassNames[matchIndex])
            resultId=ClassNames[matchIndex]
        else:
            names.append("Unknown")

    return jsonify({"user_id": resultId})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
