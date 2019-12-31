import os
from flask import Flask, render_template, request, jsonify, redirect
from SearchEngine.colordescriptor import ColorDescriptor
from SearchEngine.texturedescriptor import TextureDescriptor
from SearchEngine.searcher import Searcher
from flask.helpers import flash
from werkzeug.utils import secure_filename
import secrets
import flask
import json
import numpy as np
import cv2

# create flask instance
app = Flask(__name__)
app.secret_key = "super secret key"


INDEX = os.path.join(os.path.dirname(__file__), 'index.csv')
app.config["UPLOAD_DIRECTORY"] = "static\\uploads"


# main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST', 'GET'])
def search():

    if request.method == "POST":
        RESULTS_ARRAY = []
        if request.files["image"]:
            image = request.files["image"]

            image_dir_name = secrets.token_hex(16)

            image.save(os.path.join(
                app.config["UPLOAD_DIRECTORY"], image.filename))

            image_read = cv2.imread(
                'static\\uploads\\'+image.filename)
            # return image

            # try:
            cd = ColorDescriptor((8, 12, 3))
            ht= TextureDescriptor()
            features = list(np.array(cd.describe(image_read)))
            features = features+list(np.array(ht.extract_features(image_read)))
            # perform the search
            searcher = Searcher(INDEX)
            results = searcher.search(features)
            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})
            # return success
            results = RESULTS_ARRAY[::]
            r = json.dumps(results)
            # loaded_r = jsonify(json.loads(r))

            return render_template("index.html", jsonResult=json.loads(r), image=image.filename)
            # return loaded_r
        else:
            return render_template('index.html')


# run!
if __name__ == '__main__':
    app.run('127.0.0.1', debug=True)
