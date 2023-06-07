import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image as image_utils
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (64, 64)
UPLOAD_FOLDER = 'uploads'
vgg16 = load_model('model/sports_cnn_vgg16.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):

    from keras.preprocessing.image import img_to_array
    img  = load_img(file, target_size=IMAGE_SIZE)
    img = img_to_array(img)
    img = img.reshape(1, 64,64,3)
    dic = {0 : 'Archery', 1 : 'Baseball', 2 :"Football", 3:'Tennis'}
    result=np.argmax(vgg16.predict(img)[0])
    output=dic[result]
    
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/uploads/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
