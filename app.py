from flask import Flask, request, render_template, redirect, url_for, g
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image, target_size):
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def get_model():
    if 'model' not in g:
        g.model = tf.keras.models.load_model('model/model.h5')
        optimizer = Adam(learning_rate=0.01)
        g.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return g.model

@app.teardown_appcontext
def teardown_model(exception):
    model = g.pop('model', None)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath)
            prepared_image = prepare_image(image, target_size=(224, 224))
            model = get_model()
            # g.model.compile(optimizer=tf.keras.optimizers.adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            predictions = model.predict(prepared_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            probabilities = predictions[0]
            return render_template('result.html', label=predicted_class, probabilities=probabilities, image_url=filepath)
        except Exception as e:
            print(f"Error: {e}")
            return str(e), 500
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, redirect, url_for, g
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(image, target_size):
    image = image.resize(target_size)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    return image

def get_model():
    if 'model' not in g:
        g.model = tf.keras.models.load_model('model/my_model_44.h5')
        # optimizer = Adam(learning_rate=0.01)
        # g.model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return g.model

@app.teardown_appcontext
def teardown_model(exception):
    model = g.pop('model', None)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = Image.open(filepath)
        prepared_image = prepare_image(image, target_size=(224, 224))
        print("load model")
        model = get_model()
        print("get model success?")
        # g.model.compile(optimizer=tf.keras.optimizers.adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        # predictions = model.predict(prepared_image)
        # predicted_class = np.argmax(predictions, axis=1)[0]
        # probabilities = predictions[0]
        return render_template('result.html', label=0, probabilities=0, image_url=filepath)
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

@app.route('/', methods=['GET'])
def index():
    print("hehe")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
