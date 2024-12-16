from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

# Load InceptionV3 as the feature extractor
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
image_encoder = Model(inputs=base_model.input, outputs=base_model.output)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Directory for uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model_new.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Constants
MAX_LENGTH = 34
STARTSEQ = tokenizer.word_index['startseq']
ENDSEQ = tokenizer.word_index['endseq']

# Function: Convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function: Preprocess Image
def preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function: Predict Caption (Beam Search)
def generate_caption(model, tokenizer, image_path, max_length=MAX_LENGTH, beam_width=3):
    image_feature = preprocess_image(image_path)
    feature = image_encoder.predict(image_feature, verbose=0)

    seqs = [([STARTSEQ], 0.0)]
    completed = []

    for _ in range(max_length):
        all_candidates = []
        for seq, score in seqs:
            if seq[-1] == ENDSEQ:
                completed.append((seq, score))
                continue

            sequence = pad_sequences([seq], maxlen=max_length, padding='post')
            preds = model.predict([feature, sequence], verbose=0)[0]

            top_words = np.argsort(preds)[-beam_width:]
            for w in top_words:
                new_seq = seq + [w]
                new_score = score - np.log(preds[w])
                all_candidates.append((new_seq, new_score))

        ordered = sorted(all_candidates, key=lambda x: x[1])
        seqs = ordered[:beam_width]

    best_seq = sorted(completed, key=lambda x: x[1])[0][0] if completed else seqs[0][0]
    caption = ' '.join([idx_to_word(w, tokenizer) for w in best_seq if w != STARTSEQ and w != ENDSEQ])
    return caption.capitalize()

# Route: Home Page
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            caption = generate_caption(model, tokenizer, file_path)
            return render_template('index.html', image_path=file_path, caption=caption)
    return render_template('index.html', image_path=None, caption=None)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
