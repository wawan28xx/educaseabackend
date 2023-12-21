from flask import Flask, request, jsonify
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import requests

app = Flask(__name__)

# Path model
model_path = 'educaseamodel.h5'
IMAGE_SIZE = (299, 299, 3)

# Load model
loaded_model = load_model(model_path)

# List label kelas ikan
class_names = ["bannerfish", "bluetang", "clownfish", "dotyback", "yellowtang"]

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def get_fish_description(class_label):
    # URL Endpoint API educasea
    api_url = f'https://educasea-authlogin-default-rtdb.asia-southeast1.firebasedatabase.app/seaFish/{class_label}.json'
    response = requests.get(api_url)

    if response.status_code == 200:
        fish_data = response.json()

        # Respons format Flask dengan key API firebase
        description = {
            'class': class_label,
            'audio_url': fish_data['audio'],
            'description': fish_data['description'],
            'image_url': fish_data['imageUrl'],
            'name': fish_data['name']
        }

        return description
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima file gambar dari request
        file = request.files['image']
        image_path = 'temp.jpg'  # Simpan sementara di file temporari
        file.save(image_path)

        # Praproses gambar
        input_data = preprocess_image(image_path)

        # Prediksi kelas gambar
        predictions = loaded_model.predict(input_data)

        # Ambil indeks kelas dengan nilai prediksi tertinggi
        predicted_class = np.argmax(predictions)
        class_label = class_names[predicted_class]
        confidence = float(predictions[0, predicted_class])

        # Dapatkan deskripsi ikan dari API Firebase
        fish_description = get_fish_description(class_label)

        if fish_description:
            # Menambahkan hasil prediksi dan deskripsi ikan ke respons JSON
            result = {
                'class': class_label,
                'confidence': confidence,
                'description': fish_description['description'],
                'audio_url': fish_description['audio_url'],
                'image_url': fish_description['image_url'],
                'name': fish_description['name']
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'Ups, sayangnya ikan yang kamu foto belum ada di data kami. Jangan menyerah ayo fotoin ikan lainnya..'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
