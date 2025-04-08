
from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#replace to your own working directory
os.chdir("../Steganography_GANs_inference")
print("current working dir in UI:",os.getcwd())
import io
from steganogan import SteganoGAN
import base64
from inference import encode_image
from inference import decode_image
from inference import initialize



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(),'new5105/temp_uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    message = request.form.get('message', '')
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Save the uploaded image
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
    image.save(input_path)
    
    # Output path
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
    
    # Encode the message
    try:
        # steganogan.encode(input_path, output_path, message)
        # fake encode
        #output_path = 'output.png'
        output_path=encode_image(input_path,message)
        
        # Return the output image
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']


    # Save the uploaded image
    message = decode_image(image)
    print("text recoverd from generated image: ", message)
    decode_path = os.path.join(app.config['UPLOAD_FOLDER'], 'decode.png')
    image.save(decode_path)
    
    # Decode the message
    try:
        # message = steganogan.decode(decode_path)
        # fake decode
        #message = 'This is a super secret message!'
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)