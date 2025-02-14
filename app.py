from flask import Flask, request, jsonify, send_file
import os
import io
from PIL import Image
import uuid

app = Flask(__name__)

# Create a folder for storing uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize model to None
model = None

def save_uploaded_file(file):
    """Save uploaded file to disk and return the file path"""
    try:
        filename = str(uuid.uuid4()) + '.png'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        img = Image.open(file)
        img.save(filepath)
        
        return filepath
    except Exception as e:
        raise Exception(f"File save failed: {str(e)}")

@app.route('/upload', methods=['POST'])
def process_image():
    try:    
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        try:
            Image.open(file).verify()
            file.seek(0)
        except:
            return jsonify({'error': 'Invalid image file'}), 400

        # Save the file locally
        filepath = save_uploaded_file(file)
        
        # Get the server's base URL (in production, you'd configure this properly)
        host_url = request.host_url.rstrip('/')
        
        result_data = {
            'original_path': filepath,
            'display_url': f'{host_url}/display/{os.path.basename(filepath)}'
        }

        return jsonify(result_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/display/<filename>', methods=['GET'])
def display_image(filename):
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Image not found'}), 404
            
        return send_file(
            filepath,
            mimetype='image/png',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)