import torch
from flask import Flask, request, jsonify
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)

checkpoint = "sam_decoder_multi.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=checkpoint).to()
# sam.to(device)
predictor = SamPredictor(sam)

@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    # Check if the request contains an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if the file is an image
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    # Read the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Generate the embedding
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    np.save(file.filename+".npy",image_embedding)
    
    # Convert the embedding to a list
    embedding_list = image_embedding.tolist()
    
    return jsonify({'embedding': embedding_list})
@app.route('/points', methods=['POST'])
def getPolygon():
    # points =  request.points
    # print(points)
    predictionTuple = predictor.predict()
    return jsonify({'result1': predictionTuple[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)

