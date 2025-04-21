import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from matplotlib import pyplot as plt
from capsule import CapsuleModel, getMNIST
from threading import Thread

app = Flask(__name__)
CORS(app)


class CapsuleNetworkManager:
    def __init__(self):
        self.loading_complete = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.capdim = [(72, 4), (20, 15), (10, 20)]
        self.model = CapsuleModel(capdim=self.capdim, conv=[8, 16], in_channels=1).to(self.device)
        self.model.load_state_dict(torch.load('uploads/03.mo'))
        self.model.eval()
        self.preloaded_data = []
        self.preloaded_images = []
        self.current_loading_index = 0
        self.activations = []

    def preload_data(self, batch_size=10):
        print("Preloading data")
        data = getMNIST(batch_size=batch_size)
        for images, labels in data["train"]:
            with torch.no_grad():
                images, labels = images.to(self.device), labels.to(self.device)
                for i in range(images.shape[0]):
                    print(i)
                    c = self.model.getC(images[i].unsqueeze(0))
                    caps = self.model(images[i].unsqueeze(0))
                    self.activations = self.model.get_activations().cpu().numpy().squeeze(0)

                    preds = (caps ** 2).sum(dim=-1).argmax(-1).cpu()
                    original_image = images[i].cpu().numpy().squeeze()

                    c_processed = [tensor.cpu().numpy().squeeze().tolist() for tensor in c]
                    self.preloaded_data.append((c_processed, preds.item()))
                    self.preloaded_images.append(original_image.tolist())

                    plt.figure()
                    plt.imshow(original_image, cmap='gray')
                    plt.axis('off')

                    assets_dir = 'assets'
                    if not os.path.exists(assets_dir):
                        os.makedirs(assets_dir)

                    image_path = os.path.join(assets_dir, f"{self.current_loading_index}.png")
                    plt.savefig(image_path)
                    plt.close()

                    self.current_loading_index += 1
                    if self.current_loading_index >= 10:
                        self.loading_complete = True
                        print(self.loading_complete)
                        return

network_manager = CapsuleNetworkManager()

@app.route('/preload', methods=['POST'])
def preload_data():
    batch_size = request.json.get('batch_size', 10)
    preload_thread = Thread(target=network_manager.preload_data, args=(batch_size,))
    preload_thread.start()
    return jsonify({"message": "Data preloading started"}), 202

@app.route('/data', methods=['GET'])
def get_preloaded_data():
    if not network_manager.loading_complete:
        return jsonify({"message": "Data is still loading"}), 202

    data = {
        "images": [f"assets/{i}.png" for i in range(len(network_manager.preloaded_images))],
        "couplings": [
            {"coupling": c, "prediction": p} for c, p in network_manager.preloaded_data
        ],
    }
    return jsonify(data), 200

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)
