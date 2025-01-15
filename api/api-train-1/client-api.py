from flask import Flask, request, jsonify
import flwr as fl
import multiprocessing
import client
import shutil
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
training_status = {"status": "idle"}

def start_flower_client_with_data(train_images_bytes, train_labels_bytes, test_images_bytes, test_labels_bytes):
    global training_status
    training_status["status"] = "training"
    try:
        # Directories for raw and processed data
        raw_data_dir = "./data/MNIST/raw"
        processed_data_dir = "./data/MNIST/processed"

        # Ensure the raw and processed data directories exist
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)

        # Clear raw data directory
        for file_name in os.listdir(raw_data_dir):
            file_path = os.path.join(raw_data_dir, file_name)
            if os.path.isfile(file_path):  # Ensure it's a file
                os.remove(file_path)

        def save_bytes(data_bytes, filename):
            filepath = os.path.join(raw_data_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(data_bytes)
            return filepath

        # Save files as raw files
        save_bytes(train_images_bytes, "train-images-idx3-ubyte")
        save_bytes(train_labels_bytes, "train-labels-idx1-ubyte")
        save_bytes(test_images_bytes, "t10k-images-idx3-ubyte")
        save_bytes(test_labels_bytes, "t10k-labels-idx1-ubyte")

        # Start Flower client
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client.FlowerClient())

        # Training completed
        training_status["status"] = "completed"

    except Exception as e:
        print("Error during training:", e)
        training_status["status"] = f"error: {str(e)}"

    # Move raw files to the processed directory after training completes
    # move_files_to_processed()

def move_files_to_processed():
    raw_data_dir = "./data/MNIST/raw"
    processed_data_dir = "./data/MNIST/processed"
    
    for file_name in os.listdir(raw_data_dir):
        raw_file_path = os.path.join(raw_data_dir, file_name)
        processed_file_path = os.path.join(processed_data_dir, file_name)
        shutil.move(raw_file_path, processed_file_path)
    print("Raw files moved to processed directory.")

@app.route('/train', methods=['POST'])
def train():
    global training_status
    if training_status["status"] == "training":
        return jsonify({"status": "Training already in progress"}), 400

    files = request.files
    train_images = files.get("trainImages")
    train_labels = files.get("trainLabels")
    test_images = files.get("testImages")
    test_labels = files.get("testLabels")

    if not (train_images and train_labels and test_images and test_labels):
        return jsonify({"status": "All four files are required"}), 400

    # Read file contents into memory as bytes
    train_images_bytes = train_images.read()
    train_labels_bytes = train_labels.read()
    test_images_bytes = test_images.read()
    test_labels_bytes = test_labels.read()

    # Start training in a separate process
    process = multiprocessing.Process(target=start_flower_client_with_data, args=(
        train_images_bytes,
        train_labels_bytes,
        test_images_bytes,
        test_labels_bytes,
    ))
    process.start()

    return jsonify({"status": "Training started"}), 200

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(training_status), 200

if __name__ == "__main__":
    app.run(host="localhost", port=4000)
