from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv
import os

def train_model(dataset_dir):
    # Load the nano YOLOv8 model (super fast, great for real-time video games)
    model = YOLO('yolo26n.pt') 

    # Train the model using your downloaded Roboflow dataset
    # Note: 'data.yaml' is the file inside the folder you download from Roboflow
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    # Start the training process
    model.train(
        data=yaml_path,
        epochs=50,            # Number of training loops (increase to 100+ if accuracy is low)
        imgsz=640,            # Standard image size for YOLOv8
        batch=16,             # Number of images processed at once
        device='',            # Leave empty to automatically use your GPU if available
        project='poker_bot',  # The parent folder where results will be saved
        name='yolo26_poker'   # The specific sub-folder for this training run
    )
    
    print("\n--- Training Complete! ---")
    print("Your newly trained weights are located at:")
    print(os.path.abspath("poker_bot/yolov8_poker/weights/best.pt"))
    print("Copy 'best.pt' into your main folder to use it with poker_vision.py!")

def load_dataset():
    load_dotenv()
    rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))
    project = rf.workspace("raeditio").project("WSOP")
    version = project.version(1)
    dataset = version.download("yolo26")
    return dataset

def main():
    dataset = load_dataset()
    train_model(dataset.location)
    
if __name__ == "__main__":
    main()