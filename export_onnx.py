from ultralytics import YOLO

def export():
    print("Exporting Zone Model to ONNX...")
    zone_model = YOLO("model_unencrypted/zone_best.pt")
    # Dynamic=False ensures the size is baked in for maximum speed
    zone_model.export(format="onnx", imgsz=640, dynamic=True)
    
    print("Exporting Card Model to ONNX...")
    card_model = YOLO("model_unencrypted/card_n_640.pt")
    card_model.export(format="onnx", imgsz=1024, dynamic=True)
    
    print("\nDone! Check your 'model_unencrypted' folder for the new .onnx files.")

if __name__ == '__main__':
    export()