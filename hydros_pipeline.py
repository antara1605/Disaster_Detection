import time
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# 1. PIPELINE CONFIGURATION
# ==========================================
# Using your exact local paths
WATCH_FOLDER = r"D:\Hydros_Ground_Station_final\Hydros_Ground_Station_New\Incoming_Images" 
MODEL_PATH = r"D:\Hydros_Ground_Station_final\Hydros_Ground_Station_New\hydros_resnet50_model_1.pth"

CLASS_NAMES = ['flood', 'forest_fire', 'landslide', 'random_images']

# ==========================================
# 2. INITIALIZE AND LOAD THE MODEL
# ==========================================
print("[*] Initializing Hydros inference engine...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print(f"[*] Model loaded successfully on {device}. Ready for incoming data.")

# ==========================================
# 3. ANALYSIS FUNCTION (Runs on new image)
# ==========================================
def process_new_image(image_path):
    try:
        time.sleep(1) # Brief pause to ensure file finishes saving
        
        img = Image.open(image_path).convert('RGB')
        
        # Display the image on your screen using your default image viewer
        img.show()
        
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probabilities, 1)
            
        pred_class = CLASS_NAMES[preds.item()]
        confidence = conf.item() * 100
        
        print("\n" + "="*50)
        print(f"NEW IMAGE DETECTED: {os.path.basename(image_path)}")
        print(f"Classification : {pred_class.upper()} ({confidence:.2f}%)")
        
        if pred_class == "random_images":
            print("Status: SAFE 🟢 (No disaster detected. Dropping image.)")
        else:
            print("Status: HAZARD ALERT 🚨")
            
        print("="*50)
        
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")

# ==========================================
# 4. FOLDER MONITORING SYSTEM
# ==========================================
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_new_image(event.src_path)

if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    
    print(f"\n[*] Pipeline Active: Watching '{WATCH_FOLDER}' for new images...")
    print("[*] Press Ctrl+C to stop the system.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[*] Pipeline shut down manually.")
        
    observer.join()