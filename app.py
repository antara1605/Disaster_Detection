import time
import os
import threading
import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # Required to generate images in the background of a web server
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from flask import Flask, render_template, jsonify, send_from_directory, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# 1. CONFIGURATION & FLASK SETUP
# ==========================================
WATCH_FOLDER = r"D:\Hydros_Ground_Station_final\Hydros_Ground_Station_New\Incoming_Images" 
MODEL_PATH = r"D:\Hydros_Ground_Station_final\Hydros_Ground_Station_New\hydros_resnet50_model_1.pth"
CLASS_NAMES = ['flood', 'forest_fire', 'landslide', 'random_images']

latest_scan_data = {
    "image_name": None,
    "prediction": None,
    "confidence": None
}

app = Flask(__name__)

# ==========================================
# 2. MODEL INITIALIZATION
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

# ==========================================
# 3. ANALYSIS & PATHFINDING LOGIC
# ==========================================
def process_new_image(image_path):
    global latest_scan_data
    
    # Ignore images that the script itself just generated
    if os.path.basename(image_path).startswith("analyzed_"):
        return
        
    try:
        time.sleep(1) # Ensure file is completely saved
        original_img = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(original_img).unsqueeze(0).to(device)
        
        # 1. Model Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probabilities, 1)
            
        pred_class = CLASS_NAMES[preds.item()]
        confidence = conf.item() * 100
        
        print(f"\n[*] Processed {os.path.basename(image_path)} -> {pred_class.upper()} ({confidence:.2f}%)")
        
        # Default filename to send to frontend
        final_display_filename = os.path.basename(image_path)
        
        # 2. Calibrated Pathfinding (Only for Landslides)
        if pred_class == "landslide":
            print("[*] Generating topographical routing map...")
            
            img_resized = original_img.resize((300, 300))
            img_np = np.array(img_resized)
            
            r = img_np[:,:,0].astype(float)
            g = img_np[:,:,1].astype(float)
            b = img_np[:,:,2].astype(float)
            intensity = (r + g + b) / 3.0
            
            # Calibrated heuristic for landslides
            hazard_mask = (r > g + 5) & (r > b + 20) & (intensity > 60)
            
            # Base Cost Map
            cost_map = np.ones((300, 300), dtype=np.float32)
            cost_map[hazard_mask] = 9999.0 
            
            # Topographical Risk Gradient
            hazard_float = hazard_mask.astype(np.float32) * 800.0
            risk_gradient = cv2.GaussianBlur(hazard_float, (35, 35), 0)
            cost_map = cost_map + risk_gradient

            start_point = (299, 150)
            end_point = (0, 150)
            
            try:
                indices, _ = route_through_array(cost_map, start_point, end_point, fully_connected=True)
                indices = np.array(indices)
                
                # Plot the image securely in the background
                plt.figure(figsize=(6, 6))
                plt.imshow(img_np)
                
                hazard_overlay = np.zeros_like(img_np)
                hazard_overlay[hazard_mask] = [255, 0, 0]
                plt.imshow(hazard_overlay, alpha=0.4)
                
                plt.plot(indices[:, 1], indices[:, 0], color='lime', linewidth=4)
                plt.scatter(start_point[1], start_point[0], color='blue', s=80)
                plt.scatter(end_point[1], end_point[0], color='yellow', s=80)
                plt.axis('off')
                
                # Save the new analyzed image
                final_display_filename = "analyzed_" + os.path.basename(image_path)
                save_path = os.path.join(WATCH_FOLDER, final_display_filename)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"[*] Path extracted and saved as {final_display_filename}")
                
            except Exception as e:
                print("[!] Pathfinding obstructed or failed:", e)
                plt.close()

        # Update global dictionary for the website to read
        latest_scan_data["image_name"] = final_display_filename
        latest_scan_data["prediction"] = pred_class
        latest_scan_data["confidence"] = f"{confidence:.2f}"
        
    except Exception as e:
        print(f"[!] Error processing {image_path}: {e}")

# ==========================================
# 4. WATCHDOG & FLASK ROUTES
# ==========================================
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_new_image(event.src_path)

def start_watchdog():
    os.makedirs(WATCH_FOLDER, exist_ok=True)
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    observer.join()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/latest_scan')
def latest_scan():
    return jsonify(latest_scan_data)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(WATCH_FOLDER, filename)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    file = request.files['image']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
    if not ext or ext == '.blob': ext = '.jpg'
        
    filename = f"hydros_feed_{timestamp}{ext}"
    filepath = os.path.join(WATCH_FOLDER, filename)
    file.save(filepath)
    
    return jsonify({"success": True, "filename": filename})

# ==========================================
# 5. FETCH EXISTING IMAGES (NEW FEATURE)
# ==========================================
@app.route('/api/list_incoming_images', methods=['GET'])
def list_incoming_images():
    """Scans the folder and returns a list of all available images, newest first."""
    try:
        # Get all image files, excluding the ones already analyzed by the model
        files = [f for f in os.listdir(WATCH_FOLDER) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                 and not f.startswith('analyzed_')]
                 
        # Sort files so the most recently added images appear first
        files.sort(key=lambda x: os.path.getmtime(os.path.join(WATCH_FOLDER, x)), reverse=True)
        return jsonify({"success": True, "images": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analyze_existing/<filename>', methods=['GET'])
def analyze_existing(filename):
    """Fetches a specific existing image and forces the model to run it."""
    filepath = os.path.join(WATCH_FOLDER, filename)
    if os.path.exists(filepath):
        # Run your existing ML pipeline on this file
        process_new_image(filepath) 
        
        return jsonify({
            "success": True, 
            "message": f"Successfully fetched and analyzed {filename}",
            "data": latest_scan_data
        })
    else:
        return jsonify({"success": False, "error": "Image not found in the folder."}), 404

if __name__ == "__main__":
    watchdog_thread = threading.Thread(target=start_watchdog, daemon=True)
    watchdog_thread.start()
    
    print("\n" + "="*50)
    print("🚀 HYDROS DASHBOARD LIVE WITH PATHFINDING!")
    print("Open your web browser and go to: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)