import os
import pickle
import cv2
import numpy as np
import re
from insightface.app import FaceAnalysis

# Directory for storing student images
STUDENT_IMAGES_PATH = "students_images"
ENCODINGS_FILE = "encodings.pkl"

# Initialize FaceAnalysis
print("🔄 Initializing FaceAnalysis...")
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ FaceAnalysis initialized.")

def normalize_name(name):
    """Normalize folder names to uppercase while keeping spaces"""
    normalized = re.sub(r'[^a-zA-Z ]', '', name).upper()
    print(f"🔤 Normalized name: {name} -> {normalized}")
    return normalized

def load_or_generate_encodings():
    """Load saved encodings if available, otherwise generate new ones"""
    print("📂 Checking for existing encodings...")
    if os.path.exists(ENCODINGS_FILE) and os.path.getsize(ENCODINGS_FILE) > 0:
        with open(ENCODINGS_FILE, "rb") as file:
            data = pickle.load(file)
        known_encodings = data.get('encodings', [])
        known_names = data.get('names', [])
        processed_files = set(data.get('files', set()))  # Ensure it's a set
        print(f"✅ Loaded {len(known_names)} known faces from cache.")
    else:
        known_encodings, known_names, processed_files = [], [], set()
        print("🚀 No previous encodings found. Starting fresh.")

    # Get all student image file paths
    print("📸 Scanning student image folders...")
    current_files = {os.path.join(person, img) for person in os.listdir(STUDENT_IMAGES_PATH) 
                     if os.path.isdir(os.path.join(STUDENT_IMAGES_PATH, person))
                     for img in os.listdir(os.path.join(STUDENT_IMAGES_PATH, person))}
    print(f"🔍 Found {len(current_files)} images in total.")

    new_files = current_files - processed_files
    print(f"🆕 {len(new_files)} new images to process.")

    if not new_files:
        print("✅ No new images found. Using cached encodings.")
        return (np.array(known_encodings) if known_encodings else np.empty((0, 512)), 
                np.array(known_names) if known_names else np.array([]), 
                processed_files)

    # Track how many images per person
    person_image_count = {}

    for file in new_files:
        person_dir, img_file = os.path.split(file)
        person_name = normalize_name(person_dir)
        img_path = os.path.join(STUDENT_IMAGES_PATH, file)
        print(f"📷 Processing image: {img_path}")

        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Unable to read {img_path}, skipping.")
                continue

            print("🎨 Converting image to RGB format...")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print("🔍 Running face detection...")
            faces = app.get(img)

            if not faces or len(faces) == 0:
                print(f"⚠️ No face detected in {img_path}, skipping.")
                continue

            print("🧠 Extracting facial embeddings...")
            known_encodings.append(faces[0].embedding)
            known_names.append(person_name)
            processed_files.add(file)
            print(f"✅ Processed {img_file} for {person_name}")

            # Track the number of images processed for each person
            if person_name in person_image_count:
                person_image_count[person_name] += 1
            else:
                person_image_count[person_name] = 1

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    print("💾 Saving updated encodings...")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names, "files": list(processed_files)}, f)
    print(f"✅ Updated encodings saved with {len(known_names)} faces.")

    # Print the number of images processed for each person
    print("📊 Image processing summary per person:")
    for person_name, count in person_image_count.items():
        print(f"{person_name}: {count} image(s) processed")

    return (np.array(known_encodings) if known_encodings else np.empty((0, 512)), 
            np.array(known_names) if known_names else np.array([]), 
            processed_files)

if __name__ == "__main__":
    print("🚀 Starting encoding process...")
    load_or_generate_encodings()
    print("🏁 Encoding process complete.")

