import os
import cv2
import datetime
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from csv_utils import mark_attendance
from insightface.app import FaceAnalysis
import shutil
import sys

# Detection sizes in order of priority (Fastest → Most accurate)
DET_SIZES = [(640, 640), (800, 800), (1024, 1024)]

# Directory paths
UPLOADED_IMAGES_PATH = "uploaded_group_images"
PROCESSED_PATH = "processed"
ENCODINGS_FILE = "encodings.pkl"
SIMILARITY_THRESHOLD = 0.6

def load_encodings():
    """Load encodings from encodings.pkl file"""
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        print("✅ Loaded encodings successfully.")
        return data["encodings"], data["names"]
    except FileNotFoundError:
        print("❌ Error: encodings.pkl file not found! Please generate encodings first.")
        sys.exit(1)  # Exit the program with an error code

def detect_faces_with_fallback(img):
    """Try detecting faces using multiple det_size values"""
    for det_size in DET_SIZES:
        print(f"\n🚀 Trying det_size={det_size}...")

        # Initialize FaceAnalysis with current detection size
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=det_size)

        faces = app.get(img)  # Detect faces

        if faces:
            print(f"✅ {len(faces)} faces detected with det_size={det_size}")
            return faces  # Return as soon as detection is successful

        print(f"⚠️ No faces detected with det_size={det_size}, trying next...")

    print("❌ No faces detected with any det_size!")
    return []  # Return empty list if all attempts fail

def process_attendance():
    """Process test images for attendance"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    print(f"📅 Today's date: {today}")

    known_encodings, known_names = load_encodings()
    if len(known_encodings) == 0:
        print("❌ No known encodings found! Attendance processing aborted.")
        return

    today_processed_dir = os.path.join(PROCESSED_PATH, today)
    os.makedirs(today_processed_dir, exist_ok=True)
    print(f"📂 Created directory: {today_processed_dir}")

    try:
        knn = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn.fit(known_encodings)
        print("🧠 KNN model fitted with cosine similarity.")
    except Exception as e:
        print(f"⚠️ KNN cosine similarity failed: {e}")
        print("🔄 Switching to Euclidean distance...")
        knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        knn.fit(known_encodings)
        print("✅ KNN model fitted with Euclidean distance.")

    for img_file in os.listdir(UPLOADED_IMAGES_PATH):
        img_path = os.path.join(UPLOADED_IMAGES_PATH, img_file)
        if not os.path.isfile(img_path):
            continue

        print(f"🔍 Processing image: {img_file}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Unable to read {img_path}, skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        faces = detect_faces_with_fallback(img)  # Try multiple det_size values

        if not faces:  # If no faces were detected, move the image and continue
            print(f"⚠️ No faces detected in {img_file}. Moving to processed folder.")
            shutil.move(img_path, os.path.join(today_processed_dir, img_file))
            continue

        for face in faces:
            if face.embedding is None:
                print("⚠️ Skipping face: No embedding generated.")
                continue

            distances, indices = knn.kneighbors([face.embedding])
            similarity_score = 1 - distances[0][0]  # Cosine similarity score (higher = better match)

            # print(f"📏 Confidence Score: {similarity_score:.4f}")

            if similarity_score > SIMILARITY_THRESHOLD:
                matched_name = known_names[indices[0][0]]
                print(f"✅ Match found: {matched_name}, Confidence Score: {similarity_score:.4f}")
                mark_attendance(matched_name)
            # else:
                # print(f"⚠️ No match found for the detected face. Confidence Score: {similarity_score:.4f}")

        # Move processed image
        shutil.move(img_path, os.path.join(today_processed_dir, img_file))

    print("\n🎉 Attendance processing completed!")

if __name__ == "__main__":
    process_attendance()
