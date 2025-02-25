## 📌 Overview
This Face Recognition Attendance System uses DeepFace for accurate face recognition and automated attendance marking. It processes individual and group photos, identifies registered individuals, and records attendance in a CSV file.

## 🚀 Features
Face Detection & Recognition: Uses DeepFace (Facenet model) for high-accuracy face recognition.
Automated Attendance Marking: Recognizes individuals and logs attendance with timestamps.
Supports Group Photos: Detects multiple faces in a single image and marks attendance accordingly.
Error Handling & Logging: Handles unreadable images, detection failures, and mismatched faces.
Customizable Encoding: Allows training with multiple images for improved recognition accuracy.

## 🛠️ Tech Stack
- Python
- DeepFace (Facenet)
- OpenCV
- NumPy
- Pandas
- CSV for Attendance Logging

## 📂 Project Structure
bash
📁 Face-Recognition-Attendance  
 ├── 📂 model/             # Stores trained face encodings  
 ├── 📂 daily_photos/      # Contains images for daily attendance processing  
 ├── encodings.pkl        # Pickle file with trained face encodings  
 ├── attendance.csv       # CSV file where attendance is recorded  
 ├── train.py             # Script to train and save face encodings  
 ├── recognize.py         # Processes images and marks attendance  
 ├── README.md            # Project documentation  

 
## 🔥 How It Works
1. Train the Model:
Run train.py to store face encodings of registered individuals.

2. Process Attendance:
Place images in the daily_photos/ folder.
Run recognize.py to detect and log attendance.

3. View Attendance Records:
Open attendance.csv to see logged records.

## 📌 Installation
1. Clone this repository:
sh
git clone https://github.com/Mohitbkd/Face_Recognition_Attendance_System.git
cd Face-Recognition-Attendance

2. Install dependencies:
sh
pip install deepface opencv-python numpy pandas

3. Train the model and run the recognition script.

## 📝 Future Enhancements
- Integrate real-time webcam detection.
- Store attendance in a database (MySQL/PostgreSQL).
- Develop a Flask-based Web Interface.

🤝 Contributing
Feel free to open issues or submit PRs to improve the project!
