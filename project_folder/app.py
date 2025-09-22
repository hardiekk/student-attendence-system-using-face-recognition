# from flask import Flask, render_template, request, jsonify
# import face_recognition
# import numpy as np
# import cv2
# import base64
# import os
# import pandas as pd
# from datetime import datetime

# app = Flask(__name__, template_folder='templates')

# # Dataset loading with debug prints
# dataset_path = 'dataset'
# known_face_encodings = []
# known_face_names = []

# print("Loading dataset...")
# for student_folder in os.listdir(dataset_path):
#     student_path = os.path.join(dataset_path, student_folder)
#     if os.path.isdir(student_path):
#         for img_name in os.listdir(student_path):
#             img_path = os.path.join(student_path, img_name)
#             image = face_recognition.load_image_file(img_path)
#             encodings = face_recognition.face_encodings(image)
#             if encodings:
#                 known_face_encodings.append(encodings[0])
#                 known_face_names.append(student_folder)
# print(f"Total students loaded: {len(known_face_names)}")

# attendance_file = 'attendance.csv'
# if not os.path.exists(attendance_file):
#     df = pd.DataFrame(columns=['RollNo_Name', 'Time'])
#     df.to_csv(attendance_file, index=False)

# def mark_attendance(name):
#     df = pd.read_csv(attendance_file)
#     if name not in df['RollNo_Name'].values:
#         now = datetime.now()
#         time_string = now.strftime("%Y-%m-%d %H:%M:%S")
#         new_row = pd.DataFrame({'RollNo_Name': [name], 'Time': [time_string]})
#         df = pd.concat([df, new_row], ignore_index=True)
#         df.to_csv(attendance_file, index=False)
#         print(f"Attendance marked for: {name}")
#         return True
#     return False

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/attendance')
# def attendance():
#     df = pd.read_csv(attendance_file)
#     records = df.to_dict(orient='records')
#     return render_template('attendance.html', records=records)

# @app.route('/recognize', methods=['POST'])
# def recognize():
#     data = request.get_json(force=True)
#     if 'image' not in data:
#         print("No image sent in request")
#         return jsonify({'error':'No image sent'}), 400
#     img = decode_base64_image(data['image'])
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     face_locations = face_recognition.face_locations(rgb_img, model='hog')
#     encodings = face_recognition.face_encodings(rgb_img, face_locations)

#     print(f"Faces detected: {len(face_locations)}")

#     recognized_names = []
#     for face_encoding in encodings:
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#         best_match_index = np.argmin(face_distances)
#         if matches[best_match_index]:
#             name = known_face_names[best_match_index]
#             mark_attendance(name)
#             recognized_names.append(name)
#     print(f"Recognized names: {recognized_names}")

#     return jsonify({'names': recognized_names})

# def decode_base64_image(data_url):
#     encoded_data = data_url.split(",")[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__, template_folder='templates')

dataset_path = 'dataset'
known_face_encodings = []
known_face_names = []

print("Loading dataset...")
for student_folder in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student_folder)
    if os.path.isdir(student_path):
        for img_name in os.listdir(student_path):
            img_path = os.path.join(student_path, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(student_folder)
print(f"Total students loaded: {len(known_face_names)}")

attendance_file = 'attendance.csv'
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['RollNo_Name', 'Time'])
    df.to_csv(attendance_file, index=False)

def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    if name not in df['RollNo_Name'].values:
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({'RollNo_Name': [name], 'Time': [time_string]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for: {name}")
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance():
    # Attendance summary stats add kiye
    df_attendance = pd.read_csv(attendance_file)
    present_students = set(df_attendance['RollNo_Name'].values)
    total_students = set(known_face_names)

    absent_students = total_students - present_students

    records = df_attendance.to_dict(orient='records')

    stats = {
        'total_students': len(total_students),
        'present_students': len(present_students),
        'absent_students': len(absent_students),
        'absent_list': sorted(list(absent_students))
    }

    return render_template('attendance.html', records=records, stats=stats)

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json(force=True)
    if 'image' not in data:
        print("No image sent in request")
        return jsonify({'error':'No image sent'}), 400
    img = decode_base64_image(data['image'])
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img, model='hog')
    encodings = face_recognition.face_encodings(rgb_img, face_locations)

    print(f"Faces detected: {len(face_locations)}")

    recognized_names = []
    for face_encoding in encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)
            recognized_names.append(name)
    print(f"Recognized names: {recognized_names}")

    return jsonify({'names': recognized_names})

def decode_base64_image(data_url):
    encoded_data = data_url.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    app.run(debug=True)
