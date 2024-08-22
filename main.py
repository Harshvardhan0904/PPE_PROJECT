from ultralytics import YOLO
import cv2
import numpy as np
import pygame
import pandas as pd
from datetime import datetime, timedelta
import os
import json
pygame.mixer.init()
# alert_sound = pygame.mixer.Sound('awaz.wav')

person_model = YOLO("yolov8l.pt")
equipment_model = YOLO("best.pt")
cap = cv2.VideoCapture(0)  # Replace with your video source or camera index
class_name = ['Helmet', 'Goggles', 'Jacket', 'Gloves', 'Footwear']

# Custom thresholds for each class
custom_thresholds = {
    'Helmet': 0.65,
    'Goggles': 0.90,
    'Jacket': 0.60,
    'Gloves': 0.8,
    'Footwear': 0.70
}

def detect_objects(model, frame, custom_thresholds):
    results = model.predict(frame, save=False)
    boxes = []
    scores = []
    classes = []

    for result in results:
        for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            # Check if cls is within the valid range
            if 0 <= int(cls) < len(class_name):
                item = class_name[int(cls)]
                if score >= custom_thresholds.get(item, 0.65):
                    boxes.append(box)
                    scores.append(score)
                    classes.append(cls)
            else:
                print(f"Warning: Invalid class index {cls}")

    return np.array(boxes), np.array(scores), np.array(classes)

def adjust_box(box, reduction_factor=0.2):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    x1_new = x1 + reduction_factor * width / 2
    y1_new = y1 + reduction_factor * height / 2
    x2_new = x2 - reduction_factor * width / 2
    y2_new = y2 - reduction_factor * height / 2

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def update_excel(missing_items_summary, filename='violations_summary.xlsx', json_filename='violations_summary.json'):
    # Check if the file exists
    if os.path.exists(filename):
        # Load the existing file
        df = pd.read_excel(filename)
    else:
        # Create a new DataFrame if the file does not exist
        columns = ['Date', 'Time', 'Person', 'Helmet', 'Jacket']
        df = pd.DataFrame(columns=columns)

    # Ensure the correct columns are present
    required_columns = ['Date', 'Time', 'Person', 'Helmet', 'Jacket']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Get today's date and current time
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    for item in missing_items_summary:
        person_id = item['Person']
        missing_items = item['Missing Items'].split(", ")

        # Initialize helmet and jacket status
        helmet_status = 'No'
        jacket_status = 'No'

        if 'Helmet' not in missing_items:
            helmet_status = 'Yes'
        if 'Jacket' not in missing_items:
            jacket_status = 'Yes'
        
        if helmet_detected_frames_per_person[person_id] > helmet_not_detected_frames_per_person[person_id]:
            helmet_status = 'Yes'
        if jacket_detected_frames_per_person[person_id] > jacket_not_detected_frames_per_person[person_id]:
            jacket_status = 'Yes'

        # Check if the person already has an entry for today and the same time
        existing_entry = df[(df['Date'] == today) & (df['Time'] == current_time) & (df['Person'] == person_id)]

        if not existing_entry.empty:
            # Update the existing entry
            index = existing_entry.index[0]
            df.at[index, 'Helmet'] = helmet_status
            df.at[index, 'Jacket'] = jacket_status
        else:
            # Create a new entry
            new_entry = {
                'Date': today,
                'Time': current_time,
                'Person': person_id,
                'Helmet': helmet_status,
                'Jacket': jacket_status
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    # Save the DataFrame to an Excel file
    df.to_excel(filename, index=False)



# Add time to the missing_items_summary
def update_missing_items_summary(current_frame_missing_items, missing_items_summary):
    current_time = datetime.now().strftime('%H:%M:%S')
    for item in current_frame_missing_items:
        item['Time'] = current_time
        person_index = item['Person'] - 1
        if person_index >= len(missing_items_summary):
            missing_items_summary.append(item)
        else:
            existing_items = missing_items_summary[person_index]['Missing Items']
            new_items = item['Missing Items']
            combined_items = list(set(existing_items.split(", ") + new_items.split(", ")))
            missing_items_summary[person_index]['Missing Items'] = ", ".join(combined_items)
            missing_items_summary[person_index]['Time'] = current_time


w = 1000
h = 800
fps = 30  # Desired frames per second
frame_delay = int(1000 / fps)  # Delay in milliseconds

alert_playing = False
missing_items_summary = []
helmet_detected_frames_per_person = {}
helmet_not_detected_frames_per_person = {}
jacket_detected_frames_per_person = {}
jacket_not_detected_frames_per_person = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (w, h))

    person_boxes, _, _ = detect_objects(person_model, frame, custom_thresholds={'Person': 0.85})
    equipment_boxes, equipment_scores, equipment_classes = detect_objects(equipment_model, frame, custom_thresholds)

    person_boxes = person_boxes.astype(int)
    safety_equipment_boxes = [(x1, y1, x2, y2, class_name[int(cls)]) for (x1, y1, x2, y2), cls in zip(equipment_boxes.astype(int), equipment_classes)]
    safety_equipment_scores = [score for score, cls in zip(equipment_scores, equipment_classes)]

    safe_status = True
    current_frame_missing_items = []

    for idx, (px1, py1, px2, py2) in enumerate(person_boxes):
        person_id = idx + 1
        if person_id not in helmet_detected_frames_per_person:
            helmet_detected_frames_per_person[person_id] = 0
            helmet_not_detected_frames_per_person[person_id] = 0
        if person_id not in jacket_detected_frames_per_person:
            jacket_detected_frames_per_person[person_id] = 0
            jacket_not_detected_frames_per_person[person_id] = 0

        worn = {'Helmet': False, 'Jacket': False}
        missing_item = []

        for (ex1, ey1, ex2, ey2, equipment), score in zip(safety_equipment_boxes, safety_equipment_scores):
            if equipment in worn:
                ex1, ey1, ex2, ey2 = adjust_box((ex1, ey1, ex2, ey2), reduction_factor=0.5)

            if (px1 <= ex1 <= px2 and py1 <= ey1 <= py2 and px1 <= ex2 <= px2 and py1 <= ey2 <= py2):
                if equipment in worn:
                    worn[equipment] = True

        for item in worn:
            if not worn[item]:
                missing_item.append(item)

        if worn['Helmet']:
            helmet_detected_frames_per_person[person_id] += 1
        else:
            helmet_not_detected_frames_per_person[person_id] += 1

        if worn['Jacket']:
            jacket_detected_frames_per_person[person_id] += 1
        else:
            jacket_not_detected_frames_per_person[person_id] += 1

        safe = all(worn.values())
        warning_label = "SAFE" if safe else "NOT SAFE"
        color = (0, 255, 0) if safe else (0, 0, 255)
        safe_status = safe_status and safe

        if not safe:
            missing_text = ", ".join(missing_item)
            current_frame_missing_items.append({'Person': person_id, 'Missing Items': missing_text})
            cv2.putText(frame, f"Missing: {missing_text}", (px1, py1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Person {person_id} {warning_label}", (px1, py1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)

    # if not safe_status and not alert_playing:
    #     alert_sound.play()
    #     alert_playing = True
    # elif safe_status and alert_playing:
    #     alert_sound.stop()
    #     alert_playing = False

    for (x1, y1, x2, y2, equipment), score in zip(safety_equipment_boxes, safety_equipment_scores):
        label = f"{equipment}: {score * 100:.2f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("PPE Detection", frame)

    # Update the missing items summary
    if current_frame_missing_items:
        for item in current_frame_missing_items:
            person_index = item['Person'] - 1
            if person_index >= len(missing_items_summary):
                missing_items_summary.append(item)
            else:
                existing_items = missing_items_summary[person_index]['Missing Items']
                new_items = item['Missing Items']
                combined_items = list(set(existing_items.split(", ") + new_items.split(", ")))
                missing_items_summary[person_index]['Missing Items'] = ", ".join(combined_items)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Call the update_excel function at the end of your main loop
update_excel(missing_items_summary)
