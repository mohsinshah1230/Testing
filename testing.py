import cv2
from pyzbar.pyzbar import decode
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import sqlite3
from ultralytics import YOLO  # For YOLO models

# Initialize YOLO models
model2 = YOLO('best00.pt')  # Empty Shelf Detector
model5 = YOLO('best5.pt')  # Product Detector
model6 = YOLO('best.pt') 
model7=YOLO('best10.pt')  # Product Detector

# Connect to SQLite database
db_path = 'products1.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

# Capture live video from the camera
cap = cv2.VideoCapture(0)

# Set camera resolution to 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Decode the barcodes
    barcodes = decode(frame)

    # List to store the left bottom coordinates of each barcode and corresponding data
    left_bottom_coordinates = []

    # Process each barcode
    for idx, barcode in enumerate(barcodes, start=1):
        barcode_data = barcode.data.decode('utf-8')
        x, y, w, h = barcode.rect
        left_bottom_coord = (x, y + h)
        left_bottom_coordinates.append((left_bottom_coord, barcode_data))  # Include barcode data

    x_cord = []
    for coord, _ in left_bottom_coordinates:
        for y in coord:
            x_cord.append(y)
            break
    x_cord.sort()
    max_margin = x_cord[0] + 150

    y_axis = [(0, 0)]
    for coord, _ in left_bottom_coordinates:
        for y in coord:
            if y <= max_margin:
                y_axis.append(coord)

    y_axis.sort(key=lambda x: x[1])
    updated_coordinates = [(x, y) if i == 0 else (x, y + 200) for i, (x, y) in enumerate(y_axis)]

    # Loop through the boxes and detect products and empty shelves
    for i in range(len(updated_coordinates) - 1):
        y_start = updated_coordinates[i][1]
        y_end = updated_coordinates[i + 1][1]

        relevant_barcodes = [(coord, data) for coord, data in left_bottom_coordinates if y_start <= coord[1] <= y_end]
        relevant_barcodes.sort(key=lambda x: x[0][0])

        for j in range(len(relevant_barcodes) - 1):
            x_start = relevant_barcodes[j][0][0]
            x_end = relevant_barcodes[j + 1][0][0]
            
            top_left = (x_start, y_start)
            bottom_right = (x_end, y_end)
            box_image = frame[y_start:y_end, x_start:x_end]
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            print(f"Drew rectangle {j + 1} of horizontal slice {i + 1}: {top_left} to {bottom_right}")

            empty_shelf_results = model2(box_image)
            product_results_5 = model5(box_image)
            product_results_6 = model6(box_image)
            product_results_7 = model7(box_image)

            empty_shelf_detected = False
            product_detected = False

            # Check for empty shelves and draw boxes
            for result in empty_shelf_results:
                for det in result.boxes:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x_start + x1, y_start + y1), (x_start + x2, y_start + y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Empty Shelf", (x_start + x1, y_start + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    empty_shelf_detected = True

            # Check for products and draw boxes
            for result in product_results_5 + product_results_6 + product_results_7:
                for det in result.boxes:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(det.cls[0].cpu().numpy())
                    product_name = result.names[class_id]
                    cv2.rectangle(frame, (x_start + x1, y_start + y1), (x_start + x2, y_start + y2), (0, 255, 0), 2)
                    cv2.putText(frame, product_name, (x_start + x1, y_start + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    product_detected = True
                    print(f"Detected product: {product_name} in box {j + 1} of horizontal slice {i + 1}")

            # Check if both product and empty shelf are detected in the same box
            if empty_shelf_detected and product_detected:
                for coord, barcode_data in relevant_barcodes:
                    if x_start <= coord[0] <= x_end:
                        low_inventory_message = f"Low inventory: {barcode_data}"
                        print(low_inventory_message)
                        cv2.putText(frame, low_inventory_message, (x_start + 5, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        break

        # Draw the last segment rectangle
        if relevant_barcodes:
            x_last_start = relevant_barcodes[-1][0][0]
            x_last_end = frame.shape[1]

            box_image = frame[y_start:y_end, x_last_start:x_last_end]
            cv2.rectangle(frame, (x_last_start, y_start), (x_last_end, y_end), (0, 255, 0), 2)

            # Empty shelf and product detection for the last part
            empty_shelf_results = model2(box_image)
            product_results_5 = model5(box_image)
            product_results_6 = model6(box_image)
            product_results_7 = model7(box_image)

            empty_shelf_detected = False
            product_detected = False

            for result in empty_shelf_results:
                for det in result.boxes:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x_last_start + x1, y_start + y1), (x_last_start + x2, y_start + y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Empty Shelf", (x_last_start + x1, y_start + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    empty_shelf_detected = True

            for result in product_results_5 + product_results_6 + product_results_7:
                for det in result.boxes:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                    class_id = int(det.cls[0].cpu().numpy())
                    product_name = result.names[class_id]
                    cv2.rectangle(frame, (x_last_start + x1, y_start + y1), (x_last_start + x2, y_start + y2), (0, 255, 0), 2)
                    cv2.putText(frame, product_name, (x_last_start + x1, y_start + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    product_detected = True

            if empty_shelf_detected and product_detected:
                barcode_data = relevant_barcodes[-1][1]
                low_inventory_message = f"Low inventory: {barcode_data}"
                print(low_inventory_message)
                cv2.putText(frame, low_inventory_message, (x_last_start + 5, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the live video feed with detections
    cv2.imshow('Live Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
