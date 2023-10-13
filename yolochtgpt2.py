import cv2
import numpy as np

# Load YOLOv7-w6-pose model (update these paths)
yolo_net = cv2.dnn.readNet(r"C:\Users\SREERAJ\OneDrive\Desktop\falldet\yolov7-w6-pose.pt", r'C:\Users\SREERAJ\OneDrive\Desktop\falldet\yolov7\cfg')
classes = []  # List of classes (add relevant class names if needed)

# Load video
cap = cv2.VideoCapture(r"C:\Users\SREERAJ\OneDrive\Desktop\falldet\fallvideo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    yolo_net.setInput(blob)
    layer_names = yolo_net.getUnconnectedOutLayersNames()
    outputs = yolo_net.forward(layer_names)

    # Process the detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust confidence threshold as needed
                # Get object coordinates and draw a bounding box
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                class_name = classes[class_id]
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
