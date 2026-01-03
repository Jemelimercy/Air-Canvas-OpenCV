import cv2
import numpy as np
import mediapipe as mp

# 1. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# 2. Setup Canvas and Drawing Variables
# We'll initialize the canvas as None and create it once we know the webcam resolution
canvas = None
prev_x, prev_y = 0, 0
color = (255, 0, 255) # Purple ink
thickness = 5

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape

    # Initialize canvas with the same size as the webcam feed if not already done
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    # 3. Process Hand Landmarks
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            landmarks = hand_lms.landmark
            
            # Get coordinates for Index (8) and Middle (12) tips
            ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
            mx, my = int(landmarks[12].x * w), int(landmarks[12].y * h)

            # Check which fingers are up
            # (Tip Y is less than the joint Y because 0,0 is top-left)
            index_up = landmarks[8].y < landmarks[6].y
            middle_up = landmarks[12].y < landmarks[10].y
            ring_up = landmarks[16].y < landmarks[14].y
            pinky_up = landmarks[20].y < landmarks[18].y
            thumb_up = landmarks[4].x < landmarks[3].x # Basic thumb check

            # --- GESTURE LOGIC ---

            # A. CLEAR GESTURE: All fingers up (Palm)
            if index_up and middle_up and ring_up and pinky_up:
                canvas = np.zeros((h, w, 3), np.uint8)
                prev_x, prev_y = 0, 0

            # B. SELECTION MODE: Index and Middle are both up
            elif index_up and middle_up:
                prev_x, prev_y = 0, 0 # Reset to prevent "jumping" lines
                cv2.circle(img, (ix, iy), 10, (255, 255, 255), cv2.FILLED)

            # C. DRAWING MODE: Only Index is up
            elif index_up and not middle_up:
                cv2.circle(img, (ix, iy), 10, color, cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy
                
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), color, thickness)
                prev_x, prev_y = ix, iy
            
            else:
                prev_x, prev_y = 0, 0

    # 4. Overlay Canvas onto the Webcam Feed
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    
    # Black out the area on the webcam feed where the drawing is
    img = cv2.bitwise_and(img, img_inv)
    # Add the colorful canvas lines into those blacked-out areas
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Air Canvas", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()