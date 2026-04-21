import cv2
import mediapipe as mp
import numpy as np
import time

# Init MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup webcam and blank canvas
cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
color = (0, 0, 255)  # Default color = Red
thickness = 5
eraser_mode = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape
            x, y = int(lm[8].x * w), int(lm[8].y * h)  # Index fingertip

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            # Draw line on canvas
            draw_color = (0, 0, 0) if eraser_mode else color
            draw_thickness = 30 if eraser_mode else thickness
            cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, draw_thickness)

            prev_x, prev_y = x, y

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = 0, 0  # Reset if no hand

    # Combine webcam feed + canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(combined, f"Color: {'Eraser' if eraser_mode else str(color)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("FingerDraw", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    elif key == ord('1'):
        color = (0, 0, 255)  # Red
        eraser_mode = False

    elif key == ord('2'):
        color = (0, 255, 0)  # Green
        eraser_mode = False

    elif key == ord('3'):
        color = (255, 0, 0)  # Blue
        eraser_mode = False

    elif key == ord('e'):
        eraser_mode = True

    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print(f"✅ Saved as {filename}")

    elif key == 27:  # ESC
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
