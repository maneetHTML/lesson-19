import cv2
import mediapipe as mp
import numpy as np
import time
# Initialize Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
last_screenshot_time = 0
cooldown = 2  # seconds
def apply_filter(image, filter_type):
    if filter_type == 'grayscale':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'sepia':
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(image, sepia_filter)
        return np.clip(sepia_img, 0, 255).astype(np.uint8)
    elif filter_type == 'negative':
        return cv2.bitwise_not(image)
    elif filter_type == 'blur':
        return cv2.GaussianBlur(image, (15, 15), 0)
    return image

def finger_up(lm_list, tip_id, pip_id):
    return lm_list[tip_id][1] < lm_list[pip_id][1]

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    filter_type = None
    take_screenshot = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm_list = []
            h, w, _ = frame.shape
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))
            
            if lm_list:
                # Check finger states
                index_up = finger_up(lm_list, 8, 6)
                middle_up = finger_up(lm_list, 12, 10)
                ring_up = finger_up(lm_list, 16, 14)
                pinky_up = finger_up(lm_list, 20, 18)
                thumb_out = lm_list[4][0] > lm_list[3][0] + 20  # crude thumb out detection
                
                # Fist: all fingers down
                if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_out:
                    current_time = time.time()
                    if current_time - last_screenshot_time > cooldown:
                        take_screenshot = True
                        last_screenshot_time = current_time

                # Example filter gestures (optional)
                if index_up and middle_up and not ring_up:
                    filter_type = 'grayscale'
                elif thumb_out:
                    filter_type = 'sepia'
                elif pinky_up and not index_up:
                    filter_type = 'negative'
                elif not index_up and not middle_up and not ring_up and not pinky_up:
                    filter_type = 'blur'

    if filter_type:
        filtered_frame = apply_filter(frame, filter_type)
        if filter_type == 'grayscale':
            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
        frame = filtered_frame
    
    cv2.imshow("Gesture Controlled Photo App", frame)
    
    if take_screenshot:
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
