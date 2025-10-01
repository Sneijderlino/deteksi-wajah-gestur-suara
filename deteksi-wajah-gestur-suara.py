import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import os
import time
import threading

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)


last_gesture = ""
gesture_buffer = []
face_visible = False
last_frame_time = 0
target_fps = 60  

GESTURE_MAP = {
    (1,1,1,1,1): "Halo",
    (0,1,0,0,0): "Nama saya Sneijderlino",
    (0,1,1,0,0): "Kata kata buat hari ini",
    (1,0,0,0,0): "Jangan takut gagal, takutlah untuk tidak mencoba.",
    (0,0,0,0,1): "Gagal itu bagian dari proses. Tapi tidak mencoba? Itu kehilangan kesempatan sebelum berjuang ",
    (0,1,0,0,1): "Salam",
}


def speak(text):
    def run():
        filename = "voice.mp3"
        tts = gTTS(text=text, lang="id")
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    threading.Thread(target=run, daemon=True).start()


def deteksi_jari(hand_landmarks, hand_label):
    status_jari = {}
    if hand_label == "Right":
        status_jari["Jempol"] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
    else:
        status_jari["Jempol"] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0

    status_jari["Telunjuk"]   = 1 if hand_landmarks.landmark[8].y  < hand_landmarks.landmark[6].y  else 0
    status_jari["Tengah"]     = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    status_jari["Manis"]      = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    status_jari["Kelingking"] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0

    return status_jari


def kenali_gerakan(status_jari):
    kode = (
        status_jari["Jempol"],
        status_jari["Telunjuk"],
        status_jari["Tengah"],
        status_jari["Manis"],
        status_jari["Kelingking"]
    )
    if kode == (0,0,0,0,0):
        return ""
    return GESTURE_MAP.get(kode, "")


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, target_fps)

while cap.isOpened():
    now = time.time()
    if now - last_frame_time < 1/target_fps:
        time.sleep(0.0005)  
        continue
    last_frame_time = now

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb_frame)
    gesture_text = ""

    if results_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )
            hand_label = handedness.classification[0].label
            status_jari = deteksi_jari(hand_landmarks, hand_label)
            gesture_text = kenali_gerakan(status_jari)


    if gesture_text:
        gesture_buffer.append(gesture_text)
        if len(gesture_buffer) > 3:  
            gesture_buffer.pop(0)
        if gesture_buffer.count(gesture_text) >= 2:  
            if gesture_text != last_gesture:
                last_gesture = gesture_text
                speak(gesture_text)
            cv2.putText(frame, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)


    results_face = face_detection.process(rgb_frame)
    if results_face.detections:
        if not face_visible:
            speak("Wajah terdeteksi")
            face_visible = True
        for detection in results_face.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
    else:
        face_visible = False

    cv2.imshow("Program Deteksi Wajag Dan Gestur by Sneijderlino", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()