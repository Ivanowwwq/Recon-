import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"

AGE_LIST = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60+']
GENDER_LIST = ['Hombre','Mujer']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

def estimar_edad(frame, box):
    x,y,w,h = box
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return "?"
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), (78.426,87.769,114.896), swapRB=False)
    age_net.setInput(blob)
    p = age_net.forward().argmax()
    return AGE_LIST[p]

def estimar_genero(frame, box):
    x,y,w,h = box
    face = frame[y:y+h, x:x+w]
    if face.size == 0:
        return "?"
    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), (78.426,87.769,114.896), swapRB=False)
    gender_net.setInput(blob)
    p = gender_net.forward().argmax()
    return GENDER_LIST[p]

def estimar_expresion(landmarks):
    pts = [(lm.x, lm.y) for lm in landmarks.landmark]
    izq = np.array(pts[61])
    der = np.array(pts[291])
    sup = np.array(pts[13])
    inf = np.array(pts[14])
    ancho = np.linalg.norm(der-izq)
    alto = np.linalg.norm(inf-sup)
    r = ancho/alto if alto != 0 else 0
    if r > 3.5 and alto > 0.015:
        return "Feliz"
    if alto < 0.010:
        return "Serio"
    return "Neutral"

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        h,w,_ = frame.shape

        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks:
                xs = [lm.x for lm in fl.landmark]
                ys = [lm.y for lm in fl.landmark]
                x1 = int(min(xs)*w)
                y1 = int(min(ys)*h)
                x2 = int(max(xs)*w)
                y2 = int(max(ys)*h)
                box = (x1, y1, x2-x1, y2-y1)

                edad = estimar_edad(frame, box)
                genero = estimar_genero(frame, box)
                expresion = estimar_expresion(fl)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,255), 2)
                label = f"{genero}, {edad}, {expresion}"
                cv2.rectangle(frame, (x1, y1-25), (x1+len(label)*10, y1), (255,0,255), -1)
                cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Reconocimiento", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

last_action_time = 0
cooldown = 1.2

def ejecutar_una_vez():
    global last_action_time
    if time.time() - last_action_time > cooldown:
        last_action_time = time.time()
        return True
    return False

def captura_pantalla():
    img = pyautogui.screenshot()
    name = f"screenshot_{int(time.time())}.png"
    img.save(name)

def captura_camara(frame):
    name = f"cam_{int(time.time())}.jpg"
    cv2.imwrite(name, frame)

def volumen(action):
    if action == "up":
        pyautogui.press("volumeup")
    elif action == "down":
        pyautogui.press("volumedown")
    elif action == "mute":
        pyautogui.press("volumemute")

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands, \
     mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = hands.process(rgb)
        h, w, _ = frame.shape

        if hand_results.multi_hand_landmarks:
            for handLms in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                lm = handLms.landmark

                thumb = lm[4].x < lm[3].x
                index = lm[8].y < lm[6].y
                middle = lm[12].y < lm[10].y
                ring = lm[16].y < lm[14].y
                pinky = lm[20].y < lm[18].y

                fingers = [thumb, index, middle, ring, pinky]

                if fingers == [False, True, False, False, False] and ejecutar_una_vez():
                    volumen("up")

                if fingers == [False, False, False, False, True] and ejecutar_una_vez():
                    volumen("down")

                if fingers == [False, False, False, False, False] and ejecutar_una_vez():
                    volumen("mute")

                if fingers == [True, False, False, False, False] and ejecutar_una_vez():
                    captura_pantalla()
                    captura_camara(frame)

        cv2.imshow("Sistema", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
