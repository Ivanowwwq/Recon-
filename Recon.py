import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os
import time

# ============================
# 1. PROBAR CÁMARA
# ============================

def probar_camara():
    print("[INFO] Probando cámara...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo acceder a la cámara.")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] La cámara está ocupada o no envía imagen.")
        return False

    print("[OK] Cámara disponible.")
    return True

# ============================
# 2. PROGRAMA PRINCIPAL
# ============================

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"

AGE_LIST = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDER_LIST = ['Hombre', 'Mujer']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)


def iniciar_programa_principal():

    def set_volume(action):
        if action == "up":
            pyautogui.press("volumeup")
        elif action == "down":
            pyautogui.press("volumedown")
        elif action == "mute":
            pyautogui.press("volumemute")

    def apagar_pc():
        os.system("shutdown /s /t 5")

    def estimar_edad(frame, box):
        (x, y, w, h) = box
        face_img = frame[y:y+h, x:x+w].copy()
        if face_img.size == 0:
            return "Desconocida"
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426, 87.769, 114.896), swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        return AGE_LIST[preds[0].argmax()]

    def estimar_genero(frame, box):
        (x, y, w, h) = box
        face_img = frame[y:y+h, x:x+w].copy()
        if face_img.size == 0:
            return "Desconocido"
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426, 87.769, 114.896), swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        return GENDER_LIST[preds[0].argmax()]

    def estimar_expresion(face_landmarks):
        puntos = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        comisura_izq = np.array(puntos[61])
        comisura_der = np.array(puntos[291])
        ancho_boca = np.linalg.norm(comisura_der - comisura_izq)
        labio_sup = np.array(puntos[13])
        labio_inf = np.array(puntos[14])
        alto_boca = np.linalg.norm(labio_inf - labio_sup)
        ratio = ancho_boca / alto_boca if alto_boca != 0 else 0

        if ratio > 3.5 and alto_boca > 0.015:
            return "Feliz"
        elif alto_boca < 0.010:
            return "Serio"
        else:
            return "Neutral"

    cap = cv2.VideoCapture(0)
    mostrar_puntos = True

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands, \
         mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb)
            h, w, _ = frame.shape

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    expresion = estimar_expresion(face_landmarks)

                    if mostrar_puntos:
                        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                    xs = [l.x for l in face_landmarks.landmark]
                    ys = [l.y for l in face_landmarks.landmark]
                    box = (
                        int(min(xs) * w),
                        int(min(ys) * h),
                        int((max(xs) - min(xs)) * w),
                        int((max(ys) - min(ys)) * h),
                    )

                    edad = estimar_edad(frame, box)
                    genero = estimar_genero(frame, box)

                    cv2.putText(frame, f"Expresion: {expresion}", (30, 40), 1, 1, (0, 0, 0), 2)
                    cv2.putText(frame, f"Edad: {edad}", (30, 70), 1, 1, (0, 0, 0), 2)
                    cv2.putText(frame, f"Genero: {genero}", (30, 100), 1, 1, (0, 0, 0), 2)

            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    fingers = []

                    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
                    for tip in [8, 12, 16, 20]:
                        fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

                    if fingers == [0, 1, 0, 0, 0]:
                        set_volume("up")
                    elif fingers == [0, 0, 0, 0, 1]:
                        set_volume("down")
                    elif fingers == [0, 0, 0, 0, 0]:
                        set_volume("mute")
                    elif fingers == [0, 0, 1, 0, 0]:
                        apagar_pc()

            cv2.imshow("Sistema de reconocimiento", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('7'):
                mostrar_puntos = not mostrar_puntos

    cap.release()
    cv2.destroyAllWindows()

if probar_camara():
    iniciar_programa_principal()
else:
    print("[X] No se puede iniciar el sistema porque no hay cámara disponible.")
