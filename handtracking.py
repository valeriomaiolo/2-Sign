import cv2
import mediapipe as mp
from pythonosc import udp_client

ip = "localhost"       # localhost (se ricevi sullo stesso computer)
port = 6448            # porta su cui il ricevitore ascolta

client = udp_client.SimpleUDPClient(ip, port)

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands: # context manager  

    while True:
        ret, frame = cap.read() #return succecc e frame come array numpy
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # l'array di numpy di nome image non è più scrivibile

        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Preparo la lista piatta
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])

                # Invio la lista come unico messaggio OSC
                client.send_message("/wek/inputs", landmark_list)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27: # Se l'utente preme ESC (ASCII 27) allora esci dal ciclo
            break

cap.release()
cv2.destroyAllWindows()
