import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="El Takibi", page_icon="🖐️")
st.title("🖐️ El Tespiti ve Parmak Sayacı")

def position_detector(c0, c8, c9, c20):
    y0 = c0[2]
    y9 = c9[2]
    x8 = c8[1]
    x20 = c20[1]
    hand_pos = 1
    finger_pos = 1
    if y0 > y9:
        hand_pos = 0
    if x8 > x20:
        finger_pos = 0
    return hand_pos, finger_pos

def thumb_tip_pos_detector(c2, c3, c4, c5, hand_pos, finger_pos):
    cord_2 = c2[1:]
    cord_3 = c3[1:]
    cord_4 = c4[1:]
    len_4_2 = np.linalg.norm(np.array(cord_2) - np.array(cord_4))
    straight_len_4_2 = np.linalg.norm(np.array(cord_2) - np.array(cord_3)) + np.linalg.norm(np.array(cord_3) - np.array(cord_4))

    if len_4_2 < (straight_len_4_2 * 0.75):
        return 0

    if (finger_pos & hand_pos) or (finger_pos and hand_pos == 0):
        if c4[1] > c5[1]:
            return 0
    if (finger_pos == 0) and hand_pos:
        if c4[1] < c5[1]:
            return 0
    if (finger_pos == 0) and (hand_pos == 0):
        if c4[1] < c5[1]:
            return 0
    return 1

uploaded_video = st.file_uploader("Bir video yükleyin", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    tipIds = [4, 8, 12, 16, 20]
    output_frames = []

    st.info("Video işleniyor...")

    # Video işleme
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        lmList = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])

        if len(lmList) >= 21:
            fingers = []
            hand_pos, finger_pos = position_detector(lmList[0], lmList[8], lmList[9], lmList[20])
            if thumb_tip_pos_detector(lmList[2], lmList[3], lmList[4], lmList[5], hand_pos, finger_pos):
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if hand_pos:
                    fingers.append(0 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 1)
                else:
                    fingers.append(0 if lmList[tipIds[id]][2] > lmList[tipIds[id] - 2][2] else 1)

            totalF = fingers.count(1)
            cv2.putText(img, str(totalF), (30, 125), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 8)

        output_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cap.release()

    h, w, _ = output_frames[0].shape
    out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(out_temp.name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

    for frame in output_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    # İndirme butonu
    with open(out_temp.name, 'rb') as f:
        st.download_button("📥 İşlenmiş Videoyu İndir", f, file_name="el_tespiti.mp4")

    st.success("Video işleme tamamlandı.")
