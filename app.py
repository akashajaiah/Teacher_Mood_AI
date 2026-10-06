import streamlit as st
import cv2
from fer.fer import FER
import numpy as np


st.set_page_config(
    page_title="Teacher Mood AI",
    page_icon="👨‍🏫",
    layout="wide"
)

# -------------------------------
# STYLE
# -------------------------------

st.markdown("""
<style>

.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 18px;
    background: #1e293b;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.result {
    padding: 30px;
    border-radius: 20px;
    background: #1e293b;
    color: white;
    text-align: center;
    margin-top: 20px;
}

.mood {
    font-size: 42px;
    font-weight: bold;
    margin: 15px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<div class="title">👨‍🏫 Teacher Mood AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Real-Time Teacher Mood Detection System'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------
# FEATURE CARDS
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
        📷<br>
        <b>Camera Detection</b><br>
        Capture teacher image
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
        🧠<br>
        <b>AI Analysis</b><br>
        Facial expression analysis
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
        📊<br>
        <b>Smart Result</b><br>
        Emotion estimation
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# -------------------------------
# AI DETECTORS
# -------------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

emotion_detector = FER(mtcnn=False)


# -------------------------------
# CAMERA INPUT
# -------------------------------

st.subheader("📷 Teacher Camera")

picture = st.camera_input(
    "Take a picture of the teacher"
)


# -------------------------------
# ANALYSIS
# -------------------------------

if picture is not None:

    # ---------------------------
    # READ IMAGE
    # ---------------------------

    image_bytes = picture.getvalue()

    image_array = cv2.imdecode(
        np.frombuffer(
            image_bytes,
            dtype="uint8"
        ),
        cv2.IMREAD_COLOR
    )


    # ---------------------------
    # FACE DETECTION
    # ---------------------------

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )


    if len(faces) > 0:

        # ---------------------------
        # DRAW FACE BOXES
        # ---------------------------

        for (x, y, w, h) in faces:

            cv2.rectangle(
                image_array,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.putText(
                image_array,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ---------------------------
        # AI EMOTION ANALYSIS
        # ---------------------------

        result_image = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2RGB
        )

        emotion_result = emotion_detector.detect_emotions(
            result_image
        )


        if emotion_result:

            emotions = emotion_result[0]["emotions"]

            detected_emotion = max(
                emotions,
                key=emotions.get
            )

            confidence = emotions[detected_emotion] * 100


            # ---------------------------
            # EMOTION NAMES
            # ---------------------------

            mood_names = {

                "happy": "😊 Happy",

                "neutral": "😐 Neutral",

                "sad": "😢 Sad",

                "angry": "😠 Angry",

                "fear": "😨 Fear",

                "disgust": "🤢 Disgust",

                "surprise": "😲 Surprise"

            }


            mood = mood_names.get(
                detected_emotion,
                detected_emotion.title()
            )


        else:

            mood = "❓ Unknown"

            confidence = 0


        # ---------------------------
        # RESULT
        # ---------------------------

        st.divider()

        st.subheader("📸 Analysis Result")


        st.image(
            result_image,
            caption="Detected Teacher Face",
            width="stretch"
        )


        st.markdown(
            f"""
            <div class="result">

            <div>
            🧠 AI MOOD ANALYSIS
            </div>

            <div class="mood">
            {mood}
            </div>

            <div>
            Confidence Score:
            <b>{confidence:.1f}%</b>
            </div>

            <br>

            <div>
            ✅ Teacher face detected successfully
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.warning(
            "⚠️ No face detected. "
            "Please take a clear photo with your face visible."
        )


# -------------------------------
# FOOTER
# -------------------------------

st.markdown(
    """
    <br>
    <div style="text-align:center;color:#94a3b8;">
    Teacher Mood AI • Dasara Mini Project
    </div>
    """,
    unsafe_allow_html=True
)