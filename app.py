import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# =========================
# CẤU HÌNH APP
# =========================
st.set_page_config(
    page_title="FlorAD",
    page_icon="🌸",
    layout="wide"
)

# =========================
# SESSION STATE CHUYỂN TRANG
# =========================
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# =========================
# KÍCH THƯỚC ẢNH GIỐNG LÚC TRAIN
# =========================
img_width, img_height = 224, 224

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_flower_model():
    model = tf.keras.models.load_model("flowers.h5")
    return model

model = load_flower_model()

# =========================
# TÊN CÁC LOẠI HOA
# PHẢI ĐÚNG THỨ TỰ CLASS LÚC TRAIN
# =========================
class_names = [
    "Hoa cúc",
    "Bồ công anh",
    "Hoa hồng",
    "Hoa hướng dương",
    "Hoa tulip"
]

flower_icons = {
    "Hoa cúc": "🌼",
    "Bồ công anh": "🌾",
    "Hoa hồng": "🌹",
    "Hoa hướng dương": "🌻",
    "Hoa tulip": "🌷"
}

# =========================
# CSS GIAO DIỆN
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #FFF1F7 0%, #FFFBEA 45%, #F1FFF8 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }

    .hero {
        background: linear-gradient(135deg, #FF4F9A, #FFB84D);
        padding: 38px 35px;
        border-radius: 28px;
        color: white;
        box-shadow: 0px 14px 35px rgba(255, 79, 154, 0.25);
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 8px;
        letter-spacing: 1px;
        color: white !important;
    }

    .hero-subtitle {
        font-size: 19px;
        line-height: 1.6;
        opacity: 0.96;
        max-width: 850px;
        color: white !important;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 24px;
        padding: 25px;
        box-shadow: 0px 10px 28px rgba(120, 80, 120, 0.12);
        backdrop-filter: blur(10px);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: #C2185B;
        margin-bottom: 12px;
    }

    .small-text {
        color: #333333;
        font-size: 16px;
        line-height: 1.6;
    }

    .result-card {
        background: linear-gradient(135deg, #ffffff, #FFF0F6);
        border: 2px solid #FF8FBD;
        border-radius: 28px;
        padding: 28px;
        text-align: center;
        box-shadow: 0px 14px 30px rgba(233, 30, 99, 0.16);
        margin-top: 10px;
    }

    .result-icon {
        font-size: 72px;
        margin-bottom: 8px;
    }

    .result-label {
        font-size: 18px;
        color: #333333;
        font-weight: 600;
    }

    .result-name {
        font-size: 36px;
        color: #C2185B;
        font-weight: 900;
        margin-top: 5px;
    }

    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #FF4F9A, #FFB84D);
        color: white !important;
        padding: 10px 22px;
        border-radius: 999px;
        font-size: 19px;
        font-weight: 800;
        margin-top: 18px;
    }

    .flower-chip {
        display: inline-block;
        background: white;
        padding: 10px 15px;
        border-radius: 999px;
        margin: 5px;
        color: #C2185B !important;
        font-weight: 700;
        box-shadow: 0px 5px 14px rgba(0,0,0,0.08);
    }

    .footer {
        text-align: center;
        color: #333333;
        font-size: 15px;
        margin-top: 35px;
        padding-bottom: 20px;
    }

    .prob-text {
        color: #111111 !important;
        font-size: 17px;
        font-weight: 800;
        margin-top: 13px;
        margin-bottom: 5px;
    }

    .welcome-flower-card {
        background: white;
        border-radius: 22px;
        padding: 22px 10px;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,143,189,0.4);
        margin-bottom: 12px;
    }

    .welcome-flower-icon {
        font-size: 48px;
    }

    .welcome-flower-name {
        color: #C2185B !important;
        font-size: 18px;
        font-weight: 900;
        margin-top: 8px;
    }

    .feature-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 20px;
        min-height: 85px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,143,189,0.4);
    }

    .feature-icon {
        font-size: 30px;
        min-width: 38px;
        text-align: center;
    }

    .feature-text {
        color: #111111 !important;
        font-size: 17px;
        font-weight: 800;
        line-height: 1.4;
    }

    .custom-alert-info {
        background-color: #D9F0FF;
        color: #111111;
        padding: 16px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 650;
        margin-top: 15px;
        border-left: 6px solid #2196F3;
    }

    .custom-alert-warning {
        background-color: #FFF3CD;
        color: #111111;
        padding: 16px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 650;
        margin-top: 15px;
        border-left: 6px solid #FFB300;
    }

    .custom-alert-success {
        background-color: #DFF7E8;
        color: #111111;
        padding: 16px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 650;
        margin-top: 15px;
        border-left: 6px solid #2E7D32;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #FF4F9A, #FFB84D);
        color: white !important;
        border: none;
        border-radius: 999px;
        padding: 13px 32px;
        font-size: 19px;
        font-weight: 900;
        box-shadow: 0px 10px 25px rgba(255, 79, 154, 0.28);
        transition: 0.2s;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        color: white !important;
        box-shadow: 0px 12px 30px rgba(255, 79, 154, 0.38);
    }

    /* Upload box */
    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        padding: 24px !important;
        border-radius: 26px !important;
        border: 2px dashed #FF8FBD !important;
        box-shadow: 0px 10px 25px rgba(255, 79, 154, 0.12) !important;
    }

    div[data-testid="stFileUploaderDropzone"] {
        background: #2b2d3a !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 18px !important;
    }

    div[data-testid="stFileUploaderDropzone"] > div {
        background: #2b2d3a !important;
    }

    div[data-testid="stFileUploaderDropzone"] section {
        background: #2b2d3a !important;
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button[kind],
    div[data-testid="stFileUploaderDropzone"] button[data-testid],
    div[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"],
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"] {
        background: #1f2230 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 12px !important;
        font-weight: 900 !important;
        box-shadow: none !important;
        padding: 10px 22px !important;
    }

    div[data-testid="stFileUploaderDropzone"] button:hover,
    div[data-testid="stFileUploaderDropzone"] button[kind]:hover,
    div[data-testid="stFileUploaderDropzone"] button[data-testid]:hover,
    div[data-testid="stFileUploaderDropzone"] button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stFileUploaderDropzone"] [data-testid="baseButton-secondary"]:hover {
        background: #34384a !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.55) !important;
        transform: scale(1.02) !important;
        box-shadow: none !important;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FF4F9A, #FFB84D);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #111111 !important;
    }

    p, span, label {
        color: inherit;
    }

    div[data-testid="stFileUploaderDropzone"] p,
    div[data-testid="stFileUploaderDropzone"] span,
    div[data-testid="stFileUploaderDropzone"] small,
    div[data-testid="stFileUploaderDropzone"] div,
    div[data-testid="stFileUploaderDropzone"] section {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] button,
    div[data-testid="stFileUploaderDropzone"] button *,
    div[data-testid="stFileUploaderDropzone"] button span,
    div[data-testid="stFileUploaderDropzone"] button p {
        color: #ffffff !important;
    }

    div[data-testid="stFileUploaderDropzone"] button svg,
    div[data-testid="stFileUploaderDropzone"] button svg *,
    div[data-testid="stFileUploaderDropzone"] button path {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# TRANG CHÀO MỪNG
# =========================
if st.session_state.page == "welcome":

    st.markdown("# 🌸 Welcome to FlorAD")
    st.markdown("### Ứng dụng nhận diện hoa bằng trí tuệ nhân tạo")
    st.write("FlorAD giúp nhận diện nhanh 5 loại hoa phổ biến chỉ bằng một hình ảnh.")

    st.markdown("---")

    flower_intro = [
        ("🌼", "Hoa cúc"),
        ("🌾", "Bồ công anh"),
        ("🌹", "Hoa hồng"),
        ("🌻", "Hướng dương"),
        ("🌷", "Tulip")
    ]

    intro_cols = st.columns(5)

    for col, (icon, name) in zip(intro_cols, flower_intro):
        with col:
            st.markdown(
                f"""
                <div class="welcome-flower-card">
                    <div class="welcome-flower-icon">{icon}</div>
                    <div class="welcome-flower-name">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.markdown("### ✨ Tính năng nổi bật")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📤</div>
                <div class="feature-text">Upload ảnh hoa trực tiếp từ máy tính</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📸</div>
                <div class="feature-text">Chụp ảnh hoa trực tiếp bằng camera</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-text">Hiển thị độ tin cậy và xác suất từng loại hoa</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col_a, col_b, col_c = st.columns([1.5, 1, 1.5])

    with col_b:
        if st.button("🚀 Bắt đầu nhận diện", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.markdown(
        """
        <div class="footer">
            Made with 🌸 by AD | FlorAD App
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# TRANG CHÍNH
# =========================
else:

    # =========================
    # NÚT QUAY LẠI TRANG ĐẦU
    # =========================
    top_col1, top_col2 = st.columns([1, 5])

    with top_col1:
        if st.button("🏠 Trang đầu", use_container_width=True):
            st.session_state.page = "welcome"
            st.rerun()

    # =========================
    # HERO
    # =========================
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🌸 FlorAD</div>
            <div class="hero-subtitle">
                Ứng dụng trí tuệ nhân tạo giúp nhận diện 5 loại hoa phổ biến:
                hoa cúc, bồ công anh, hoa hồng, hoa hướng dương và hoa tulip.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # DANH SÁCH HOA
    # =========================
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 20px;">
            <span class="flower-chip">🌼 Hoa cúc</span>
            <span class="flower-chip">🌾 Bồ công anh</span>
            <span class="flower-chip">🌹 Hoa hồng</span>
            <span class="flower-chip">🌻 Hoa hướng dương</span>
            <span class="flower-chip">🌷 Hoa tulip</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Biến ảnh dùng chung cho upload và camera
    img = None

    # =========================
    # LAYOUT CHÍNH
    # =========================
    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">📤 Đưa ảnh hoa vào hệ thống</div>
                <div class="small-text">
                    Bạn có thể tải ảnh hoa từ máy tính hoặc chụp ảnh trực tiếp bằng camera.
                    Hệ thống sẽ tự động resize ảnh về kích thước 224 x 224 giống lúc train model.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        input_method = st.radio(
            "Chọn cách đưa ảnh vào hệ thống:",
            ["📤 Tải ảnh từ máy", "📸 Chụp ảnh trực tiếp"],
            horizontal=True
        )

        if input_method == "📤 Tải ảnh từ máy":
            uploaded_file = st.file_uploader(
                "Chọn ảnh hoa",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")

                st.markdown("### 🖼️ Ảnh đã tải lên")
                st.image(
                    img,
                    caption="Ảnh đầu vào",
                    use_container_width=True
                )

        else:
            camera_file = st.camera_input("📸 Chụp ảnh hoa trực tiếp")

            if camera_file is not None:
                img = Image.open(camera_file).convert("RGB")

                st.markdown("### 📸 Ảnh vừa chụp")
                st.image(
                    img,
                    caption="Ảnh chụp từ camera",
                    use_container_width=True
                )

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">🤖 Kết quả AI</div>
                <div class="small-text">
                    Sau khi có ảnh, model CNN sẽ phân tích và đưa ra loại hoa có xác suất cao nhất.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if img is not None:
            # =========================
            # TIỀN XỬ LÝ ẢNH
            # =========================
            img_resized = img.resize((img_width, img_height))
            img_array = image.img_to_array(img_resized)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # =========================
            # DỰ ĐOÁN
            # =========================
            predictions = model.predict(img_array)

            predicted_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = class_names[predicted_index]
            icon = flower_icons[predicted_class]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-icon">{icon}</div>
                    <div class="result-label">Loài hoa được dự đoán là</div>
                    <div class="result-name">{predicted_class}</div>
                    <div class="confidence-badge">Độ tin cậy: {confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📊 Xác suất từng loại hoa")

            for i, prob in enumerate(predictions[0]):
                percent = prob * 100
                flower_name = class_names[i]
                flower_icon = flower_icons[flower_name]

                st.markdown(
                    f"""
                    <div class="prob-text">
                        {flower_icon} {flower_name}: {percent:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(float(prob))

            if confidence < 60:
                st.markdown(
                    """
                    <div class="custom-alert-warning">
                        ⚠️ Độ tin cậy chưa cao. Bạn nên thử ảnh rõ hơn, đủ sáng hơn hoặc hoa nằm ở trung tâm ảnh.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif confidence < 80:
                st.markdown(
                    """
                    <div class="custom-alert-info">
                        ℹ️ Kết quả tương đối ổn, nhưng bạn vẫn nên kiểm tra thêm với ảnh khác.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="custom-alert-success">
                        ✅ Kết quả có độ tin cậy cao.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.markdown(
                """
                <div class="result-card">
                    <div class="result-icon">🌷</div>
                    <div class="result-label">Chưa có ảnh nào được đưa vào</div>
                    <div class="result-name" style="font-size:28px;">Hãy tải ảnh hoặc chụp ảnh</div>
                    <div class="confidence-badge">Đang chờ dự đoán...</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========================
    # THÔNG TIN THÊM
    # =========================
    st.markdown("---")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">🧠 Model CNN</div>
                <div class="small-text">
                    Ứng dụng sử dụng mô hình CNN đã train với ảnh kích thước 224 x 224.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_col2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">⚡ Xử lý nhanh</div>
                <div class="small-text">
                    Ảnh được chuẩn hóa về khoảng 0 - 1 trước khi đưa vào model dự đoán.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with info_col3:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">🌺 5 loại hoa</div>
                <div class="small-text">
                    Hệ thống hỗ trợ nhận diện: hoa cúc, bồ công anh, hoa hồng, hướng dương và tulip.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer">
            Made with 🌸 by AD | FlorAD App
        </div>
        """,
        unsafe_allow_html=True
    )