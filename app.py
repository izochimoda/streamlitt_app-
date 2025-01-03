import streamlit as st
import cv2
import numpy as np
import tempfile
import pytesseract
import re
from ultralytics import solutions


def configure_streamlit_ui():
    """Configure l'interface utilisateur de Streamlit."""
    st.set_page_config(page_title="Surveillance en temps réel des Exploitations Minières", layout="wide", page_icon="🚚")

    st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;  /* Fond clair pour la page */
        color: #333333;
    }
    .stSidebar {
        background-color: #2C3E50;  /* Un bleu foncé pour la barre latérale */
    }
    .header {
        background-color: #2980B9;  /* Un bleu vif pour l'entête */
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .header h1 {
        color: #FFFFFF;
        font-size: 48px;
        font-weight: bold;
    }
    .header h2 {
        color: #FFFFFF;
        font-size: 28px;
        font-weight: 300;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin: 10px;
    }
    .metric-card .value {
        font-size: 30px;
        color: #2ECC71;  /* Vert pour les valeurs positives */
        font-weight: bold;
    }
    .metric-card .label {
        font-size: 18px;
        color: #7F8C8D;  /* Gris pour les labels */
    }
    .metric-card.green { border-left: 5px solid #27AE60; }  /* Vert plus clair pour les camions IN */
    .metric-card.blue { border-left: 5px solid #3498DB; }   /* Bleu plus clair pour le volume par camion */
    .metric-card.orange { border-left: 5px solid #E67E22; } /* Orange pour le volume total extrait */
    .metric-card.red { border-left: 5px solid #E74C3C; }    /* Rouge pour les camions OUT */

    /* Ajouter la règle pour le titre ROI */
    .roi-title {
        color: #000000;  /* Noir */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="header"><h1>Surveillance en temps réel des Exploitations Minières</h1><h2>Détection Automatisée des Camions grâce à l\'Intelligence Artificielle et l\'Analyse SIG</h2></div>',
        unsafe_allow_html=True)


def display_dashboard(detection_counts, volume_per_truck, in_count, out_count, total_trucks_detected, truck_type,
                      substance_type):
    """Affiche un tableau de bord amélioré avec des cartes et des valeurs attrayantes."""
    total_trucks = sum(detection_counts)
    total_volume = total_trucks_detected * volume_per_truck

    st.markdown("### Tableau de bord")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="label">Nombre total de camions détectés</div>
            <div class="value">{total_trucks_detected}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="label">Volume par camion (en m³)</div>
            <div class="value">{volume_per_truck}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="label">Volume total extrait (en m³)</div>
            <div class="value">{total_volume}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card red">
            <div class="label">Camions OUT</div>
            <div class="value">{out_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="label">Camions IN</div>
            <div class="value">{in_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="label">Volume en fonction du types de camions(20 m³, 25 m³ , 30 m³)</div>
            <div class="value">{truck_type}</div>
        </div>
        """, unsafe_allow_html=True)

    with col7:
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="label">Substance transportée</div>
            <div class="value">{substance_type}</div>
        </div>
        """, unsafe_allow_html=True)


def configure_roi_type(roi_option, width, height):
    """Configure les points de la ROI en fonction du type choisi."""
    if roi_option == "Barre verticale":
        x_pos = st.slider("Position de la barre verticale (X)", 0, width, width // 2)
        region_points = [(x_pos, 0), (x_pos, height)]
    elif roi_option == "Barre horizontale":
        y_pos = st.slider("Position de la barre horizontale (Y)", 0, height, height // 2)
        region_points = [(0, y_pos), (width, y_pos)]
    elif roi_option == "Toute la surface":
        left = st.slider("Limite gauche", 0, width, 0)
        top = st.slider("Limite haute", 0, height, 0)
        right = st.slider("Limite droite", 0, width, width)
        bottom = st.slider("Limite basse", 0, height, height)
        region_points = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
            (left, top)
        ]
    else:
        region_points = []

    return region_points


def draw_roi_on_frame(frame, region_points, roi_option):
    """Dessine la ROI sur une frame selon le type choisi."""
    roi_color = (0, 255, 0)  # Vert pour la ROI
    thickness = 2  # Épaisseur du tracé

    if roi_option in ["Barre verticale", "Barre horizontale"]:
        cv2.line(frame, region_points[0], region_points[1], roi_color, thickness)
    elif roi_option == "Toute la surface":
        for i in range(len(region_points) - 1):
            cv2.line(frame, region_points[i], region_points[i + 1], roi_color, thickness)

    return frame


def is_in_roi(x, y, region):
    """Vérifie si un point est dans la ROI."""
    return region[0][0] <= x <= region[1][0] and region[0][1] <= y <= region[2][1]


def extract_in_out_info(text):
    """Extrait les informations 'IN' et 'OUT' du texte et évite les doublons."""
    in_count = len(re.findall(r'\bIN\b', text))
    out_count = len(re.findall(r'\bOUT\b', text))

    return in_count, out_count


def display_video_with_roi_and_detection(video_path, region_points, roi_option, model_path):
    """Affiche la vidéo avec la ROI tracée et les détections YOLO."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Erreur lors de l'ouverture de la vidéo."

    temp_output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    counter = solutions.ObjectCounter(
        show=False,
        region=region_points,
        model=model_path
    )

    detection_counts = []
    in_count = 0
    out_count = 0
    total_trucks_detected = 0

    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = counter.count(frame)
        truck_count = 0

        if isinstance(results, list) and results:
            detections = results[0].get('detections', [])
            for detection in detections:
                x1, y1, x2, y2 = detection[:4]
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                if is_in_roi(x_center, y_center, region_points):
                    truck_count += 1

        detection_counts.append(truck_count)
        frame = draw_roi_on_frame(frame, region_points, roi_option)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        stframe.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    out.release()

    return detection_counts, in_count, out_count, total_trucks_detected


def main():
    configure_streamlit_ui()

    st.markdown("### Charger une vidéo et paramétrer le modèle")
    uploaded_video = st.file_uploader("Uploader une vidéo", type=["mp4", "avi", "mov"])

    st.markdown("### Charger le modèle YOLO")
    model_path = st.text_input("Chemin du modèle YOLO", value="C:/Users/user/Documents/yolov5-5.0/best.pt")

    st.markdown("### Zone d'intérêt (ROI) : Choisissez le type")
    roi_option = st.selectbox(
        "Sélectionnez le type de ROI",
        ["Barre verticale", "Barre horizontale", "Toute la surface"]
    )
    st.markdown("### Volume des camions (en m³)")
    volume_per_truck = st.number_input("Volume des camions (en m³)", value=20, min_value=1)

    # Ajout des champs pour le type de camion et la substance transportée
    st.markdown("#### Volume en fonction du types de camions(20 m³, 25 m³ , 30 m³)")
    truck_type = st.text_input("Volume en fonction du types de camions(20 m³, 25 m³ , 30 m³)", "20 m³ ")
    st.markdown("### Substance transportée")
    substance_type = st.text_input("Substance transportée", "Sable")

    # Tableau de bord permanent
    detection_counts = []
    in_count = 0
    out_count = 0
    total_trucks_detected = 0

    # Mise à jour du tableau de bord avec les nouveaux champs
    display_dashboard(detection_counts, volume_per_truck, in_count, out_count, total_trucks_detected, truck_type, substance_type)

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            temp_video_path = temp_file.name

        cap = cv2.VideoCapture(temp_video_path)
        assert cap.isOpened(), "Erreur lors de l'ouverture de la vidéo."
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        region_points = configure_roi_type(roi_option, width, height)

        st.markdown("### Visualisation de la vidéo avec ROI et détections")
        detection_counts, temp_output_path, in_count, out_count, total_trucks_detected = display_video_with_roi_and_detection(
            temp_video_path, region_points, roi_option, model_path
        )

        display_dashboard(detection_counts, volume_per_truck, in_count, out_count, total_trucks_detected, truck_type, substance_type)

        st.markdown("### Télécharger la vidéo traitée")
        with open(temp_output_path, "rb") as video_file:
            st.download_button(
                label="Télécharger la vidéo traitée",
                data=video_file,
                file_name="video_detected_with_annotations.mp4",
                mime="video/mp4"
            )

if __name__ == "__main__":
    main()

