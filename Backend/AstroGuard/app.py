import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import yaml
from ultralytics import YOLO
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import io

# Page configuration
st.set_page_config(
    page_title="AstroGuard - Space Station Safety Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model."""
    try:
        model_path = Path(__file__).parent / "runs" / "detect" / "train" / "weights" / "best.pt"
        if not model_path.exists():
            st.error(f"Model not found at {model_path}")
            return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_config():
    """Load the YOLO configuration file."""
    try:
        config_path = Path(__file__).parent / "models" / "yolo_params.yaml"
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None

def draw_detections(image, results, class_names):
    """Draw bounding boxes and labels on the image."""
    img_array = np.array(image)
    
    # Convert to PIL Image for drawing
    pil_image = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    detections = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # Draw bounding box
                color = (255, 0, 0) if class_name == "FireExtinguisher" else (0, 255, 0) if class_name == "ToolBox" else (0, 0, 255)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1-25), label, fill="white", font=font)
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
    
    return pil_image, detections

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 AstroGuard</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Space Station Safety Equipment Detection</h2>', unsafe_allow_html=True)
    
    # Load model and config
    model = load_model()
    config = load_config()
    
    if model is None or config is None:
        st.error("Failed to load model or configuration. Please check your setup.")
        return
    
    class_names = config.get('names', ['FireExtinguisher', 'ToolBox', 'OxygenTank'])
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        
        # Model metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("mAP@0.5", "82.2%")
        st.metric("Precision", "89.2%")
        st.metric("Recall", "74.3%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection classes
        st.markdown("## 🎯 Detection Classes")
        for i, class_name in enumerate(class_names):
            colors = ["🔴", "🟢", "🔵"]
            st.write(f"{colors[i]} {class_name}")
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Adjust the minimum confidence level for detections"
        )
        
        # Falcon information
        st.markdown("## 🦅 Falcon Model Updates")
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("""
        **Falcon** enables continuous model improvement:
        
        🔄 **Incremental Learning**: Update model with new data without full retraining
        
        📈 **Performance Tracking**: Monitor model drift and performance degradation
        
        🎯 **Active Learning**: Automatically identify images that need labeling
        
        🔧 **Model Versioning**: Track different model versions and their performance
        
        **Benefits:**
        - Reduce retraining time by 80%
        - Maintain model accuracy over time
        - Automate the ML lifecycle
        - Scale to production efficiently
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a space station image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a space station to detect safety equipment"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Convert to bytes for model input
            img_bytes = uploaded_file.getvalue()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    with col2:
        st.markdown("## 🔍 Detection Results")
        
        if uploaded_file is not None:
            with st.spinner("Running detection..."):
                # Run detection
                results = model.predict(
                    img_cv,
                    conf=confidence_threshold,
                    verbose=False
                )
                
                # Draw detections
                annotated_image, detections = draw_detections(image, results, class_names)
                
                # Display annotated image
                st.image(annotated_image, caption="Detection Results", use_container_width=True)
                
                # Display detection details
                if detections:
                    st.markdown("### 📋 Detection Details")
                    
                    # Create a DataFrame-like display
                    for i, detection in enumerate(detections):
                        col_a, col_b, col_c = st.columns([2, 2, 1])
                        with col_a:
                            st.write(f"**{detection['class']}**")
                        with col_b:
                            st.progress(detection['confidence'])
                        with col_c:
                            st.write(f"{detection['confidence']:.2f}")
                    
                    # Summary statistics
                    st.markdown("### 📊 Summary")
                    class_counts = {}
                    for detection in detections:
                        class_name = detection['class']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    for class_name, count in class_counts.items():
                        st.write(f"• {class_name}: {count} detected")
                        
                else:
                    st.info("No objects detected with the current confidence threshold.")
                    st.write("Try lowering the confidence threshold in the sidebar.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AstroGuard - Space Station Safety Detection System</p>
        <p>Powered by YOLOv8 and Ultralytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 