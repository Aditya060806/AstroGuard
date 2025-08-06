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
import base64
import matplotlib.pyplot as plt
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AstroGuard - Space Station Safety Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject animated space CSS and interactive effects
st.markdown("""
<style>
body, .stApp {
    background: #121629 !important;
    min-height: 100vh;
    font-family: 'Orbitron', 'Segoe UI', sans-serif;
    overflow-x: hidden;
}
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

/* Animated stars background using CSS keyframes */
.stApp:before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -2;
    pointer-events: none;
    background: transparent;
}
.stApp:after {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -1;
    pointer-events: none;
    background: url("https://raw.githubusercontent.com/rajatdiptabiswas/Animated-Star-Background/main/star-bg.png");
    opacity: 0.22;
    animation: starMove 60s linear infinite;
}
@keyframes starMove {
    0% {background-position: 0 0;}
    100% {background-position: 1000px 1000px;}
}

/* Card-style panel */
.block-container {
    background: rgba(30, 34, 54, 0.92);
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 #23294688;
    padding: 2rem 2.5vw;
    margin: 2rem auto;
    max-width: 900px;
    transition: box-shadow 0.4s, transform 0.4s;
}
.block-container:hover {
    box-shadow: 0 12px 40px 0 #ffb74d88, 0 0 32px #6a89cc88;
    transform: scale(1.01);
}

/* Header and sub-header */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #e0e6f7;
    text-shadow: 0 0 12px #6a89cc, 0 0 32px #232946;
    text-align: center;
    margin-bottom: 1.5rem;
    font-family: 'Orbitron', 'Segoe UI', sans-serif;
    letter-spacing: 2px;
    transition: color 0.3s, text-shadow 0.3s;
}
.main-header:hover {
    color: #ffb74d;
    text-shadow: 0 0 24px #ffb74d, 0 0 48px #232946;
}
.sub-header {
    font-size: 1.5rem;
    color: #ffb74d;
    margin-bottom: 1rem;
    text-align: center;
    text-shadow: 0 0 8px #ffb74d44;
    font-family: 'Orbitron', 'Segoe UI', sans-serif;
    transition: color 0.3s, text-shadow 0.3s;
}
.sub-header:hover {
    color: #90caf9;
    text-shadow: 0 0 16px #90caf9;
}

/* Card for detection results */
.card {
    background: linear-gradient(135deg, #232946 60%, #6a89cc 100%);
    border-radius: 14px;
    box-shadow: 0 4px 24px #23294688;
    padding: 1.5rem;
    margin-bottom: 2rem;
    transition: box-shadow 0.3s, transform 0.3s;
}
.card:hover {
    box-shadow: 0 8px 32px #ffb74d88, 0 0 24px #6a89cc88;
    transform: scale(1.03);
}

/* Detection details */
.det-detail {
    background: rgba(36, 41, 70, 0.85);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px #6a89cc44;
    transition: box-shadow 0.3s, background 0.3s, transform 0.3s;
}
.det-detail:hover {
    background: #232946;
    box-shadow: 0 4px 16px #ffb74d88, 0 0 24px #90caf988;
    transform: scale(1.04) rotate(-1deg);
}

/* Buttons and inputs */
.stButton > button, .stDownloadButton > button {
    position: relative;
    overflow: hidden;
    background: linear-gradient(90deg, #6a89cc, #232946);
    color: #fff;
    border-radius: 10px;
    border: none;
    box-shadow: 0 0 12px #6a89cc88;
    font-size: 1.1rem;
    padding: 0.5rem 1.2rem;
    transition: box-shadow 0.2s, transform 0.2s, background 0.2s;
}
.stButton > button::after, .stDownloadButton > button::after {
    content: "";
    position: absolute;
    left: 50%; top: 50%;
    width: 0; height: 0;
    background: rgba(255,183,77,0.25);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.4s, height 0.4s;
    z-index: 0;
}
.stButton > button:active::after, .stDownloadButton > button:active::after {
    width: 120px; height: 120px;
    transition: width 0.2s, height 0.2s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 0 24px #ffb74d, 0 0 32px #6a89cc;
    transform: scale(1.09);
    background: linear-gradient(90deg, #ffb74d, #232946);
    color: #232946;
}

/* Inputs */
.stTextInput > div > input, .stSlider > div {
    border-radius: 8px !important;
    box-shadow: 0 0 8px #6a89cc44;
    transition: box-shadow 0.2s, transform 0.2s;
}
.stTextInput > div > input:focus, .stSlider > div:focus-within {
    box-shadow: 0 0 16px #ffb74d88;
    transform: scale(1.03);
}

/* Expander and sidebar info */
.stExpanderHeader, .sidebar-info {
    transition: box-shadow 0.3s, background 0.3s, transform 0.3s;
}
.stExpanderHeader:hover, .sidebar-info:hover {
    box-shadow: 0 4px 16px #ffb74d88;
    background: #232946;
    transform: scale(1.02);
}

/* Responsive layout */
@media (max-width: 900px) {
    .block-container { max-width: 98vw; padding: 1rem 1vw; }
    .main-header { font-size: 2.2rem; }
    .sub-header { font-size: 1.1rem; }
    .card { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# Inject interactive particle background (spider-web style, space theme)
st.markdown("""
<style>
#particle-bg {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  z-index: -10;
  pointer-events: none;
}
</style>
<canvas id="particle-bg"></canvas>
<script>
const canvas = document.getElementById('particle-bg');
const ctx = canvas.getContext('2d');
let w = window.innerWidth, h = window.innerHeight;
canvas.width = w; canvas.height = h;

let mouse = { x: w/2, y: h/2 };
document.onmousemove = (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
};

window.onresize = () => {
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = w; canvas.height = h;
};

const PARTICLE_COUNT = 80;
const particles = [];
for(let i=0;i<PARTICLE_COUNT;i++){
    particles.push({
        x: Math.random()*w,
        y: Math.random()*h,
        vx: (Math.random()-0.5)*1.2,
        vy: (Math.random()-0.5)*1.2,
        r: Math.random()*2+1.5,
        glow: Math.random()*12+8
    });
}

function draw(){
    ctx.clearRect(0,0,w,h);
    // Move and draw particles
    for(let i=0;i<PARTICLE_COUNT;i++){
        let p = particles[i];
        // Mouse repulsion
        let dx = p.x - mouse.x, dy = p.y - mouse.y;
        let dist = Math.sqrt(dx*dx + dy*dy);
        if(dist < 120){
            p.vx += dx/dist*0.08;
            p.vy += dy/dist*0.08;
        }
        // Move
        p.x += p.vx;
        p.y += p.vy;
        // Slow down
        p.vx *= 0.98;
        p.vy *= 0.98;
        // Bounce
        if(p.x < 0 || p.x > w) p.vx *= -1;
        if(p.y < 0 || p.y > h) p.vy *= -1;
        // Draw particle
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, 2*Math.PI);
        ctx.shadowColor = "#90caf9";
        ctx.shadowBlur = p.glow;
        ctx.fillStyle = "#e0e6f7";
        ctx.globalAlpha = 0.85;
        ctx.fill();
        ctx.restore();
    }
    // Draw lines
    for(let i=0;i<PARTICLE_COUNT;i++){
        for(let j=i+1;j<PARTICLE_COUNT;j++){
            let p1 = particles[i], p2 = particles[j];
            let dx = p1.x-p2.x, dy = p1.y-p2.y;
            let dist = Math.sqrt(dx*dx + dy*dy);
            if(dist < 120){
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.strokeStyle = "rgba(144,202,249,"+(1-dist/120)*0.35+")";
                ctx.lineWidth = 1.2-(dist/120);
                ctx.shadowColor = "#90caf9";
                ctx.shadowBlur = 8;
                ctx.stroke();
                ctx.restore();
            }
        }
    }
    requestAnimationFrame(draw);
}
draw();
</script>
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

def get_image_download_link(img, filename="annotated.png"):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download Annotated Image</a>'
    return href

def company_details_section():
    st.markdown("---")
    st.markdown('<h2 class="main-header">🛰 About Falcon by Duality AI</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(36,41,70,0.85); border-radius:14px; box-shadow:0 4px 24px #23294688; padding:2rem; margin-bottom:2rem;">
    <h3 style="color:#90caf9;">What is Falcon?</h3>
    <p>
    Falcon is a synthetic data platform developed by <b>Duality AI</b>, designed to simulate real-world environments at photorealistic, physics-based levels — and generate high-quality labeled data for training AI and computer vision models.
    </p>
    <ul>
      <li>🔬 <b>Solves the “data bottleneck”</b> in AI by generating massive, diverse, and perfectly labeled datasets for object detection, segmentation, depth estimation, and more — without needing real-world data collection.</li>
    </ul>
    <h3 style="color:#ffb74d;">Who is Behind Duality AI?</h3>
    <p>
    Duality AI is a US-based company focused on accelerating AI development through simulation. The team combines experience from:
    <ul>
      <li>Nvidia (Omniverse, PhysX)</li>
      <li>DARPA and defense simulation research</li>
      <li>Game engines like Unreal Engine</li>
      <li>Robotics and autonomous systems</li>
    </ul>
    </p>
    <h3 style="color:#90caf9;">✅ Real-World Problems Falcon Solves</h3>
    <table style="width:100%; color:#e0e6f7;">
      <tr><td>📉 <b>Lack of diverse training data</b></td><td>Falcon generates varied datasets for rare events, edge cases, or dangerous environments</td></tr>
      <tr><td>🧼 <b>Labeling errors in real data</b></td><td>Falcon provides 100% accurate, pixel-perfect annotations</td></tr>
      <tr><td>🕵 <b>Privacy or regulatory restrictions</b></td><td>No real-world cameras needed, bypassing privacy/legal barriers</td></tr>
      <tr><td>💸 <b>High cost of real data collection</b></td><td>Synthetic scenes are reusable and scalable at near-zero cost</td></tr>
      <tr><td>🧪 <b>Model failure in rare conditions</b></td><td>Simulate edge cases (fog, night, clutter, damage) for robust models</td></tr>
    </table>
    <h3 style="color:#ffb74d;">🧰 Falcon's Capabilities</h3>
    <ul>
      <li>✅ RGB images</li>
      <li>✅ Bounding boxes (YOLO, COCO, Pascal VOC)</li>
      <li>✅ Semantic segmentation maps</li>
      <li>✅ Depth maps</li>
      <li>✅ Instance segmentation</li>
      <li>✅ Optical flow</li>
      <li>✅ Stereo pairs (3D vision)</li>
    </ul>
    <h3 style="color:#90caf9;">🌐 Technologies Behind Falcon</h3>
    <ul>
      <li>Unreal Engine 5 for real-time rendering</li>
      <li>Physics-based simulations for realism</li>
      <li>Programmatic scene generation via API/SDK</li>
      <li>Cloud or local execution</li>
      <li>Export-ready formats: COCO, YOLO, custom JSON, etc.</li>
    </ul>
    <h3 style="color:#ffb74d;">🧪 Sample Use Cases</h3>
    <table style="width:100%; color:#e0e6f7;">
      <tr><td>🚗 Autonomous Vehicles</td><td>Simulate traffic, night driving, pedestrians, sensor occlusion</td></tr>
      <tr><td>🏭 Robotics</td><td>Train robots to detect tools, navigate rooms, identify objects on conveyor belts</td></tr>
      <tr><td>🛰 Space & Defense</td><td>Train models to recognize spacecraft parts, astronaut tools, satellite damage</td></tr>
      <tr><td>🏥 Healthcare</td><td>Simulate body parts or surgical scenes for vision-based AI</td></tr>
      <tr><td>🧯 Disaster Response</td><td>Simulate smoke, fire, debris, rubble for drones or emergency AI agents</td></tr>
    </table>
    <h3 style="color:#90caf9;">🧑‍💻 How Falcon Works for Developers</h3>
    <ol>
      <li>Define a Scene: Pick environment (e.g., space station, warehouse)</li>
      <li>Add Objects: Choose what to render and detect (e.g., tools, bots)</li>
      <li>Configure Camera & Motion: Set FOV, path, jitter, lighting, etc.</li>
      <li>Set Output Requirements: Bounding boxes, depth, etc.</li>
      <li>Launch Generation: Via web interface, API, or Python SDK</li>
      <li>Download Labeled Dataset: Automatically formatted for training</li>
    </ol>
    <h3 style="color:#ffb74d;">🛠 Falcon API Capabilities</h3>
    <ul>
      <li>Authenticate and manage projects</li>
      <li>Configure batch synthetic data jobs</li>
      <li>Query generated dataset metadata</li>
      <li>Integrate Falcon into pipelines with YOLOv8, Detectron2, etc.</li>
    </ul>
    <h3 style="color:#90caf9;">📈 Impact in Real AI Workflows</h3>
    <ul>
      <li>🧠 <b>YOLOv8 + Falcon:</b> Falcon’s data improves detection performance especially on hard edge cases like low lighting or occlusion.</li>
      <li>🧪 <b>mAP boost:</b> Many users report 10–25% higher mAP on rare events after training on Falcon-generated data.</li>
    </ul>
    <h3 style="color:#ffb74d;">🔐 Access & Pricing</h3>
    <ul>
      <li>Falcon currently operates in beta or early-access mode</li>
      <li>Used in Duality AI Hackathons and by enterprises or defense</li>
      <li>To request access: <a href="https://falcon.duality.ai" target="_blank" style="color:#90caf9;">falcon.duality.ai</a></li>
    </ul>
    <h3 style="color:#90caf9;">🧾 Example: Real Use Case (NASA x Duality AI)</h3>
    <ul>
      <li><b>Problem:</b> Detect space tool damage during EVA (spacewalk)</li>
      <li><b>Solution:</b> Falcon simulated 1000s of scenarios with lighting variation, motion blur, and tool types</li>
      <li><b>Outcome:</b> Trained a YOLOv8 model with Falcon data → achieved &gt;90% accuracy on real EVA video</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

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
    
    # Add session state for detection history
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <style>
        .sidebar-metric-card {
            background: linear-gradient(135deg, #232946 60%, #6a89cc 100%);
            border-radius: 16px;
            box-shadow: 0 4px 24px #23294688;
            padding: 1.2rem 1rem 1rem 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .sidebar-metric-row {
            display: flex;
            justify-content: center;
            align-items: flex-end;
            gap: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .sidebar-metric {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .sidebar-metric-value {
            font-size: 2.2rem;
            font-weight: bold;
            margin-bottom: 0.1rem;
            line-height: 1;
        }
        .sidebar-metric-label {
            color: #e0e6f7;
            font-weight: bold;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }
        .sidebar-metric-map { color: #ffb74d; text-shadow: 0 0 8px #ffb74d88; }
        .sidebar-metric-prec { color: #90caf9; text-shadow: 0 0 8px #90caf988; }
        .sidebar-metric-recall { color: #66bb6a; text-shadow: 0 0 8px #66bb6a88; }
        .sidebar-metric-footer {
            margin-top: 0.7rem;
            font-size: 1rem;
            color: #e0e6f7;
        }
        .sidebar-metric-footer span { color: #ffb74d; }
        .sidebar-classes-title {
            font-size: 1.15rem;
            color: #90caf9;
            font-weight: bold;
            margin-bottom: 0.5rem;
            margin-top: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }
        .sidebar-class-chip {
            background: rgba(36,41,70,0.85);
            border-radius: 8px;
            box-shadow: 0 2px 8px #23294644;
            padding: 0.4rem 1rem;
            color: #fff;
            font-weight: bold;
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
            margin-right: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: box-shadow 0.3s, transform 0.3s;
            border: 2px solid transparent;
        }
        .sidebar-class-chip.red { color: #ff1744; border-color: #ff1744; }
        .sidebar-class-chip.green { color: #00e676; border-color: #00e676; }
        .sidebar-class-chip.blue { color: #2979ff; border-color: #2979ff; }
        .sidebar-class-chip.yellow { color: #ffd600; border-color: #ffd600; }
        .sidebar-class-chip.orange { color: #ff6d00; border-color: #ff6d00; }
        .sidebar-class-chip:hover {
            box-shadow: 0 4px 16px #ffb74d88;
            transform: scale(1.08);
            background: #232946;
        }
        .sidebar-threshold-title {
            font-size: 1.1rem;
            color: #ffb74d;
            font-weight: bold;
            margin-bottom: 0.5rem;
            margin-top: 1.2rem;
            text-shadow: 0 0 8px #ffb74d88;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## <span style='display:flex;align-items:center;gap:0.5rem;'>📊 Model Information</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-metric-card" style="
            background: linear-gradient(135deg, #232946 60%, #6a89cc 100%);
            border-radius: 16px;
            box-shadow: 0 4px 24px #23294688;
            padding: 1.2rem 1rem 1rem 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            max-width: 260px;
            min-width: 220px;
        ">
            <div style="display: flex; justify-content: space-between; width: 100%; gap: 0.5rem;">
                <div style="flex:1;">
                    <span style="font-size:2.2rem; color:#ffb74d; text-shadow:0 0 8px #ffb74d88;">82.2%</span>
                    <div style="color:#e0e6f7; font-weight:bold; font-size:1rem;">mAP@0.5</div>
                </div>
                <div style="flex:1;">
                    <span style="font-size:2.2rem; color:#90caf9; text-shadow:0 0 8px #90caf988;">89.2%</span>
                    <div style="color:#e0e6f7; font-weight:bold; font-size:1rem;">Precision</div>
                </div>
            </div>
            <div style="margin: 0.7rem 0 0.3rem 0;">
                <span style="font-size:2.2rem; color:#66bb6a; text-shadow:0 0 8px #66bb6a88;">74.3%</span>
                <div style="color:#e0e6f7; font-weight:bold; font-size:1rem;">Recall</div>
            </div>
            <div class="sidebar-metric-footer" style="margin-top:0.7rem; font-size:1rem; color:#e0e6f7;">
                YOLOv8 Model • <span style="color:#ffb74d;">Falcon Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-classes-title">🎯 Detection Classes</div>
        """, unsafe_allow_html=True)
        for i, class_name in enumerate(class_names):
            color_class = ["red", "green", "blue", "yellow", "orange"][i % 5]
            emoji = ["🔴", "🟢", "🔵", "🟡", "🟠"][i % 5]
            st.markdown(
                f"""
                <div class="sidebar-class-chip {color_class}">
                    {emoji} {class_name}
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("""
        <div class="sidebar-threshold-title">Confidence Threshold</div>
        """, unsafe_allow_html=True)
        confidence_threshold = st.slider(
            "",
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
        *Falcon* enables continuous model improvement:
        
        🔄 *Incremental Learning*: Update model with new data without full retraining
        
        📈 *Performance Tracking*: Monitor model drift and performance degradation
        
        🎯 *Active Learning*: Automatically identify images that need labeling
        
        🔧 *Model Versioning*: Track different model versions and their performance
        
        *Benefits:*
        - Reduce retraining time by 80%
        - Maintain model accuracy over time
        - Automate the ML lifecycle
        - Scale to production efficiently
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sidebar: Add class filter
        st.markdown("## 🗂 Class Filter")
        selected_classes = st.multiselect(
            "Select classes to display",
            options=class_names,
            default=class_names
        )
        
    # Main content
    col1, _ = st.columns([1, 0.01])  # Only use col1 for results

    with col1:
        st.markdown("## 📤 Upload Images (Batch)")
        uploaded_files = st.file_uploader(
            "Choose one or more space station images...",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload images of a space station to detect safety equipment"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.markdown(f"### 🖼 {uploaded_file.name}")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

                img_bytes = uploaded_file.getvalue()
                img_array = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                with st.spinner(f"Running detection on {uploaded_file.name}..."):
                    results = model.predict(
                        img_cv,
                        conf=confidence_threshold,
                        verbose=False
                    )
                    annotated_image, detections = draw_detections(image, results, class_names)
                    detections = [d for d in detections if d["class"] in selected_classes]

                    st.session_state["history"].append({
                        "filename": uploaded_file.name,
                        "detections": detections,
                        "image": annotated_image.copy()
                    })

                    # Synchronized display: image and detection details side-by-side
                    img_col, det_col = st.columns([2, 3])
                    with img_col:
                        st.image(annotated_image, caption="Detection Results", use_container_width=True)
                        st.markdown(get_image_download_link(annotated_image), unsafe_allow_html=True)
                    with det_col:
                        if detections:
                            st.markdown("#### 📋 Detection Details")
                            for detection in detections:
                                st.write(f"{detection['class']}")
                                st.progress(detection['confidence'])
                                st.write(f"Confidence: {detection['confidence']:.2f}")
                            st.markdown("#### 📊 Summary")
                            class_counts = {}
                            for detection in detections:
                                class_name = detection['class']
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            for class_name, count in class_counts.items():
                                st.write(f"• {class_name}: {count} detected")
                        else:
                            st.info("No objects detected with the current confidence threshold.")
                            st.write("Try lowering the confidence threshold in the sidebar.")

    # Detection history section
    st.markdown("---")
    st.markdown("## 🕑 Detection History")

    with st.expander("Show Detection History (Last 5)", expanded=False):
        if st.session_state["history"]:
            for entry in st.session_state["history"][-5:][::-1]:  # Show last 5
                st.write(f"{entry['filename']}")
                st.image(entry["image"], caption="Previous Detection", use_container_width=True)
                for d in entry["detections"]:
                    st.write(f"- {d['class']} ({d['confidence']:.2f})")
                st.markdown("---")
        else:
            st.info("No detection history yet.")

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
    company_details_section()

# Add enhanced UI animations and transitions
st.markdown("""
<style>
/* Animated card entrance */
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(40px);}
    100% { opacity: 1; transform: translateY(0);}
}
.card, .sidebar-metric-card, .det-detail, .sidebar-class-chip {
    animation: fadeInUp 0.7s cubic-bezier(.23,1.01,.32,1) both;
}

/* Animated hover for detection details */
.det-detail {
    transition: box-shadow 0.3s, background 0.3s, transform 0.3s;
}
.det-detail:hover {
    background: #232946;
    box-shadow: 0 4px 16px #ffb74d88, 0 0 24px #90caf988;
    transform: scale(1.04) rotate(-1deg);
}

/* Animated button pulse */
.stButton > button, .stDownloadButton > button {
    position: relative;
    overflow: hidden;
}
.stButton > button::after, .stDownloadButton > button::after {
    content: "";
    position: absolute;
    left: 50%; top: 50%;
    width: 0; height: 0;
    background: rgba(255,183,77,0.25);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.4s, height 0.4s;
    z-index: 0;
}
.stButton > button:active::after, .stDownloadButton > button:active::after {
    width: 120px; height: 120px;
    transition: width 0.2s, height 0.2s;
}

/* Animated progress bar */
.stProgress > div > div {
    transition: width 0.7s cubic-bezier(.23,1.01,.32,1);
}

/* Animated expander arrow */
.stExpanderHeader:after {
    transition: transform 0.3s;
}
.stExpanderHeader:hover:after {
    transform: scale(1.2) rotate(10deg);
}

/* Animated detection class chip hover */
.sidebar-class-chip {
    transition: box-shadow 0.3s, transform 0.3s, background 0.3s;
}
.sidebar-class-chip:hover {
    box-shadow: 0 4px 16px #ffb74d88, 0 0 24px #90caf988;
    background: #232946;
    transform: scale(1.08) rotate(-2deg);
}

/* Animated detection history image hover */
[data-testid="stImage"] img {
    transition: box-shadow 0.3s, transform 0.3s;
}
[data-testid="stImage"]:hover img {
    box-shadow: 0 8px 32px #ffb74d88, 0 0 32px #6a89cc88;
    transform: scale(1.03) rotate(-1deg);
}
</style>
""", unsafe_allow_html=True)
