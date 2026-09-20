import os
import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import segmentation_models_pytorch as smp

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal ROP Screening Engine",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Multimodal Retinopathy of Prematurity (ROP) Screening Engine")
st.markdown(
    "Automated clinical decision support integrating Microvascular Segmentation, "
    "Tri-Architecture Staging, ICROP-3 Zone Boundary Analysis, Plus Tortuosity, and ETROP Triage."
)

# Force CPU inference on free hosting instances
DEVICE = torch.device("cpu")
IMAGE_SIZE = (384, 384)

# ------------------------------------------------------------------------------
# DIRECT CHECKPOINT DOWNLOADER & RESOLVER
# ------------------------------------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE_IDS = {
    "unetpp_vessel_best.pth": "1sL-GILhsFhaEkmybQKQd7eFIRUIIP8sG",
    "binary_rop_best.pth": "1KNM1Q1pT9XI23cEfXwnYVcJa2GTPYsgi",
    "patho_stage_best.pth": "11kQM6-Ag5PjPDbf1cASQhvvKKOr_Hddf",
    "unified_rop_best.pth": "1yU4NoUS0arwFpA5VbWjlvBom7AruuiqV",
    "convnext_rop_best.pth": "1tJCRuNBEZemvRVwORKITKhELynJH3xaG",
    "best.pt": "1sZZ79w5OQflUy1EP6sHwlAUv7syw4PCk",
    "zone.pt": "1sA9DgOk4BLULgLQvXi6AE6rODIL63-WX",
    "efficientnet_b4_plus_best.pth": "1vfKCQRXtr-EmtZS9ZDyrKTvDOtDHLdQ3"
}

@st.cache_resource(show_spinner=False)
def ensure_models_downloaded():
    for fname, file_id in MODEL_FILE_IDS.items():
        dst_path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
            with st.spinner(f"Downloading {fname} from Google Drive..."):
                gdown.download(id=file_id, output=dst_path, quiet=False)
    return True

ensure_models_downloaded()

def get_path(fname):
    p = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        raise FileNotFoundError(f"Missing required model checkpoint: {fname}")
    return p

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURES
# ------------------------------------------------------------------------------
class BaseMultiModalNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        stem = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            4, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False
        )
        in_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.tabular_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim + 32, 256), nn.BatchNorm1d(256),
            nn.SiLU(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, img, mask, tab):
        x = torch.cat([img, mask], dim=1)
        return self.classifier(torch.cat([self.backbone(x), self.tabular_mlp(tab)], dim=1))

class UnifiedROPNet(nn.Module):
    def __init__(self, tabular_dim=2):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        stem = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            4, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False
        )
        visual_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU()
        )
        self.shared_dense = nn.Sequential(
            nn.Linear(visual_dim + 32, 256), nn.BatchNorm1d(256),
            nn.SiLU(), nn.Dropout(0.3)
        )
        self.head_binary = nn.Linear(256, 2)
        self.head_stage = nn.Linear(256, 5)

    def forward(self, img, mask, tabular):
        x_vis = torch.cat([img, mask], dim=1)
        fused = torch.cat([self.backbone(x_vis), self.tabular_mlp(tabular)], dim=1)
        shared = self.shared_dense(fused)
        return self.head_binary(shared), self.head_stage(shared)

class ConvNeXtMultiTaskROPNet(nn.Module):
    def __init__(self, tabular_dim=2):
        super().__init__()
        self.backbone = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, in_chans=4, num_classes=0)
        visual_dim = self.backbone.num_features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(32, 32), nn.BatchNorm1d(32), nn.GELU()
        )
        self.shared_dense = nn.Sequential(
            nn.Linear(visual_dim + 32, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.3)
        )
        self.head_binary = nn.Linear(256, 2)
        self.head_stage = nn.Linear(256, 5)

    def forward(self, img, mask, tabular):
        x_vis = torch.cat([img, mask], dim=1)
        fused = torch.cat([self.backbone(x_vis), self.tabular_mlp(tabular)], dim=1)
        shared = self.shared_dense(fused)
        return self.head_binary(shared), self.head_stage(shared)

def build_4channel_plus_model():
    m = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    orig_conv = m.conv_stem
    m.conv_stem = nn.Conv2d(4, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                            stride=orig_conv.stride, padding=orig_conv.padding, bias=(orig_conv.bias is not None))
    return m

# ------------------------------------------------------------------------------
# LOAD MODELS ONCE INTO MEMORY
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_models():
    unet = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    u_ckpt = torch.load(get_path("unetpp_vessel_best.pth"), map_location=DEVICE, weights_only=False)
    unet.load_state_dict(u_ckpt.get('state_dict', u_ckpt.get('model_state_dict', u_ckpt)))
    unet.to(DEVICE).eval()

    bin_m = BaseMultiModalNet(num_classes=2).to(DEVICE)
    bin_m.load_state_dict(torch.load(get_path("binary_rop_best.pth"), map_location=DEVICE, weights_only=False)['model_state_dict'])
    bin_m.eval()

    patho_m = BaseMultiModalNet(num_classes=4).to(DEVICE)
    patho_m.load_state_dict(torch.load(get_path("patho_stage_best.pth"), map_location=DEVICE, weights_only=False)['model_state_dict'])
    patho_m.eval()

    eff_u = UnifiedROPNet().to(DEVICE)
    eff_u.load_state_dict(torch.load(get_path("unified_rop_best.pth"), map_location=DEVICE, weights_only=False)['model_state_dict'])
    eff_u.eval()

    conv_m = ConvNeXtMultiTaskROPNet().to(DEVICE)
    conv_m.load_state_dict(torch.load(get_path("convnext_rop_best.pth"), map_location=DEVICE, weights_only=False)['model_state_dict'])
    conv_m.eval()

    plus_m = build_4channel_plus_model().to(DEVICE)
    plus_m.load_state_dict(torch.load(get_path("efficientnet_b4_plus_best.pth"), map_location=DEVICE, weights_only=False)['model_state_dict'])
    plus_m.eval()

    y_od = YOLO(get_path("best.pt"))
    y_ridge = YOLO(get_path("zone.pt"))

    return unet, bin_m, patho_m, eff_u, conv_m, plus_m, y_od, y_ridge

with st.spinner("Loading clinical inference pipelines into memory..."):
    unet_model, bin_model, patho_stager, eff_unified, convnext_model, plus_model, yolo_od, yolo_ridge = load_all_models()

STAGE_NAMES = [
    'Stage 0 (Normal)',
    'Stage 1 (Demarcation Line)',
    'Stage 2 (Ridge)',
    'Stage 3 (Extraretinal Neovascularization)',
    'Stage 4+ (Retinal Detachment)'
]

# ------------------------------------------------------------------------------
# CLINICAL DECISION LOGIC
# ------------------------------------------------------------------------------
def estimate_blindness_risk(stage, zone, plus_status, plus_prob, ga_weeks=28.0, bw_grams=1000.0):
    if stage >= 4:
        base_risk = 85.0 + (stage - 4) * 10.0
    elif zone == "Zone I":
        base_risk = 55.0 if stage == 3 else (18.0 if stage in [1, 2] else 5.0)
    elif zone == "Zone II":
        base_risk = 22.0 if stage == 3 else (8.0 if stage == 2 else (3.0 if stage == 1 else 1.0))
    elif zone == "Zone III":
        base_risk = 2.0 if stage > 0 else 0.1
    else:
        base_risk = 0.05

    if plus_status == "Plus":
        if zone == "Zone I":
            base_risk = max(base_risk * 2.8, 65.0)
        elif zone == "Zone II":
            base_risk = max(base_risk * 2.5, 45.0)
        else:
            base_risk = max(base_risk * 1.8, 20.0)
    else:
        base_risk += (plus_prob * 4.0)

    demo_penalty = 0.0
    if ga_weeks < 28.0:
        demo_penalty += (28.0 - ga_weeks) * 1.5
    if bw_grams < 1000.0:
        demo_penalty += ((1000.0 - bw_grams) / 100.0) * 1.0

    total_risk = float(np.clip(base_risk + demo_penalty, 0.1, 99.0))
    risk_category = "CRITICAL / IMMINENT" if total_risk >= 50.0 else ("MODERATE" if total_risk >= 15.0 else "LOW")
    return {"risk_percentage": round(total_risk, 1), "risk_category": risk_category}

def evaluate_etrop_risk(stage, zone, plus_status):
    if stage == 0 or zone == "Fully Vascularized":
        return {
            "level": "LOW RISK", "action": "Routine Monitoring",
            "clinical_code": "ETROP Low Risk (Non-Pathological / Fully Vascularized)",
            "urgency": "Standard follow-up at next routine visit", "color": "#27AE60"
        }
    if stage >= 4:
        return {
            "level": "HIGH RISK", "action": "Urgent Treatment within 48-72 Hours",
            "clinical_code": "Type 1 ROP: Advanced Retinal Detachment (Stage 4+)",
            "urgency": "Immediate bedside laser photocoagulation or anti-VEGF injection", "color": "#E74C3C"
        }
    if plus_status == "Plus" and zone in ["Zone I", "Zone II"]:
        return {
            "level": "HIGH RISK", "action": "Urgent Treatment within 48-72 Hours",
            "clinical_code": f"Type 1 ROP: Severe Posterior Vessel Tortuosity ({zone} with Plus)",
            "urgency": "Urgent retinal intervention required within 48 to 72 hours", "color": "#E74C3C"
        }
    if stage == 3 and zone == "Zone I":
        return {
            "level": "HIGH RISK", "action": "Urgent Treatment within 48-72 Hours",
            "clinical_code": "Type 1 ROP: Zone I Stage 3 Neovascularization",
            "urgency": "Urgent treatment required to prevent detachment", "color": "#E74C3C"
        }
    if plus_status == "Normal":
        if stage in [1, 2] and zone == "Zone I":
            return {
                "level": "MODERATE RISK", "action": "Rescreen in 3 to 7 Days",
                "clinical_code": "Type 2 ROP: Posterior Stage 1/2 without Plus",
                "urgency": "Close serial surveillance to catch potential progression", "color": "#F39C12"
            }
        if stage == 3 and zone == "Zone II":
            return {
                "level": "MODERATE RISK", "action": "Rescreen in 3 to 7 Days",
                "clinical_code": "Type 2 ROP: Zone II Stage 3 without Plus",
                "urgency": "Re-evaluate within 3 to 7 days for development of Plus disease", "color": "#F39C12"
            }
    return {
        "level": "LOW RISK", "action": "Routine Monitoring (1 to 2 Weeks)",
        "clinical_code": f"Low Risk: Mild Disease ({zone}, Stage {stage} without Plus)",
        "urgency": "Routine serial examination until full retinal vascularization", "color": "#27AE60"
    }

# ------------------------------------------------------------------------------
# USER INPUT SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.header("📋 Patient Demographics")
uploaded_file = st.sidebar.file_uploader("Upload Retinal Fundus Scan", type=["jpg", "jpeg", "png"])
ga_weeks = st.sidebar.number_input("Gestational Age (weeks)", min_value=22.0, max_value=42.0, value=28.0, step=0.5)
bw_grams = st.sidebar.number_input("Birth Weight (grams)", min_value=350.0, max_value=3500.0, value=1000.0, step=10.0)

run_button = st.sidebar.button("🚀 Run Screening Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# INFERENCE PIPELINE
# ------------------------------------------------------------------------------
if run_button and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = raw_rgb.shape[:2]

    with st.spinner("Processing retinal scan across multi-model pipeline..."):
        # 1. Vessel Segmentation
        u_in = cv2.resize(raw_rgb, (512, 512)).astype(np.float32) / 255.0
        u_in = (u_in - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        u_tensor = torch.from_numpy(u_in).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            v_out = unet_model(u_tensor)
            vmask_512 = (torch.sigmoid(v_out).squeeze().numpy() > 0.5).astype(np.uint8)
            vmask_orig = cv2.resize(vmask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # 2. Stage Classification
        img_384 = cv2.resize(raw_rgb, IMAGE_SIZE)
        mask_384 = cv2.resize(vmask_orig, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        img_t = TF.normalize(TF.to_tensor(img_384), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(DEVICE)
        mask_t = TF.to_tensor(mask_384).unsqueeze(0).to(DEVICE)
        tab_t = torch.tensor([[np.clip((ga_weeks - 22.0) / 16.0, 0, 1), np.clip((bw_grams - 400.0) / 2600.0, 0, 1)]], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            _, c_logits = convnext_model(img_t, mask_t, tab_t)
            _, e_logits = eff_unified(img_t, mask_t, tab_t)
            p_bin = torch.softmax(bin_model(img_t, mask_t, tab_t), dim=1).numpy()[0]
            p_patho = torch.softmax(patho_stager(img_t, mask_t, tab_t), dim=1).numpy()[0]

            p_conv = torch.softmax(c_logits, dim=1).numpy()[0]
            p_eff = torch.softmax(e_logits, dim=1).numpy()[0]
            p_hier = np.zeros(5)
            p_hier[0] = p_bin[0]
            for k in range(4):
                p_hier[k+1] = p_bin[1] * p_patho[k]

            p_stage = (0.27 * p_conv) + (0.42 * p_eff) + (0.31 * p_hier)
            pred_stage = int(np.argmax(p_stage))
            stage_conf = p_stage[pred_stage]

        # 3. ICROP-3 Zone Detection
        cv2.imwrite("temp_eval.jpg", raw_bgr)
        od_res = yolo_od.predict("temp_eval.jpg", conf=0.25, verbose=False)[0]
        if len(od_res.boxes) > 0:
            box = od_res.boxes.xyxy.cpu().numpy()[0].astype(int)
            x_od, y_od = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            dd = max(box[2] - box[0], box[3] - box[1])
        else:
            x_od, y_od, dd = int(w_orig / 2), int(h_orig / 2), 50

        r_z1 = 2.0 * (3.0 * float(dd))
        r_z2 = 3.0 * (3.0 * float(dd))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(clahe.apply(raw_rgb[:, :, 1]), cv2.COLOR_GRAY2BGR)
        ridge_res = yolo_ridge.predict(enhanced, conf=0.15, verbose=False)[0]

        assigned_zone = "Zone III"
        if len(ridge_res.boxes) > 0 and ridge_res.masks is not None:
            min_d = float("inf")
            for m in ridge_res.masks.data.cpu().numpy():
                m_full = cv2.resize(m, (w_orig, h_orig)) > 0.5
                ys, xs = np.where(m_full)
                if len(xs) > 0:
                    d = float(np.percentile(np.sqrt((xs - x_od) ** 2 + (ys - y_od) ** 2), 5))
                    min_d = min(min_d, d)
            if min_d <= r_z1:
                assigned_zone = "Zone I"
            elif min_d <= r_z2:
                assigned_zone = "Zone II"

        # 4. Plus Disease Detection
        with torch.no_grad():
            plus_prob = torch.sigmoid(plus_model(torch.cat([img_t, mask_t], dim=1))).item()
        plus_status = "Plus" if plus_prob >= 0.5 else "Normal"

        # 5. Risk & Triage Formulation
        blindness = estimate_blindness_risk(pred_stage, assigned_zone, plus_status, plus_prob, ga_weeks, bw_grams)
        triage = evaluate_etrop_risk(pred_stage, assigned_zone, plus_status)

    # --------------------------------------------------------------------------
    # RENDER INTERFACE OUTPUT
    # --------------------------------------------------------------------------
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Diagnostic Visualizations")
        canvas = raw_rgb.copy()
        cv2.circle(canvas, (x_od, y_od), int(r_z1), (255, 0, 255), 3)
        cv2.circle(canvas, (x_od, y_od), int(r_z2), (255, 165, 0), 3)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(vmask_orig, cmap="gray")
        ax[0].set_title("Vessel Segmentation (U-Net++)")
        ax[0].axis("off")
        ax[1].imshow(canvas)
        ax[1].set_title(f"Zone Demarcation: {assigned_zone}")
        ax[1].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        risk_color = "#E74C3C" if blindness['risk_category'] == "CRITICAL / IMMINENT" else (
                     "#F39C12" if blindness['risk_category'] == "MODERATE" else "#27AE60")

        st.subheader("📋 Patient Diagnostic Findings")
        st.markdown(f"- **Severity Stage**: `{STAGE_NAMES[pred_stage]}` ({stage_conf*100:.1f}%)")
        st.markdown(f"- **ICROP-3 Zone**: `{assigned_zone}`")
        st.markdown(f"- **Plus Disease Status**: `{plus_status.upper()}` ({plus_prob*100:.1f}%)")
        st.markdown(f"- **Demographics**: `GA = {ga_weeks} wks` | `BW = {bw_grams} g`")

        #st.markdown("---")
        #st.subheader("👁️ Estimated Risk of Vision Loss / Blindness")
        #st.markdown(f"**Calculated Probability**: <span style='font-size:1.4em; color:{risk_color}; font-weight:bold;'>{blindness['risk_percentage']}%</span>", unsafe_allow_html=True)
        #st.markdown(f"**Clinical Category**: <span style='color:{risk_color}; font-weight:bold;'>{blindness['risk_category']}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🚨 ETROP Clinical Triage")
        st.markdown(f"**Triage Status**: <span style='color:{triage['color']}; font-weight:bold;'>{triage['level']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Prescribed Action**: **{triage['action']}**")
        st.markdown(f"**Urgency**: *{triage['urgency']}*")
        st.markdown(f"**Protocol Reference**: `{triage['clinical_code']}`")

elif run_button and uploaded_file is None:
    st.warning("⚠️ Please upload a retinal fundus image before initiating screening analysis.")
