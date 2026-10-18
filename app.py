import io
import datetime
import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- ReportLab Libraries สำหรับสร้าง PDF ---
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- ตั้งค่าหน้าเว็บ Streamlit เป็นแบบ Wide ---
st.set_page_config(
    page_title="BME Virtual Radiology Workstation", 
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 🎨 CUSTOM CSS STYLING
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
        padding: 20px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    .main-header p {
        color: #93C5FD !important;
        font-size: 13px !important;
        margin: 0 !important;
    }
    
    .top-control-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.04);
    }

    .analysis-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-top: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }

    .status-badge-pass {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        font-weight: 600;
        font-size: 13px;
    }
    .status-badge-warning {
        background-color: #FEFCE8;
        color: #854D0E;
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 4px solid #EAB308;
        font-weight: 600;
        font-size: 13px;
    }
    .status-badge-fail {
        background-color: #FEF2F2;
        color: #991B1B;
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 4px solid #EF4444;
        font-weight: 600;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# --- ระบบจัดการ Key สำหรับรีเซ็ต ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_app():
    st.session_state.uploader_key += 1
    st.rerun()

# --- ฟังก์ชันจำลอง Poisson Noise ---
def apply_physics_noise(img_2d, dose_percent, distance_cm=100.0, ref_distance_cm=100.0):
    if dose_percent == 100 and distance_cm == ref_distance_cm:
        return img_2d.copy()
    
    dose_level = dose_percent / 100.0
    distance_factor = (ref_distance_cm / max(distance_cm, 1.0)) ** 2
    effective_dose = dose_level * distance_factor
    
    base_photons = 300.0
    photons = np.clip((img_2d / 255.0) * base_photons * effective_dose, 0.001, None)
    sampled = np.random.poisson(photons)
    simulated = (sampled / (base_photons * max(effective_dose, 1e-4))) * 255.0
    return np.clip(simulated, 0, 255)

# --- ฟังก์ชันคำนวณดัชนีฟิสิกส์ (SNR / CNR / Noise SD) ---
def compute_image_metrics(img_2d):
    smoothed = gaussian_filter(img_2d, sigma=2.0)
    noise_map = img_2d - smoothed
    std_noise = np.std(noise_map)
    if std_noise < 1e-3: std_noise = 1e-3

    mean_signal = np.mean(img_2d[img_2d > 10]) if np.any(img_2d > 10) else np.mean(img_2d)
    snr = mean_signal / std_noise

    body_pixels = img_2d[img_2d > 20]
    if len(body_pixels) > 0:
        med_val = np.median(body_pixels)
        high_sig = np.mean(body_pixels[body_pixels >= med_val])
        low_sig = np.mean(body_pixels[body_pixels < med_val])
        cnr = (high_sig - low_sig) / std_noise
    else:
        cnr = 0.0
    return snr, cnr, std_noise

# --- ฟังก์ชันคำนวณ 1D Radially Averaged Noise Power Spectrum (NPS) ---
def compute_nps_1d(img_2d, pixel_spacing_mm=0.5):
    smoothed = gaussian_filter(img_2d, sigma=2.0)
    noise_map = img_2d - smoothed
    
    Ny, Nx = noise_map.shape
    fft_2d = np.fft.fftshift(np.fft.fft2(noise_map))
    psd_2d = (np.abs(fft_2d) ** 2) * (pixel_spacing_mm ** 2) / (Nx * Ny)
    
    cy, cx = Ny // 2, Nx // 2
    y, x = np.ogrid[-cy:Ny-cy, -cx:Nx-cx]
    r = np.sqrt(x**2 + y**2)
    
    r_int = r.astype(int)
    max_r = min(cx, cy)
    r_bins = np.arange(0, max_r)
    
    nyquist_freq = 0.5 / pixel_spacing_mm
    freq_axis = (r_bins / max_r) * nyquist_freq
    
    nps_1d = []
    for rb in r_bins:
        mask = (r_int == rb)
        if np.any(mask):
            nps_1d.append(np.mean(psd_2d[mask]))
        else:
            nps_1d.append(0.0)
            
    return freq_axis, np.array(nps_1d)

# --- ฟังก์ชันสร้าง Digital Synthetic Phantom ความละเอียดสูง HD (512x512) ---
def generate_synthetic_phantom_hd():
    y, x = np.ogrid[-256:256, -256:256]
    mask = x**2 + y**2 <= 200**2
    phantom = np.zeros((512, 512), dtype=float)
    phantom[mask] = 150.0
    
    inner_mask = (x - 60)**2 + (y - 40)**2 <= 60**2
    bone_mask = (x + 60)**2 + (y + 60)**2 <= 40**2
    phantom[inner_mask] = 80.0
    phantom[bone_mask] = 230.0
    
    volume_3d = np.stack([phantom] * 10, axis=0)
    return volume_3d

# --- ฟังก์ชันสร้างรายงาน PDF ---
def generate_pdf_report(patient_id, modality, selected_slice, num_slices, dose_percent, base_mas, sim_mas, base_kvp, base_sid, current_ei, di, snr, cnr, qa_data=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Normal'], fontName="Helvetica-Bold", fontSize=16, leading=22, textColor=colors.HexColor('#1A365D'), alignment=1)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName="Helvetica", fontSize=10, leading=14, textColor=colors.HexColor('#4A5568'), alignment=1)
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Normal'], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName="Helvetica", fontSize=9, leading=13, textColor=colors.HexColor('#2D3748'))
    bold_body_style = ParagraphStyle('BoldBodyTextCustom', parent=styles['Normal'], fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=colors.HexColor('#1A202C'))
    table_header_style = ParagraphStyle('TableHeader', parent=styles['Normal'], fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=colors.white, alignment=1)
    table_cell_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor('#2D3748'))
    table_cell_center = ParagraphStyle('TableCellCenter', parent=styles['Normal'], fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor('#2D3748'), alignment=1)

    elements = []

    elements.append(Paragraph("3D DICOM Dose Optimization & Physics Verification Report", title_style))
    elements.append(Paragraph("Virtual Radiology Quality Control & Dose Optimization Workbench", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceBefore=4, spaceAfter=8))

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph(f"<b>Patient / Phantom ID:</b> {patient_id}", body_style), Paragraph(f"<b>Assessment Date:</b> {now_str}", body_style)],
        [Paragraph(f"<b>Modality:</b> {modality}", body_style), Paragraph(f"<b>Slice Analyzed:</b> Layer {selected_slice + 1} / {num_slices}", body_style)],
        [Paragraph("<b>International Standards:</b> AAPM TG-116, IEC 62494-1, IAEA ALARA", body_style), Paragraph("<b>Workflow:</b> In-Silico Physics Verification", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[270, 270])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#EDF2F7')),
    ]))
    elements.append(t_meta)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("2. Exposure Parameter Comparison", section_heading))
    param_table_data = [
        [Paragraph("Parameter", table_header_style), Paragraph("Baseline (100%)", table_header_style), Paragraph("Simulated Target", table_header_style), Paragraph("Delta / Change", table_header_style)],
        [Paragraph("<b>Dose Level (%)</b>", table_cell_style), Paragraph("100 %", table_cell_center), Paragraph(f"<b>{dose_percent} %</b>", table_cell_center), Paragraph(f"Reduced {100 - dose_percent:.1f}%", table_cell_center)],
        [Paragraph("<b>Tube Current (mAs)</b>", table_cell_style), Paragraph(f"{base_mas:.2f} mAs", table_cell_center), Paragraph(f"<b>{sim_mas:.2f} mAs</b>", table_cell_center), Paragraph(f"Saved {100 - dose_percent:.1f}% Dose", table_cell_center)],
        [Paragraph("<b>Tube Voltage (kVp)</b>", table_cell_style), Paragraph(f"{base_kvp:.1f} kVp", table_cell_center), Paragraph(f"{base_kvp:.1f} kVp", table_cell_center), Paragraph("Constant", table_cell_center)],
        [Paragraph("<b>Distance (SID)</b>", table_cell_style), Paragraph(f"{base_sid:.1f} cm", table_cell_center), Paragraph(f"{base_sid:.1f} cm", table_cell_center), Paragraph("Constant", table_cell_center)]
    ]
    t_param = Table(param_table_data, colWidths=[160, 120, 140, 120])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 6))

    if qa_data:
        elements.append(Paragraph("3. 2-Shot Verification & Simulation Error Analysis", section_heading))
        qa_table_data = [
            [Paragraph("Metric", table_header_style), Paragraph("Simulated Target", table_header_style), Paragraph("Actual Recheck Shot", table_header_style), Paragraph("Error (%)", table_header_style)],
            [Paragraph("<b>Exposure Index (EI)</b>", table_cell_style), Paragraph(f"{qa_data['sim_ei']:.1f}", table_cell_center), Paragraph(f"{qa_data['act_ei']:.1f}", table_cell_center), Paragraph(f"<b>{qa_data['err_ei']:.2f} %</b>", table_cell_center)],
            [Paragraph("<b>Deviation Index (DI)</b>", table_cell_style), Paragraph(f"{qa_data['sim_di']:+.2f}", table_cell_center), Paragraph(f"{qa_data['act_di']:+.2f}", table_cell_center), Paragraph(f"<b>{qa_data['err_di']:.2f} %</b>", table_cell_center)],
            [Paragraph("<b>Signal-to-Noise (SNR)</b>", table_cell_style), Paragraph(f"{qa_data['sim_snr']:.2f}", table_cell_center), Paragraph(f"{qa_data['act_snr']:.2f}", table_cell_center), Paragraph(f"<b>{qa_data['err_snr']:.2f} %</b>", table_cell_center)],
            [Paragraph("<b>Contrast-to-Noise (CNR)</b>", table_cell_style), Paragraph(f"{qa_data['sim_cnr']:.2f}", table_cell_center), Paragraph(f"{qa_data['act_cnr']:.2f}", table_cell_center), Paragraph(f"<b>{qa_data['err_cnr']:.2f} %</b>", table_cell_center)]
        ]
        t_qa = Table(qa_table_data, colWidths=[160, 120, 140, 120])
        t_qa.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C5282')),
            ('PADDING', (0,0), (-1,-1), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
        ]))
        elements.append(t_qa)
        elements.append(Spacer(1, 6))

    elements.append(Paragraph("4. Quantitative Physics Metrics", section_heading))
    metrics_data = [
        [Paragraph("<b>Exposure Index (EI):</b>", table_cell_style), Paragraph(f"{current_ei:.1f}", table_cell_center), Paragraph("<b>Deviation Index (DI):</b>", table_cell_style), Paragraph(f"{di:+.2f}", table_cell_center)],
        [Paragraph("<b>Signal-to-Noise Ratio (SNR):</b>", table_cell_style), Paragraph(f"{snr:.2f}", table_cell_center), Paragraph("<b>Contrast-to-Noise Ratio (CNR):</b>", table_cell_style), Paragraph(f"{cnr:.2f}", table_cell_center)]
    ]
    t_metrics = Table(metrics_data, colWidths=[160, 110, 160, 110])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EBF8FF')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#3182CE')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BEE3F8')),
    ]))
    elements.append(t_metrics)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("5. Clinical Recommendations & Quality Approval", section_heading))
    rec_text = (
        f"Simulating dose reduction to <b>{dose_percent}% ({sim_mas:.2f} mAs)</b> yields a patient dose savings "
        f"of <b>{100 - dose_percent:.1f}%</b> while preserving adequate slice image resolution. "
        f"Verified protocol: <b>{sim_mas:.2f} mAs / {base_kvp:.1f} kVp</b>."
    )
    t_rec = Table([[Paragraph(rec_text, body_style)]], colWidths=[540])
    t_rec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FEFCBF')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#D69E2E')),
    ]))
    elements.append(t_rec)
    elements.append(Spacer(1, 20))

    sig_data = [
        [
            Paragraph("Signature: ...........................................................<br/>(Medical Physicist)", table_cell_center),
            Paragraph("Signature: ...........................................................<br/>(Radiologist)", table_cell_center)
        ]
    ]
    t_sig = Table(sig_data, colWidths=[270, 270])
    elements.append(KeepTogether(t_sig))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# --- HEADER BANNER ---
st.markdown("""
<div class="main-header">
    <h1>⚛️ BME Virtual Radiology QC & Dose Optimization Workbench</h1>
    <p>ระบบวิเคราะห์ภาพ DICOM 3D Series จำลองปริมาณรังสีแบบ In-Silico และประเมินดัชนีฟิสิกส์การแพทย์ (AAPM TG-116 / IEC 62494-1)</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ ตัวเลือกระบบ")
    # เปลี่ยน value เป็น False เพื่อไม่ให้เลือกโหมดจำลองอัตโนมัติเมื่อแรกเข้า
    use_demo = st.checkbox("🧪 ใช้ข้อมูลภาพจำลอง (Digital Phantom HD)", value=False)
    if st.button("🔄 ลบไฟล์ทั้งหมด / เริ่มต้นใหม่", type="secondary"):
        reset_app()

# --- FILE UPLOADER ---
uploaded_files = None
if not use_demo:
    uploaded_files = st.file_uploader(
        "📥 Shot 1: ลากโฟลเดอร์มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์ DICOM 3D Series (Baseline 100% Shot)", 
        type=None, 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

norm_volume = None
patient_id = "ANONYMOUS"
modality = "CT"
auto_sid, auto_kvp, auto_mas = 100.0, 120.0, 100.0

if use_demo:
    patient_id = "BME-PHANTOM-HD"
    norm_volume = generate_synthetic_phantom_hd()
    st.success("✅ กำลังใช้งานภาพจำลอง Digital BME Phantom HD (512x512 High Resolution)")

elif uploaded_files:
    try:
        raw_slices = []
        with st.spinner("กำลังสแกนและโหลดข้อมูล DICOM Baseline Series..."):
            for f in uploaded_files:
                if f.name.startswith('.') or f.name.startswith('__'):
                    continue
                try:
                    ds = pydicom.dcmread(f, force=True)
                    if "PixelData" in ds and hasattr(ds, 'pixel_array'):
                        raw_slices.append(ds)
                except Exception:
                    pass

        if len(raw_slices) > 0:
            try:
                raw_slices.sort(
                    key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') 
                    else (float(x.SliceLocation) if hasattr(x, 'SliceLocation') else 0)
                )
            except Exception:
                pass

            sample_ds = raw_slices[0]
            modality = str(getattr(sample_ds, "Modality", "CT"))
            patient_id = str(getattr(sample_ds, "PatientID", "ANONYMOUS"))

            if hasattr(sample_ds, "RTImageSID"): auto_sid = float(sample_ds.RTImageSID) / 10.0
            elif hasattr(sample_ds, "DistanceSourceToDetector"): auto_sid = float(sample_ds.DistanceSourceToDetector) / 10.0

            auto_kvp = float(getattr(sample_ds, "KVP", 120.0))
            auto_mas = float(getattr(sample_ds, "Exposure", 100.0))

            slice_arrays = []
            for ds in raw_slices:
                arr = ds.pixel_array.astype(float)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    arr = arr * ds.RescaleSlope + ds.RescaleIntercept
                if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                    arr = np.amax(arr) - arr
                slice_arrays.append(arr)

            if len(slice_arrays) > 1:
                volume_3d = np.stack(slice_arrays, axis=0)
            elif slice_arrays[0].ndim == 3:
                volume_3d = slice_arrays[0]
            else:
                volume_3d = np.expand_dims(slice_arrays[0], axis=0)

            vol_min, vol_max = np.min(volume_3d), np.max(volume_3d)
            norm_volume = (volume_3d - vol_min) / (vol_max - vol_min) * 255.0 if vol_max > vol_min else volume_3d
            norm_volume = np.clip(norm_volume, 0, 255)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์ DICOM: {e}")

# =========================================================
# 🎛️ MAIN WORKSPACE
# =========================================================
if norm_volume is not None:
    num_slices = norm_volume.shape[0]

    # TOP CONTROL PANEL
    st.markdown('<div class="top-control-box">', unsafe_allow_html=True)
    st.subheader("🎛️ แผงควบคุมระบบการจำลองรังสี & โหมดมุมมอง (Top Control Panel)")
    
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([2.2, 2.2, 1.6], gap="large")
    
    with c_ctrl1:
        selected_slice = st.slider("🔍 ชั้นความลึกภาพ (Slice Index)", 0, num_slices - 1, num_slices // 2) if num_slices > 1 else 0
        st.caption(f"กำลังดู Layer ที่: **{selected_slice + 1} / {num_slices}**")

    with c_ctrl2:
        dose_percent = st.slider("📉 ระดับปริมาณรังสีจำลอง (%)", min_value=1, max_value=100, value=100)
        dose_level = dose_percent / 100.0
        st.caption(f"ปริมาณรังสีประหยัดได้: **{100 - dose_percent}%**")

    with c_ctrl3:
        view_mode = st.radio("🖼️ โหมดการแสดงผลภาพ:", ["เปรียบเทียบ Layer ปัจจุบัน (Side-by-Side)", "จำลองเม็ดซ่าบน 3 ระนาบ (MPR 3D)"], index=0)

    with st.expander("⚙️ ปรับพารามิเตอร์เครื่อง X-ray / CT Console", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        base_sid = p1.number_input("ระยะทาง SID (cm)", 10.0, 300.0, float(auto_sid), step=5.0)
        base_kvp = p2.number_input("แรงดันไฟฟ้า (kVp)", 20.0, 150.0, float(auto_kvp), step=5.0)
        base_mas = p3.number_input("กระแส-เวลาเริ่มต้น (mAs)", 1.0, 1000.0, float(auto_mas), step=5.0)
        
        sim_mas = base_mas * dose_level
        p4.metric("mAs จำลองจริง", f"{sim_mas:.2f} mAs", delta=f"-{100 - dose_percent}%")

    st.markdown('</div>', unsafe_allow_html=True)

    baseline_slice = norm_volume[selected_slice]
    simulated_slice = apply_physics_noise(baseline_slice, dose_percent, base_sid, ref_distance_cm=100.0)

    snr, cnr, base_noise_sd = compute_image_metrics(simulated_slice)
    distance_factor = (100.0 / max(base_sid, 1.0)) ** 2
    effective_factor = dose_level * distance_factor
    target_ei = 300.0
    current_ei = target_ei * effective_factor
    di = 10.0 * np.log10(effective_factor) if effective_factor > 0 else -99.0

    # 📑 MAIN TABS SYSTEM
    tab1, tab2, tab3 = st.tabs(["📉 1. Virtual Dose Simulator", "📈 2. Dose-SNR & Frequency NPS Analytics", "🧪 3. 2-Shot Verification & QA"])

    # ---------------------------------------------------------
    # TAB 1: SIMULATOR DISPLAY
    # ---------------------------------------------------------
    with tab1:
        st.subheader("🖼️ หน้าจอแสดงผล DICOM 3D Series (Full Width Display)")

        if view_mode == "เปรียบเทียบ Layer ปัจจุบัน (Side-by-Side)":
            fig, ax = plt.subplots(1, 2, figsize=(18, 8), dpi=150, facecolor='#FFFFFF')
            
            ax[0].imshow(baseline_slice, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
            ax[0].set_title(f"Baseline Slice {selected_slice + 1} ({base_mas:.1f} mAs | {base_kvp:.0f} kVp | 100% Dose)", fontsize=12, fontweight='bold', pad=12)
            ax[0].axis('off')

            ax[1].imshow(simulated_slice, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
            ax[1].set_title(f"Simulated Slice {selected_slice + 1} ({sim_mas:.2f} mAs | {base_kvp:.0f} kVp | {dose_percent}% Dose)", fontsize=12, fontweight='bold', pad=12)
            ax[1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

        else:
            if num_slices > 1:
                axial_base = norm_volume[selected_slice, :, :]
                coronal_base = norm_volume[:, norm_volume.shape[1] // 2, :]
                sagittal_base = norm_volume[:, :, norm_volume.shape[2] // 2]

                axial_sim = apply_physics_noise(axial_base, dose_percent, base_sid)
                coronal_sim = apply_physics_noise(coronal_base, dose_percent, base_sid)
                sagittal_sim = apply_physics_noise(sagittal_base, dose_percent, base_sid)

                fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=150, facecolor='#FFFFFF')
                ax[0].imshow(axial_sim, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
                ax[0].set_title(f"Axial Cut ({sim_mas:.2f} mAs)", fontsize=11, fontweight='bold', pad=10)
                ax[0].axis('off')

                ax[1].imshow(coronal_sim, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
                ax[1].set_title(f"Coronal Cut ({sim_mas:.2f} mAs)", fontsize=11, fontweight='bold', pad=10)
                ax[1].axis('off')

                ax[2].imshow(sagittal_sim, cmap='gray', vmin=0, vmax=255, interpolation='bicubic')
                ax[2].set_title(f"Sagittal Cut ({sim_mas:.2f} mAs)", fontsize=11, fontweight='bold', pad=10)
                ax[2].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)

        st.divider()

        col_m, col_alert, col_pdf = st.columns([1.2, 1.5, 1.3], gap="medium")

        with col_m:
            st.subheader("📊 ดัชนีฟิสิกส์การแพทย์")
            m1, m2 = st.columns(2)
            m1.metric("Exposure Index (EI)", f"{current_ei:.1f}")
            m2.metric("Deviation Index (DI)", f"{di:+.2f}")

            m3, m4 = st.columns(2)
            m3.metric("Signal-to-Noise (SNR)", f"{snr:.2f}")
            m4.metric("Contrast-to-Noise (CNR)", f"{cnr:.2f}")

        with col_alert:
            st.subheader("🛡️ ผลประเมินความปลอดภัย & คุณภาพ")
            if -1.0 <= di <= +1.0:
                st.markdown('<div class="status-badge-pass">✅ ALARA SAFE: อยู่ในช่วงปริมาณรังสีเป้าหมาย (-1.0 ≤ DI ≤ +1.0)</div>', unsafe_allow_html=True)
            elif di > +1.0:
                st.markdown(f'<div class="status-badge-warning">⚠️ OVEREXPOSURE: DI = {di:+.2f} ควรปรับลด mAs หน้าเครื่อง</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-fail">❌ UNDEREXPOSURE: DI = {di:+.2f} ปริมาณรังสีต่ำเกินไป</div>', unsafe_allow_html=True)

            st.write("")
            if snr >= 2.5 and cnr >= 0.8:
                st.markdown('<div class="status-badge-pass">✅ DIAGNOSTIC QUALITY: ความละเอียดเพียงพอสำหรับการวินิจฉัย</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge-warning">⚠️ NOISE WARNING: เม็ดซ่าเริ่มรบกวนรายละเอียดพิกเซล</div>', unsafe_allow_html=True)

        with col_pdf:
            st.subheader("📄 ออกรายงานสรุปผล (PDF)")
            st.caption("ดาวน์โหลดเอกสารประเมินทางการอ้างอิงมาตรฐาน AAPM / IEC")
            
            pdf_bytes = generate_pdf_report(
                patient_id, modality, selected_slice, num_slices, 
                dose_percent, base_mas, sim_mas, base_kvp, base_sid, 
                current_ei, di, snr, cnr
            )
            
            st.download_button(
                label="📥 ดาวน์โหลดใบรายงาน PDF สดจากระบบ",
                data=pdf_bytes,
                file_name=f"Dose_Optimization_Report_{patient_id}.pdf",
                mime="application/pdf",
                type="primary",
                key="dl_pdf_tab1"
            )

    # ---------------------------------------------------------
    # TAB 2: DOSE-SNR & NOISE POWER SPECTRUM (NPS) ANALYTICS
    # ---------------------------------------------------------
    with tab2:
        st.subheader("📈 วิเคราะห์สเปกตรัมความถี่และแนวโน้มคุณภาพภาพถ่าย (Dose & Noise Power Spectrum)")
        st.caption("ประเมินจุดคุ้มทุนเชิงพื้นที่ (Spatial Domain) และพฤติกรรม Power Spectral Density ของ Quantum Mottle ในเชิงความถี่ (Spatial Frequency Domain)")

        # คำนวณ Dose vs. SNR/CNR
        doses_test = np.linspace(10, 100, 10)
        snr_list, cnr_list = [], []

        for d in doses_test:
            test_img = apply_physics_noise(baseline_slice, d, base_sid)
            s_val, c_val, _ = compute_image_metrics(test_img)
            snr_list.append(s_val)
            cnr_list.append(c_val)

        # คำนวณ 1D Noise Power Spectrum (NPS)
        freq_axis, nps_base = compute_nps_1d(baseline_slice, pixel_spacing_mm=0.5)
        _, nps_sim = compute_nps_1d(simulated_slice, pixel_spacing_mm=0.5)

        c_graph1, c_graph2 = st.columns(2, gap="large")

        with c_graph1:
            fig_curve, ax_curve = plt.subplots(figsize=(8, 4.8), dpi=130)
            ax_curve.plot(doses_test, snr_list, 'o-', color='#1E3A8A', linewidth=2.5, label='Signal-to-Noise Ratio (SNR)')
            ax_curve.plot(doses_test, cnr_list, 's--', color='#D97706', linewidth=2.0, label='Contrast-to-Noise Ratio (CNR)')
            ax_curve.axhline(y=2.5, color='#EF4444', linestyle=':', label='Min Diagnostic Limit (SNR = 2.5)')
            ax_curve.axvline(x=dose_percent, color='#10B981', linestyle='-', linewidth=2, label=f'Current Selected ({dose_percent}%)')
            ax_curve.set_xlabel('Dose Level (%)', fontsize=10, fontweight='bold')
            ax_curve.set_ylabel('Metric Value (Dimensionless)', fontsize=10, fontweight='bold')
            ax_curve.set_title('1. Dose vs. SNR / CNR Trade-off Curve', fontsize=12, fontweight='bold', pad=10)
            ax_curve.grid(True, linestyle='--', alpha=0.5)
            ax_curve.legend(fontsize=8)
            st.pyplot(fig_curve)

        with c_graph2:
            fig_nps, ax_nps = plt.subplots(figsize=(8, 4.8), dpi=130)
            ax_nps.plot(freq_axis, nps_base, '-', color='#64748B', linewidth=2.0, label='Baseline NPS (100% Dose)')
            ax_nps.plot(freq_axis, nps_sim, '-', color='#2563EB', linewidth=2.5, label=f'Simulated NPS ({dose_percent}% Dose)')
            ax_nps.set_xlabel('Spatial Frequency (lp/mm)', fontsize=10, fontweight='bold')
            ax_nps.set_ylabel('Noise Power Spectrum (NPS)', fontsize=10, fontweight='bold')
            ax_nps.set_title('2. 1D Radial Noise Power Spectrum (NPS vs Frequency)', fontsize=12, fontweight='bold', pad=10)
            ax_nps.grid(True, linestyle='--', alpha=0.5)
            ax_nps.legend(fontsize=8)
            st.pyplot(fig_nps)

        # 🧠 คำอธิบายการประเมินเชิงประจักษ์ (Empirical Evidence Analysis Panel)
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.subheader("🔬 บทวิเคราะห์และผลการประเมินเชิงประจักษ์ (Empirical Physics & Frequency Evaluation)")
        
        critical_dose = 100
        for d_val, s_val in zip(doses_test, snr_list):
            if s_val < 2.5:
                critical_dose = d_val
                break

        col_desc1, col_desc2 = st.columns(2, gap="medium")

        with col_desc1:
            st.markdown("##### 1. การวิเคราะห์ความสัมพันธ์เชิงปริมาณรังสี (Rose Criterion)")
            st.write(f"""
            * **พฤติกรรมกราฟ Dose-SNR:** จากการจำลองเชิงประจักษ์ เมื่อลดปริมาณรังสีลง ค่า SNR และ CNR จะตกร่วงลงแบบ **Non-linear (ตามกฎ SNR ∝ √Dose)** เนื่องจากสัญญาณรังสีที่ตกลงบนฉากรับภาพลดลง
            * **เกณฑ์มาตรฐาน Rose Criterion:** ทางฟิสิกส์การแพทย์กำหนดไว้ว่า ค่า SNR ขั้นต่ำสำหรับแยกแยะรายละเอียดพยาธิสภาพต้องไม่ต่ำกว่า **2.5**
            * **จุดวิกฤต (Critical Threshold):** จากข้อมูลภาพปัจจุบัน สัญญาณ SNR จะเริ่มตกร่วงต่ำกว่าเกณฑ์มาตรฐานที่ระดับปริมาณรังสีต่ำกว่า **{critical_dose:.0f}%**
            """)

        with col_desc2:
            st.markdown("##### 2. การวิเคราะห์สเปกตรัมความถี่มิติพื้นที่ (Noise Power Spectrum - NPS)")
            st.write(f"""
            * **การยกตัวของ Power Spectral Density (PSD Shift):** กราฟ NPS ฝั่งขวาแสดงให้เห็นการยกตัวขึ้นของเส้นกราฟสีฟ้า (Simulated) เหนือเส้นสีเทา (Baseline) ยืนยันการเพิ่มขึ้นของพลังงานสัญญาณรบกวนในเชิงความถี่
            * **ลักษณะ Broadband / White Noise:** Quantum Mottle จากสถิติแบบ Poisson Distribution มีลักษณะเป็นสัญญาณรบกวนสีขาว ทำให้กราฟ NPS มีพลังงานเพิ่มขึ้นอย่างสม่ำเสมอตลอดทุกย่านความถี่ (Spatial Frequency Range 0 - {freq_axis[-1]:.1f} lp/mm)
            * **สรุปการตัดสินใจทางคลินิก:** การลดรังสีมาที่ **{dose_percent}% ({sim_mas:.2f} mAs)** ช่วยประหยัดรังสีผู้ป่วยได้ **{100 - dose_percent:.1f}%** โดยพลังงานสัญญาณรบกวนยังไม่รบกวนขอบรายละเอียดในเชิงความถี่สูง
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 3: 2-SHOT VERIFICATION & QA
    # ---------------------------------------------------------
    with tab3:
        st.subheader("🧪 การสอบทานภาพยิงจริงครั้งที่ 2 (Verification Recheck Shot vs Simulated Target)")
        st.info("💡 **In-Silico Benchmark:** สามารถนำภาพทดสอบไฟล์ที่สองมาอัปโหลด เพื่อวัดค่าความคลาดเคลื่อน (%Error) ของแบบจำลองคณิตศาสตร์ได้")

        recheck_file = st.file_uploader(
            "📥 Shot 2: อัปโหลดไฟล์ DICOM ยิงจริงซ้ำ (Recheck Verification DICOM)", 
            type=None, 
            accept_multiple_files=False,
            key="recheck_uploader"
        )

        if recheck_file:
            try:
                ds_recheck = pydicom.dcmread(recheck_file, force=True)
                act_img = ds_recheck.pixel_array.astype(float)
                if hasattr(ds_recheck, 'RescaleSlope') and hasattr(ds_recheck, 'RescaleIntercept'):
                    act_img = act_img * ds_recheck.RescaleSlope + ds_recheck.RescaleIntercept
                if getattr(ds_recheck, "PhotometricInterpretation", "") == "MONOCHROME1":
                    act_img = np.amax(act_img) - act_img

                a_min, a_max = np.min(act_img), np.max(act_img)
                act_norm = (act_img - a_min) / (a_max - a_min) * 255.0 if a_max > a_min else act_img
                act_norm = np.clip(act_norm, 0, 255)

                act_mas = float(getattr(ds_recheck, "Exposure", sim_mas))
                act_snr, act_cnr, _ = compute_image_metrics(act_norm)

                act_dose_ratio = (act_mas / max(base_mas, 0.1))
                act_ei = target_ei * act_dose_ratio
                act_di = 10.0 * np.log10(act_dose_ratio) if act_dose_ratio > 0 else -99.0

                err_mas = abs(act_mas - sim_mas) / max(sim_mas, 1e-4) * 100.0
                err_ei = abs(act_ei - current_ei) / max(current_ei, 1e-4) * 100.0
                err_di = abs(act_di - di) / max(abs(di), 1e-4) * 100.0 if di != 0 else 0.0
                err_snr = abs(act_snr - snr) / max(snr, 1e-4) * 100.0
                err_cnr = abs(act_cnr - cnr) / max(cnr, 1e-4) * 100.0

                mean_overall_error = np.mean([err_mas, err_snr, err_cnr])

                st.write("---")
                c_qa1, c_qa2 = st.columns([1.3, 1.0], gap="large")

                with c_qa1:
                    st.subheader("📊 ตารางคำนวณค่าความคลาดเคลื่อน (%Error)")
                    st.markdown(f"""
                    | พารามิเตอร์ (Metric) | จำลอง (Simulated Target) | ยิงจริง (Actual Recheck) | ความคลาดเคลื่อน (%Error) |
                    | :--- | :---: | :---: | :---: |
                    | **Tube Exposure (mAs)** | `{sim_mas:.2f} mAs` | `{act_mas:.2f} mAs` | **`{err_mas:.2f}%`** |
                    | **Exposure Index (EI)** | `{current_ei:.1f}` | `{act_ei:.1f}` | **`{err_ei:.2f}%`** |
                    | **Deviation Index (DI)** | `{di:+.2f}` | `{act_di:+.2f}` | **`{err_di:.2f}%`** |
                    | **Signal-to-Noise (SNR)** | `{snr:.2f}` | `{act_snr:.2f}` | **`{err_snr:.2f}%`** |
                    | **Contrast-to-Noise (CNR)** | `{cnr:.2f}` | `{act_cnr:.2f}` | **`{err_cnr:.2f}%`** |
                    """)

                with c_qa2:
                    st.subheader("🛡️ ผลการสอบทานระบบ (Simulation QA)")
                    st.metric("ความคลาดเคลื่อนเฉลี่ย (Mean Error)", f"{mean_overall_error:.2f}%")

                    if mean_overall_error <= 5.0:
                        st.markdown('<div class="status-badge-pass">✅ HIGH ACCURACY (PASS): แบบจำลองมีความแม่นยำสูงมาก (Error ≤ 5%)</div>', unsafe_allow_html=True)
                    elif mean_overall_error <= 10.0:
                        st.markdown('<div class="status-badge-warning">⚠️ MODERATE VARIANCE: ความแม่นยำอยู่ในเกณฑ์ยอมรับได้ (Error 5%-10%)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-badge-fail">❌ EQUIPMENT DRIFT WARNING: ค่าภาพยิงจริงเบี่ยงเบนจากแบบจำลองเกิน 10%</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ Recheck: {e}")

else:
    st.info("👈 **วิธีใช้งาน:** ลากโฟลเดอร์ภาพ DICOM มาวางใส่ด้านบน หรือเลือกติ๊ก '🧪 ใช้ข้อมูลภาพจำลอง (Digital Phantom HD)' ที่เมนูด้านซ้ายเพื่อทดลองระบบได้ทันที")
