import io
import re
import inspect
import datetime
import streamlit as st
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager
from scipy.ndimage import gaussian_filter

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    HAS_PLOTLY_SELECT = "on_select" in inspect.signature(st.plotly_chart).parameters
except ImportError:
    HAS_PLOTLY = False
    HAS_PLOTLY_SELECT = False

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    from PIL import Image, ImageDraw
    HAS_IMG_COORDS = True
except ImportError:
    HAS_IMG_COORDS = False

st.set_page_config(page_title="3D DICOM Series & Virtual Lab Workbench", layout="wide")

# ==============================================================================
# ฐานข้อมูลคุณสมบัติทางฟิสิกส์ของวัสดุมาตรฐาน (อ้างอิงที่พลังงานอ้างอิง 120 kVp)
# ==============================================================================
MATERIALS_DB = {
    "Soft Tissue": {"density": 1.06, "mu_ref": 0.215, "zeff": 7.40, "desc": "เนื้อเยื่ออ่อนและกล้ามเนื้อทั่วไป"},
    "Water": {"density": 1.00, "mu_ref": 0.206, "zeff": 7.42, "desc": "ตัวกลางอ้างอิงมาตรฐานสากล (Dosimetry)"},
    "PMMA": {"density": 1.19, "mu_ref": 0.229, "zeff": 6.47, "desc": "แผ่นอะคริลิกมาตรฐานสำหรับ Slab / QA Phantom"},
    "Adipose (Fat)": {"density": 0.92, "mu_ref": 0.184, "zeff": 6.30, "desc": "เนื้อเยื่อไขมัน (ทดสอบ Low-Contrast)"},
    "HDPE": {"density": 0.95, "mu_ref": 0.182, "zeff": 5.45, "desc": "โพลีเอทิลีนความหนาแน่นสูง (ACR Phantom)"},
    "Lung": {"density": 0.30, "mu_ref": 0.062, "zeff": 7.40, "desc": "เนื้อเยื่อปอดที่มีอากาศแทรกอยู่"},
    "Trabecular Bone": {"density": 1.18, "mu_ref": 0.265, "zeff": 10.50, "desc": "กระดูกพรุน / กระดูกสันหลังส่วนใน"},
    "Cortical Bone": {"density": 1.85, "mu_ref": 0.420, "zeff": 13.80, "desc": "กระดูกทึบ (ซี่โครง กะโหลก กระดูกแข็ง)"},
    "PTFE (Teflon)": {"density": 2.20, "mu_ref": 0.440, "zeff": 8.43, "desc": "เทฟลอนมาตรฐานจำลองกระดูกใน ACR Phantom"},
    "Iodine (15 mg/mL)": {"density": 1.02, "mu_ref": 0.315, "zeff": 11.50, "desc": "สารทึบรังสีไอโอดีน (ไวต่อการเปลี่ยน kVp)"},
    "Aluminum": {"density": 2.70, "mu_ref": 0.680, "zeff": 13.00, "desc": "อะลูมิเนียมมาตรฐานสำหรับ HVL / Step Wedge"},
    "Air": {"density": 0.0012, "mu_ref": 0.0002, "zeff": 7.64, "desc": "ช่องอากาศ / หลอดลม / แก๊สในลำไส้"},
}

PHANTOM_PRESETS = {
    "ชุดที่ 1: Standard Body Phantom (ค่าเริ่มต้นตรวจสอบหลัก — 3 วัสดุ)": {
        "num_mat": 3,
        "bg": "Soft Tissue",
        "ins1": "Cortical Bone",
        "ins2": "Air",
        "thick": 15.0,
        "kvp": 100.0,
        "ma": 250,
        "ms": 450,
        "sid": 100.0,
        "grid": True,
        "desc": "ตัวกลาง Soft Tissue 15 cm พร้อม Cortical Bone (ซ้าย) และ Air (ขวา) — ชุดเริ่มต้นหลักสำหรับพิสูจน์สมการฟิสิกส์และเทียบค่า ImageJ",
    },
    "ชุดที่ 2: Chest / Thorax Phantom (จำลองทรวงอกและปอด — 3 วัสดุ)": {
        "num_mat": 3,
        "bg": "Soft Tissue",
        "ins1": "Cortical Bone",
        "ins2": "Lung",
        "thick": 18.0,
        "kvp": 120.0,
        "ma": 400,
        "ms": 800,
        "sid": 180.0,
        "grid": True,
        "desc": "ตัวกลาง Soft Tissue 18 cm พร้อม Cortical Bone (ซี่โครง) และ Lung (เนื้อปอด) ที่ระยะ SID 180 cm, 120 kVp — ทดสอบการแยกปอดกับกระดูก",
    },
    "ชุดที่ 3: Abdomen / Spine Phantom (จำลองช่องท้องและกระดูกสันหลัง — 3 วัสดุ)": {
        "num_mat": 3,
        "bg": "Soft Tissue",
        "ins1": "Trabecular Bone",
        "ins2": "Air",
        "thick": 20.0,
        "kvp": 100.0,
        "ma": 400,
        "ms": 890,
        "sid": 100.0,
        "grid": True,
        "desc": "ตัวกลาง Soft Tissue 20 cm พร้อม Trabecular Bone (กระดูกสันหลัง) และ Air (แก๊สในลำไส้) — ทดสอบความหนาอวัยวะมากและรังสีกระเจิง",
    },
    "ชุดที่ 4: Physical QA Slab Phantom (หุ่นทดสอบมาตรฐาน PMMA/PTFE/HDPE — 3 วัสดุ)": {
        "num_mat": 3,
        "bg": "PMMA",
        "ins1": "PTFE (Teflon)",
        "ins2": "HDPE",
        "thick": 15.0,
        "kvp": 100.0,
        "ma": 280,
        "ms": 500,
        "sid": 100.0,
        "grid": True,
        "desc": "แผ่นอะคริลิก PMMA 15 cm พร้อมแท่งเทฟลอน PTFE (แทนกระดูก) และ HDPE (แทนไขมัน) — ตรงตามวัสดุหุ่นทดสอบมาตรฐานโรงพยาบาล (ACR QA)",
    },
    "ชุดที่ 5: Contrast Media & Low-Contrast (จำลองไอโอดีนและไขมัน — 3 วัสดุ)": {
        "num_mat": 3,
        "bg": "Water",
        "ins1": "Iodine (15 mg/mL)",
        "ins2": "Adipose (Fat)",
        "thick": 15.0,
        "kvp": 80.0,
        "ma": 310,
        "ms": 800,
        "sid": 100.0,
        "grid": True,
        "desc": "ตัวกลาง Water 15 cm พร้อมสารทึบรังสี Iodine (ซ้าย) และ Adipose ไขมัน (ขวา) — ใช้พิสูจน์กฎ 15% kVp และขีดจำกัด Low-Contrast",
    },
    "ชุดที่ 6: Single-Insert Beer-Lambert (แฟนธอม 2 วัสดุมาตรฐานแบบเดิม)": {
        "num_mat": 2,
        "bg": "Water",
        "ins1": "Cortical Bone",
        "ins2": "Air",
        "thick": 15.0,
        "kvp": 100.0,
        "ma": 240,
        "ms": 400,
        "sid": 100.0,
        "grid": True,
        "desc": "ตัวกลาง Water 15 cm พร้อมวัตถุทดสอบ Cortical Bone 1 ชิ้นตรงกึ่งกลางภาพ — ใช้ทดสอบกฎ Beer-Lambert และเทียบค่า CNR 1 ต่อ 1 กับ ImageJ",
    },
    "กำหนดเอง (Custom Phantom & Protocol)": None,
}


def get_linear_attenuation(mat_name: str, kvp: float) -> float:
    """คำนวณสัมประสิทธิ์การลดทอนเชิงเส้นตามพลังงานรังสี (แยกผล Photoelectric vs Compton)"""
    info = MATERIALS_DB[mat_name]
    exp_p = 0.95 if info["zeff"] > 9.0 else 0.55
    return info["mu_ref"] * ((120.0 / max(kvp, 20.0)) ** exp_p)


# ==============================================================================
# ระบบจัดการฟอนต์และตัดข้อความให้พอดีกรอบกระดาษ A4 (แก้กล่องสี่เหลี่ยม & ล้นขอบ)
# ==============================================================================
def _get_report_font_props():
    thai_candidates = [
        "Sarabun",
        "TH Sarabun New",
        "Leelawadee UI",
        "Tahoma",
        "Thonburi",
        "Ayuthaya",
        "Garuda",
        "Loma",
        "Waree",
        "Noto Sans Thai",
    ]
    by_name = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    thai_fp = None
    for name in thai_candidates:
        if name in by_name:
            thai_fp = font_manager.FontProperties(fname=by_name[name])
            break
    latin_fp = font_manager.FontProperties(family="DejaVu Sans")
    return latin_fp, thai_fp


LATIN_FP, THAI_FP = _get_report_font_props()


def _measure_mixed_width_mm(ax, renderer, inv, text: str, fontsize: float, fontweight: str = "normal") -> float:
    if not text:
        return 0.0
    tokens = re.findall(r"[\u0e00-\u0e7f]+|[^\u0e00-\u0e7f]+", text)
    total_px = 0.0
    for tok in tokens:
        use_thai = bool(re.match(r"[\u0e00-\u0e7f]", tok)) and (THAI_FP is not None)
        fp = (THAI_FP if use_thai else LATIN_FP).copy()
        fp.set_size(fontsize)
        fp.set_weight(fontweight)
        w_px, _, _ = renderer.get_text_width_height_descent(tok, fp, ismath=False)
        total_px += w_px
    (x0_mm, _), (x1_mm, _) = inv.transform([(0.0, 0.0), (total_px, 0.0)])
    return max(x1_mm - x0_mm, 0.0)


def _wrap_to_mm(
    ax,
    renderer,
    inv,
    text: str,
    max_width_mm: float,
    fontsize: float,
    fontweight: str = "normal",
    max_lines: int = 2,
) -> str:
    s = str(text).strip()
    if not s:
        return ""
    if _measure_mixed_width_mm(ax, renderer, inv, s, fontsize, fontweight) <= max_width_mm:
        return s

    words = s.split(" ")
    lines = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if _measure_mixed_width_mm(ax, renderer, inv, cand, fontsize, fontweight) <= max_width_mm:
            cur = cand
        else:
            if cur:
                lines.append(cur)
                cur = ""
            if _measure_mixed_width_mm(ax, renderer, inv, w, fontsize, fontweight) > max_width_mm:
                chunk = ""
                for ch in w:
                    if _measure_mixed_width_mm(ax, renderer, inv, chunk + ch, fontsize, fontweight) <= max_width_mm:
                        chunk += ch
                    else:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                cur = chunk
            else:
                cur = w
    if cur:
        lines.append(cur)

    if len(lines) > max_lines:
        kept = lines[:max_lines]
        last = kept[-1]
        while len(last) > 1 and _measure_mixed_width_mm(ax, renderer, inv, last + "...", fontsize, fontweight) > max_width_mm:
            last = last[:-1].rstrip()
        kept[-1] = last + "..."
        return "\n".join(kept)
    return "\n".join(lines)


def _draw_text_mm(
    ax,
    renderer,
    inv,
    x: float,
    y: float,
    text: str,
    fontsize: float = 6.5,
    fontweight: str = "normal",
    color: str = "#102A43",
    ha: str = "left",
    va: str = "center",
    linespacing: float = 1.2,
):
    s = str(text)
    if not s:
        return
    has_thai = bool(re.search(r"[\u0e00-\u0e7f]", s))
    if not has_thai:
        fp = LATIN_FP.copy()
        fp.set_size(fontsize)
        fp.set_weight(fontweight)
        ax.text(
            x,
            y,
            s,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha=ha,
            va=va,
            linespacing=linespacing,
            fontproperties=fp,
        )
        return

    if THAI_FP is None:
        clean_s = re.sub(r"[\u0e00-\u0e7f]+", "[TH]", s)
        fp = LATIN_FP.copy()
        fp.set_size(fontsize)
        fp.set_weight(fontweight)
        ax.text(
            x,
            y,
            clean_s,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha=ha,
            va=va,
            linespacing=linespacing,
            fontproperties=fp,
        )
        return

    lines = s.split("\n")
    line_step_mm = fontsize * 0.3527 * linespacing
    if va == "top":
        start_y = y
    elif va == "bottom":
        start_y = y + (len(lines) - 1) * line_step_mm
    else:
        start_y = y + (len(lines) - 1) * line_step_mm * 0.5

    for l_idx, line in enumerate(lines):
        cur_y = start_y - l_idx * line_step_mm
        tokens = re.findall(r"[\u0e00-\u0e7f]+|[^\u0e00-\u0e7f]+", line)
        seg_widths = [_measure_mixed_width_mm(ax, renderer, inv, tok, fontsize, fontweight) for tok in tokens]
        total_w = sum(seg_widths)
        if ha == "center":
            cur_x = x - total_w / 2.0
        elif ha == "right":
            cur_x = x - total_w
        else:
            cur_x = x

        for tok, w_mm in zip(tokens, seg_widths):
            use_thai = bool(re.match(r"[\u0e00-\u0e7f]", tok))
            fp = (THAI_FP if use_thai else LATIN_FP).copy()
            fp.set_size(fontsize)
            fp.set_weight(fontweight)
            ax.text(
                cur_x,
                cur_y,
                tok,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                ha="left",
                va=va,
                fontproperties=fp,
            )
            cur_x += w_mm


# ==============================================================================
# ระบบจัดการ Key สำหรับรีเซ็ต File Uploader และค่าจำลอง
# ==============================================================================
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def reset_app():
    st.session_state.uploader_key += 1
    prefixes_to_clear = [
        "s_sid",
        "s_kvp",
        "s_mas",
        "s_ang",
        "last_dose_sync",
        "last_base_sig",
        "post_qa_plan",
        "m2_last_preset",
    ]
    for k in list(st.session_state.keys()):
        if k in prefixes_to_clear or k.startswith(("m1_roi_", "m2_roi_", "m3_roi_", "m2_cfg_")):
            del st.session_state[k]
    st.rerun()


# ==============================================================================
# ระบบควบคุมพิกัด ROI แบบอิสระ (เลือกสีกรอบ + คลิกจุดบนภาพ + แสดงพิกัดทันที)
# ==============================================================================
def _get_roi_bounds(w: int, h: int, roi_size: int):
    eff_roi = int(min(max(roi_size, 2), h, w))
    half_floor = eff_roi // 2
    half_ceil = eff_roi - half_floor
    min_x = half_floor
    max_x = max(w - half_ceil, half_floor)
    min_y = half_floor
    max_y = max(h - half_ceil, half_floor)
    return eff_roi, min_x, max_x, min_y, max_y


def ensure_roi_state(
    prefix: str,
    w: int,
    h: int,
    roi_size: int,
    default_bg_offset: int,
    init_coords: tuple = None,
    has_third_roi: bool = False,
):
    """เตรียมและล็อกขอบเขตค่าพิกัด ROI ใน Session State"""
    _, min_x, max_x, min_y, max_y = _get_roi_bounds(w, h, roi_size)
    shape_sig = (w, h, has_third_roi)
    prev_shape = st.session_state.get(f"{prefix}_shape")

    if init_coords is not None and st.session_state.get(f"{prefix}_init_tag") != init_coords:
        st.session_state[f"{prefix}_init_tag"] = init_coords
        st.session_state[f"{prefix}_shape"] = shape_sig
        st.session_state[f"{prefix}_sig_cx"] = int(np.clip(init_coords[0], min_x, max_x))
        st.session_state[f"{prefix}_sig_cy"] = int(np.clip(init_coords[1], min_y, max_y))
        st.session_state[f"{prefix}_bg_cx"] = int(np.clip(init_coords[2], min_x, max_x))
        st.session_state[f"{prefix}_bg_cy"] = int(np.clip(init_coords[3], min_y, max_y))
        if has_third_roi and len(init_coords) >= 6:
            st.session_state[f"{prefix}_ins2_cx"] = int(np.clip(init_coords[4], min_x, max_x))
            st.session_state[f"{prefix}_ins2_cy"] = int(np.clip(init_coords[5], min_y, max_y))
        return

    if prev_shape != shape_sig or f"{prefix}_sig_cx" not in st.session_state:
        st.session_state[f"{prefix}_shape"] = shape_sig
        if has_third_roi:
            st.session_state[f"{prefix}_sig_cx"] = int(np.clip(round(w * 0.27), min_x, max_x))
            st.session_state[f"{prefix}_sig_cy"] = int(np.clip(h // 2, min_y, max_y))
            st.session_state[f"{prefix}_bg_cx"] = int(np.clip(w // 2, min_x, max_x))
            st.session_state[f"{prefix}_bg_cy"] = int(np.clip(h // 2, min_y, max_y))
            st.session_state[f"{prefix}_ins2_cx"] = int(np.clip(round(w * 0.73), min_x, max_x))
            st.session_state[f"{prefix}_ins2_cy"] = int(np.clip(h // 2, min_y, max_y))
        else:
            def_sig_x = int(np.clip(w // 2, min_x, max_x))
            def_sig_y = int(np.clip(h // 2, min_y, max_y))
            def_bg_x = int(np.clip(def_sig_x + default_bg_offset, min_x, max_x))
            st.session_state[f"{prefix}_sig_cx"] = def_sig_x
            st.session_state[f"{prefix}_sig_cy"] = def_sig_y
            st.session_state[f"{prefix}_bg_cx"] = def_bg_x
            st.session_state[f"{prefix}_bg_cy"] = def_sig_y
    else:
        st.session_state[f"{prefix}_sig_cx"] = int(np.clip(st.session_state[f"{prefix}_sig_cx"], min_x, max_x))
        st.session_state[f"{prefix}_sig_cy"] = int(np.clip(st.session_state[f"{prefix}_sig_cy"], min_y, max_y))
        st.session_state[f"{prefix}_bg_cx"] = int(np.clip(st.session_state[f"{prefix}_bg_cx"], min_x, max_x))
        st.session_state[f"{prefix}_bg_cy"] = int(np.clip(st.session_state[f"{prefix}_bg_cy"], min_y, max_y))
        if has_third_roi:
            st.session_state[f"{prefix}_ins2_cx"] = int(
                np.clip(st.session_state.get(f"{prefix}_ins2_cx", round(w * 0.73)), min_x, max_x)
            )
            st.session_state[f"{prefix}_ins2_cy"] = int(
                np.clip(st.session_state.get(f"{prefix}_ins2_cy", h // 2), min_y, max_y)
            )


def _clear_last_click(prefix: str):
    st.session_state[f"{prefix}_last_click"] = None


def _apply_clicked_point(prefix: str, target_mode: str, click_x: float, click_y: float, w: int, h: int, roi_size: int):
    _, min_x, max_x, min_y, max_y = _get_roi_bounds(w, h, roi_size)
    cx = int(np.clip(round(float(click_x)), min_x, max_x))
    cy = int(np.clip(round(float(click_y)), min_y, max_y))
    if target_mode.startswith("🔴"):
        st.session_state[f"{prefix}_sig_cx"] = cx
        st.session_state[f"{prefix}_sig_cy"] = cy
    elif target_mode.startswith("🔵"):
        st.session_state[f"{prefix}_bg_cx"] = cx
        st.session_state[f"{prefix}_bg_cy"] = cy
    else:
        st.session_state[f"{prefix}_ins2_cx"] = cx
        st.session_state[f"{prefix}_ins2_cy"] = cy
    st.session_state[f"{prefix}_last_click"] = (target_mode, cx, cy)
    st.session_state[f"{prefix}_chart_ver"] = st.session_state.get(f"{prefix}_chart_ver", 0) + 1
    st.rerun()


def render_interactive_roi_viewer(
    prefix: str,
    images: list,
    titles: list,
    roi_size: int,
    show_roi: bool = True,
    vmin: float = None,
    vmax: float = None,
    has_third_roi: bool = False,
    roi_labels: tuple = ("🔴 กรอบสีแดง (Signal ROI)", "🔵 กรอบสีฟ้า (Background ROI)", "🟢 กรอบสีเขียว (Insert 2 ROI)"),
):
    """แสดงภาพพร้อมตัวเลือกสีกรอบ ROI ที่คลิกกำหนดจุดบนภาพได้ทันที พร้อมบอกพิกัด (X, Y)"""
    h, w = images[0].shape
    sig_cx = st.session_state[f"{prefix}_sig_cx"]
    sig_cy = st.session_state[f"{prefix}_sig_cy"]
    bg_cx = st.session_state[f"{prefix}_bg_cx"]
    bg_cy = st.session_state[f"{prefix}_bg_cy"]
    ins2_cx = st.session_state.get(f"{prefix}_ins2_cx", round(w * 0.73))
    ins2_cy = st.session_state.get(f"{prefix}_ins2_cy", h // 2)

    eff_roi, _, _, _, _ = _get_roi_bounds(w, h, roi_size)
    half_floor = eff_roi // 2
    sx0, sy0 = sig_cx - half_floor, sig_cy - half_floor
    bx0, by0 = bg_cx - half_floor, bg_cy - half_floor
    i2x0, i2y0 = ins2_cx - half_floor, ins2_cy - half_floor

    radio_opts = [roi_labels[0], roi_labels[1]] + ([roi_labels[2]] if has_third_roi else [])
    target_mode = st.radio(
        "🎨 1) เลือกสีกรอบที่ต้องการย้าย  →  🖱️ 2) คลิกกำหนดจุดบนรูปภาพด้านล่างได้เลย:",
        radio_opts,
        horizontal=True,
        key=f"{prefix}_click_target",
    )

    last_click = st.session_state.get(f"{prefix}_last_click")
    coord_summary = (
        f"📍 **พิกัดจุดศูนย์กลางกรอบ ROI ปัจจุบัน (ขนาด {eff_roi}×{eff_roi} px):** &nbsp; "
        f"**🔴 สีแดง:** `(X={sig_cx}, Y={sig_cy})` &nbsp;|&nbsp; "
        f"**🔵 สีฟ้า:** `(X={bg_cx}, Y={bg_cy})`"
    )
    if has_third_roi:
        coord_summary += f" &nbsp;|&nbsp; **🟢 สีเขียว:** `(X={ins2_cx}, Y={ins2_cy})`"

    if last_click is not None:
        st.success(
            f"✅ ย้าย **{last_click[0]}** ไปที่พิกัด **`X = {last_click[1]} px, Y = {last_click[2]} px`** เรียบร้อยแล้ว!\n\n{coord_summary}"
        )
    else:
        st.caption(coord_summary)

    chart_ver = st.session_state.get(f"{prefix}_chart_ver", 0)

    if HAS_IMG_COORDS:
        cols = st.columns(len(images))
        for idx, (col, img, title) in enumerate(zip(cols, images, titles)):
            with col:
                st.caption(f"**{title}**")
                ih, iw = img.shape
                zmin = float(np.min(img)) if vmin is None else vmin
                zmax = float(np.max(img)) if vmax is None else vmax
                norm = np.clip((img - zmin) / max(zmax - zmin, 1e-6) * 255.0, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(norm).convert("RGB")
                if show_roi:
                    draw = ImageDraw.Draw(pil_img)
                    draw.rectangle([sx0, sy0, sx0 + eff_roi - 1, sy0 + eff_roi - 1], outline=(255, 75, 75), width=2)
                    draw.text((max(sx0, 2), max(sy0 - 12, 2)), f"({sig_cx},{sig_cy})", fill=(255, 95, 95))
                    draw.rectangle([bx0, by0, bx0 + eff_roi - 1, by0 + eff_roi - 1], outline=(0, 212, 255), width=2)
                    draw.text((max(bx0, 2), max(by0 - 12, 2)), f"({bg_cx},{bg_cy})", fill=(56, 189, 248))
                    if has_third_roi:
                        draw.rectangle([i2x0, i2y0, i2x0 + eff_roi - 1, i2y0 + eff_roi - 1], outline=(16, 185, 129), width=2)
                        draw.text((max(i2x0, 2), max(i2y0 - 12, 2)), f"({ins2_cx},{ins2_cy})", fill=(52, 211, 153))
                disp_w = 420 if len(images) == 1 else (340 if len(images) == 2 else 250)
                scale = disp_w / float(iw)
                coords = streamlit_image_coordinates(pil_img, width=disp_w, key=f"{prefix}_imgcoord_{idx}_{chart_ver}")
                if coords is not None:
                    _apply_clicked_point(prefix, target_mode, coords["x"] / scale, coords["y"] / scale, w, h, roi_size)

    elif HAS_PLOTLY and HAS_PLOTLY_SELECT:
        n_imgs = len(images)
        fig = make_subplots(
            rows=1,
            cols=n_imgs,
            subplot_titles=titles,
            horizontal_spacing=0.03 if n_imgs > 1 else 0.01,
        )

        # กริดจุดรับการคลิกขนาดกำลังดี (64x64) เพื่อให้คลิกแล้วตอบสนองทันที
        nx = min(w, 64)
        ny = min(h, 64)
        gx = np.unique(np.round(np.linspace(0, w - 1, nx)).astype(int))
        gy = np.unique(np.round(np.linspace(0, h - 1, ny)).astype(int))
        grid_x, grid_y = np.meshgrid(gx, gy)
        flat_x = grid_x.ravel()
        flat_y = grid_y.ravel()

        for idx, img in enumerate(images, start=1):
            ih, iw = img.shape
            zmin = float(np.min(img)) if vmin is None else vmin
            zmax = float(np.max(img)) if vmax is None else vmax
            if zmax <= zmin:
                zmax = zmin + 1.0

            gx_i = np.clip(gx, 0, iw - 1)
            gy_i = np.clip(gy, 0, ih - 1)
            flat_val = img[np.ix_(gy_i, gx_i)].ravel()

            fig.add_trace(
                go.Heatmap(
                    z=img,
                    colorscale="gray",
                    zmin=zmin,
                    zmax=zmax,
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=idx,
            )

            # เลเยอร์จุดโปร่งใสที่เปิด dragmode="select" + clickmode="event+select" เพื่อรับการคลิกทันที
            fig.add_trace(
                go.Scatter(
                    x=flat_x,
                    y=flat_y,
                    customdata=flat_val,
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="rgba(255, 255, 255, 0.02)",
                        line=dict(width=0),
                    ),
                    hovertemplate=(
                        f"<b>คลิกเพื่อย้าย {target_mode}</b><br>"
                        "พิกัด X: <b>%{x} px</b> | Y: <b>%{y} px</b><br>"
                        "ค่าความเข้มแสง: %{customdata:.1f} ADU<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=idx,
            )

            if show_roi:
                xref_str = f"x{idx}" if idx > 1 else "x"
                yref_str = f"y{idx}" if idx > 1 else "y"

                # 1. กรอบสีแดง (Signal / Insert 1)
                fig.add_shape(
                    type="rect",
                    x0=sx0 - 0.5,
                    y0=sy0 - 0.5,
                    x1=sx0 + eff_roi - 0.5,
                    y1=sy0 + eff_roi - 0.5,
                    line=dict(color="#FF4B4B", width=2.4),
                    xref=xref_str,
                    yref=yref_str,
                )
                fig.add_annotation(
                    x=sig_cx,
                    y=max(sy0 - 3, 4),
                    text=f"🔴 ({sig_cx},{sig_cy})",
                    showarrow=False,
                    font=dict(color="#FF6B6B", size=10),
                    bgcolor="rgba(15, 23, 42, 0.80)",
                    bordercolor="#FF4B4B",
                    borderwidth=1,
                    borderpad=2,
                    xref=xref_str,
                    yref=yref_str,
                )

                # 2. กรอบสีฟ้า (Background)
                fig.add_shape(
                    type="rect",
                    x0=bx0 - 0.5,
                    y0=by0 - 0.5,
                    x1=bx0 + eff_roi - 0.5,
                    y1=by0 + eff_roi - 0.5,
                    line=dict(color="#00D4FF", width=2.4),
                    xref=xref_str,
                    yref=yref_str,
                )
                fig.add_annotation(
                    x=bg_cx,
                    y=min(by0 + eff_roi + 10, ih - 5) if abs(bg_cy - sig_cy) < 18 and abs(bg_cx - sig_cx) < 45 else max(by0 - 3, 4),
                    text=f"🔵 ({bg_cx},{bg_cy})",
                    showarrow=False,
                    font=dict(color="#38BDF8", size=10),
                    bgcolor="rgba(15, 23, 42, 0.80)",
                    bordercolor="#00D4FF",
                    borderwidth=1,
                    borderpad=2,
                    xref=xref_str,
                    yref=yref_str,
                )

                # 3. กรอบสีเขียว (Insert 2 - โหมด 3 วัสดุ)
                if has_third_roi:
                    fig.add_shape(
                        type="rect",
                        x0=i2x0 - 0.5,
                        y0=i2y0 - 0.5,
                        x1=i2x0 + eff_roi - 0.5,
                        y1=i2y0 + eff_roi - 0.5,
                        line=dict(color="#10B981", width=2.4),
                        xref=xref_str,
                        yref=yref_str,
                    )
                    fig.add_annotation(
                        x=ins2_cx,
                        y=max(i2y0 - 3, 4),
                        text=f"🟢 ({ins2_cx},{ins2_cy})",
                        showarrow=False,
                        font=dict(color="#34D399", size=10),
                        bgcolor="rgba(15, 23, 42, 0.80)",
                        bordercolor="#10B981",
                        borderwidth=1,
                        borderpad=2,
                        xref=xref_str,
                        yref=yref_str,
                    )

            fig.update_xaxes(
                range=[-0.5, iw - 0.5],
                showgrid=False,
                zeroline=False,
                constrain="domain",
                row=1,
                col=idx,
            )
            fig.update_yaxes(
                range=[ih - 0.5, -0.5],
                showgrid=False,
                zeroline=False,
                scaleanchor=f"x{idx}" if idx > 1 else "x",
                scaleratio=1,
                row=1,
                col=idx,
            )

        fig.update_layout(
            height=440 if n_imgs >= 2 else 460,
            margin=dict(l=10, r=10, t=38, b=10),
            hovermode="closest",
            hoverdistance=40,
            clickmode="event+select",
            dragmode="select",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            config={"displayModeBar": False, "scrollZoom": False},
            key=f"{prefix}_plotly_{chart_ver}",
        )
        if event is not None:
            sel = event.get("selection", {}) if isinstance(event, dict) else getattr(event, "selection", None)
            pts = sel.get("points", []) if isinstance(sel, dict) else (getattr(sel, "points", []) if sel else [])
            if pts:
                pt = pts[0]
                px_val = pt.get("x") if isinstance(pt, dict) else getattr(pt, "x", None)
                py_val = pt.get("y") if isinstance(pt, dict) else getattr(pt, "y", None)
                if px_val is not None and py_val is not None:
                    _apply_clicked_point(prefix, target_mode, px_val, py_val, w, h, roi_size)

    else:
        fig, axes = plt.subplots(1, len(images), figsize=(5.2 * len(images), 4.8))
        if len(images) == 1:
            axes = [axes]
        for ax_i, img_i, title_i in zip(axes, images, titles):
            ax_i.imshow(img_i, cmap="gray", vmin=vmin, vmax=vmax)
            ax_i.set_title(title_i, fontsize=10)
            ax_i.axis("off")
            if show_roi:
                ax_i.add_patch(
                    patches.Rectangle((sx0, sy0), eff_roi, eff_roi, linewidth=1.6, edgecolor="#FF4B4B", facecolor="none")
                )
                ax_i.add_patch(
                    patches.Rectangle((bx0, by0), eff_roi, eff_roi, linewidth=1.6, edgecolor="#00D4FF", facecolor="none")
                )
                if has_third_roi:
                    ax_i.add_patch(
                        patches.Rectangle((i2x0, i2y0), eff_roi, eff_roi, linewidth=1.6, edgecolor="#10B981", facecolor="none")
                    )
        st.pyplot(fig)
        plt.close(fig)


def render_roi_coordinate_inputs(
    prefix: str,
    w: int,
    h: int,
    roi_size: int,
    default_bg_offset: int,
    has_third_roi: bool = False,
    labels: tuple = ("🔴 กรอบสีแดง (Signal ROI)", "🔵 กรอบสีฟ้า (Background ROI)", "🟢 กรอบสีเขียว (Insert 2 ROI)"),
):
    """แสดงช่องกรอกเลขพิกัด X, Y แบบอิสระ"""
    _, min_x, max_x, min_y, max_y = _get_roi_bounds(w, h, roi_size)

    if has_third_roi:
        c_sig, c_bg, c_ins2, c_rst = st.columns([2.0, 2.0, 2.0, 1.0])
    else:
        c_sig, c_bg, c_rst = st.columns([2.2, 2.2, 1.1])
        c_ins2 = None

    with c_sig:
        st.markdown(f"**{labels[0]}**")
        sc1, sc2 = st.columns(2)
        sc1.number_input("X (px)", min_value=min_x, max_value=max_x, step=1, key=f"{prefix}_sig_cx", on_change=_clear_last_click, args=(prefix,))
        sc2.number_input("Y (px)", min_value=min_y, max_value=max_y, step=1, key=f"{prefix}_sig_cy", on_change=_clear_last_click, args=(prefix,))

    with c_bg:
        st.markdown(f"**{labels[1]}**")
        bc1, bc2 = st.columns(2)
        bc1.number_input("X (px)", min_value=min_x, max_value=max_x, step=1, key=f"{prefix}_bg_cx", on_change=_clear_last_click, args=(prefix,))
        bc2.number_input("Y (px)", min_value=min_y, max_value=max_y, step=1, key=f"{prefix}_bg_cy", on_change=_clear_last_click, args=(prefix,))

    if has_third_roi and c_ins2 is not None:
        with c_ins2:
            st.markdown(f"**{labels[2]}**")
            ic1, ic2 = st.columns(2)
            ic1.number_input("X (px)", min_value=min_x, max_value=max_x, step=1, key=f"{prefix}_ins2_cx", on_change=_clear_last_click, args=(prefix,))
            ic2.number_input("Y (px)", min_value=min_y, max_value=max_y, step=1, key=f"{prefix}_ins2_cy", on_change=_clear_last_click, args=(prefix,))

    with c_rst:
        st.markdown("**&nbsp;**")
        if st.button("↺ รีเซ็ต", key=f"{prefix}_reset_btn", use_container_width=True):
            if has_third_roi:
                st.session_state[f"{prefix}_sig_cx"] = int(np.clip(round(w * 0.27), min_x, max_x))
                st.session_state[f"{prefix}_sig_cy"] = int(np.clip(h // 2, min_y, max_y))
                st.session_state[f"{prefix}_bg_cx"] = int(np.clip(w // 2, min_x, max_x))
                st.session_state[f"{prefix}_bg_cy"] = int(np.clip(h // 2, min_y, max_y))
                st.session_state[f"{prefix}_ins2_cx"] = int(np.clip(round(w * 0.73), min_x, max_x))
                st.session_state[f"{prefix}_ins2_cy"] = int(np.clip(h // 2, min_y, max_y))
            else:
                def_sig_x = int(np.clip(w // 2, min_x, max_x))
                def_sig_y = int(np.clip(h // 2, min_y, max_y))
                def_bg_x = int(np.clip(def_sig_x + default_bg_offset, min_x, max_x))
                st.session_state[f"{prefix}_sig_cx"] = def_sig_x
                st.session_state[f"{prefix}_sig_cy"] = def_sig_y
                st.session_state[f"{prefix}_bg_cx"] = def_bg_x
                st.session_state[f"{prefix}_bg_cy"] = def_sig_y
            st.session_state[f"{prefix}_last_click"] = None
            st.session_state[f"{prefix}_chart_ver"] = st.session_state.get(f"{prefix}_chart_ver", 0) + 1
            st.rerun()

    if has_third_roi:
        return (
            int(st.session_state[f"{prefix}_sig_cx"]),
            int(st.session_state[f"{prefix}_sig_cy"]),
            int(st.session_state[f"{prefix}_bg_cx"]),
            int(st.session_state[f"{prefix}_bg_cy"]),
            int(st.session_state[f"{prefix}_ins2_cx"]),
            int(st.session_state[f"{prefix}_ins2_cy"]),
        )
    return (
        int(st.session_state[f"{prefix}_sig_cx"]),
        int(st.session_state[f"{prefix}_sig_cy"]),
        int(st.session_state[f"{prefix}_bg_cx"]),
        int(st.session_state[f"{prefix}_bg_cy"]),
    )


# ==============================================================================
# ฟังก์ชันช่วยอ่านไฟล์ DICOM Series และพารามิเตอร์จาก Header
# ==============================================================================
def load_dicom_series(uploaded_files):
    raw_slices = []
    for f in uploaded_files:
        if f.name.startswith(".") or f.name.startswith("__"):
            continue
        try:
            f.seek(0)
            ds = pydicom.dcmread(f, force=True)
            if "PixelData" in ds and hasattr(ds, "pixel_array"):
                raw_slices.append(ds)
        except Exception:
            pass

    if len(raw_slices) == 0:
        return None

    try:
        raw_slices.sort(
            key=lambda x: int(x.InstanceNumber)
            if hasattr(x, "InstanceNumber")
            else (float(x.SliceLocation) if hasattr(x, "SliceLocation") else 0)
        )
    except Exception:
        pass

    sample_ds = raw_slices[0]
    modality = str(getattr(sample_ds, "Modality", "Unspecified"))
    patient_id = str(getattr(sample_ds, "PatientID", "Anonymized"))
    is_rt_file = modality.startswith("RT") or "BeamSequence" in sample_ds

    auto_sid = 100.0
    if hasattr(sample_ds, "RTImageSID"):
        auto_sid = float(sample_ds.RTImageSID) / 10.0
    elif hasattr(sample_ds, "RadiationMachineSAD"):
        auto_sid = float(sample_ds.RadiationMachineSAD) / 10.0
    elif hasattr(sample_ds, "DistanceSourceToDetector"):
        auto_sid = float(sample_ds.DistanceSourceToDetector) / 10.0

    auto_angle = 0.0
    if hasattr(sample_ds, "GantryAngle"):
        auto_angle = float(sample_ds.GantryAngle)
    elif "BeamSequence" in sample_ds and len(sample_ds.BeamSequence) > 0:
        beam = sample_ds.BeamSequence[0]
        if hasattr(beam, "ControlPointSequence") and len(beam.ControlPointSequence) > 0:
            auto_angle = float(getattr(beam.ControlPointSequence[0], "GantryAngle", 0.0))

    auto_kvp = float(getattr(sample_ds, "KVP", 120.0))

    if hasattr(sample_ds, "Exposure"):
        auto_mas = float(sample_ds.Exposure)
    elif hasattr(sample_ds, "ExposureInuAs"):
        auto_mas = float(sample_ds.ExposureInuAs) / 1000.0
    elif hasattr(sample_ds, "XRayTubeCurrent") and hasattr(sample_ds, "ExposureTime"):
        auto_mas = (float(sample_ds.XRayTubeCurrent) * float(sample_ds.ExposureTime)) / 1000.0
    else:
        auto_mas = 100.0

    tag_ei = float(sample_ds.ExposureIndex) if hasattr(sample_ds, "ExposureIndex") else None
    tag_di = float(sample_ds.DeviationIndex) if hasattr(sample_ds, "DeviationIndex") else None

    slice_arrays = []
    for ds in raw_slices:
        arr = ds.pixel_array.astype(float)
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
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

    vol_min, vol_max = float(np.min(volume_3d)), float(np.max(volume_3d))
    norm_volume = (
        (volume_3d - vol_min) / (vol_max - vol_min) * 255.0
        if vol_max > vol_min
        else volume_3d.copy()
    )
    norm_volume = np.clip(norm_volume, 0, 255)

    return {
        "volume_3d": volume_3d,
        "norm_volume": norm_volume,
        "num_slices": norm_volume.shape[0],
        "modality": modality,
        "patient_id": patient_id,
        "is_rt_file": is_rt_file,
        "sid": auto_sid,
        "kvp": auto_kvp,
        "mas": auto_mas,
        "angle": auto_angle,
        "tag_ei": tag_ei,
        "tag_di": tag_di,
        "vol_min": vol_min,
        "vol_max": vol_max,
    }


# ==============================================================================
# ฟังก์ชันคำนวณร่วม: วิเคราะห์กรอบ ROI มาตรฐาน (ตรงกับสูตรของ ImageJ 100%)
# ==============================================================================
def analyze_image_qc(
    image: np.ndarray,
    roi_size: int = 30,
    bg_offset_px: int = 60,
    sig_cx: int = None,
    sig_cy: int = None,
    bg_cx: int = None,
    bg_cy: int = None,
    ins2_cx: int = None,
    ins2_cy: int = None,
):
    h, w = image.shape
    roi_size = int(min(max(roi_size, 2), h, w))
    half_floor = roi_size // 2
    half_ceil = roi_size - half_floor

    if sig_cx is None:
        sig_cx = w // 2
    if sig_cy is None:
        sig_cy = h // 2
    if bg_cx is None:
        bg_cx = sig_cx + bg_offset_px
    if bg_cy is None:
        bg_cy = sig_cy

    sig_cx = int(np.clip(round(sig_cx), half_floor, max(w - half_ceil, half_floor)))
    sig_cy = int(np.clip(round(sig_cy), half_floor, max(h - half_ceil, half_floor)))
    bg_cx = int(np.clip(round(bg_cx), half_floor, max(w - half_ceil, half_floor)))
    bg_cy = int(np.clip(round(bg_cy), half_floor, max(h - half_ceil, half_floor)))

    sy0, sy1 = sig_cy - half_floor, sig_cy - half_floor + roi_size
    sx0, sx1 = sig_cx - half_floor, sig_cx - half_floor + roi_size
    sig_roi = image[sy0:sy1, sx0:sx1]

    by0, by1 = bg_cy - half_floor, bg_cy - half_floor + roi_size
    bx0, bx1 = bg_cx - half_floor, bg_cx - half_floor + roi_size
    bg_roi = image[by0:by1, bx0:bx1]

    sig_mean = float(np.mean(sig_roi))
    sig_sd = float(np.std(sig_roi, ddof=1)) if sig_roi.size > 1 else 0.0
    bg_mean = float(np.mean(bg_roi))
    bg_sd = float(np.std(bg_roi, ddof=1)) if bg_roi.size > 1 else 0.0

    snr = sig_mean / sig_sd if sig_sd > 0 else 0.0
    cnr = abs(sig_mean - bg_mean) / bg_sd if bg_sd > 0 else 0.0
    bg_snr = bg_mean / bg_sd if bg_sd > 0 else 0.0

    roi_info = {
        "roi_size": roi_size,
        "sig_cx": sig_cx,
        "sig_cy": sig_cy,
        "bg_cx": bg_cx,
        "bg_cy": bg_cy,
        "sig_x0": sx0,
        "sig_y0": sy0,
        "bg_x0": bx0,
        "bg_y0": by0,
        "sig_roi": sig_roi,
        "bg_roi": bg_roi,
    }
    metrics = {
        "Signal Mean": sig_mean,
        "Signal SD": sig_sd,
        "Background Mean": bg_mean,
        "Background SD": bg_sd,
        "Background SNR": bg_snr,
        "SNR": snr,
        "CNR": cnr,
    }

    if ins2_cx is not None and ins2_cy is not None:
        ins2_cx = int(np.clip(round(ins2_cx), half_floor, max(w - half_ceil, half_floor)))
        ins2_cy = int(np.clip(round(ins2_cy), half_floor, max(h - half_ceil, half_floor)))
        i2y0, i2y1 = ins2_cy - half_floor, ins2_cy - half_floor + roi_size
        i2x0, i2x1 = ins2_cx - half_floor, ins2_cx - half_floor + roi_size
        ins2_roi = image[i2y0:i2y1, i2x0:i2x1]
        ins2_mean = float(np.mean(ins2_roi))
        ins2_sd = float(np.std(ins2_roi, ddof=1)) if ins2_roi.size > 1 else 0.0
        ins2_snr = ins2_mean / ins2_sd if ins2_sd > 0 else 0.0
        ins2_cnr = abs(ins2_mean - bg_mean) / bg_sd if bg_sd > 0 else 0.0

        roi_info["ins2_cx"] = ins2_cx
        roi_info["ins2_cy"] = ins2_cy
        roi_info["ins2_x0"] = i2x0
        roi_info["ins2_y0"] = i2y0
        roi_info["ins2_roi"] = ins2_roi

        metrics["Insert2 Mean"] = ins2_mean
        metrics["Insert2 SD"] = ins2_sd
        metrics["Insert2 SNR"] = ins2_snr
        metrics["Insert2 CNR"] = ins2_cnr

    return metrics, roi_info


def compute_global_image_metrics(img_2d: np.ndarray):
    smoothed_img = gaussian_filter(img_2d, sigma=2.0)
    noise_map = img_2d - smoothed_img
    std_noise = max(float(np.std(noise_map)), 1e-3)

    mean_signal = (
        float(np.mean(img_2d[img_2d > 10]))
        if np.any(img_2d > 10)
        else float(np.mean(img_2d))
    )
    snr = mean_signal / std_noise

    body_pixels = img_2d[img_2d > 20]
    if len(body_pixels) > 0:
        med_val = np.median(body_pixels)
        high_sig = float(np.mean(body_pixels[body_pixels >= med_val]))
        low_sig = float(np.mean(body_pixels[body_pixels < med_val]))
        cnr = (high_sig - low_sig) / std_noise
    else:
        cnr = 0.0

    return {
        "mean_signal": mean_signal,
        "std_noise": std_noise,
        "snr": snr,
        "cnr": cnr,
    }


# ==============================================================================
# ฟังก์ชันวินิจฉัยความคลาดเคลื่อนระหว่างผลคาดการณ์ vs ยิงจริงรอบ 2 (Hardware QA)
# ==============================================================================
def diagnose_post_exposure_qa(
    plan_kvp: float,
    plan_mas: float,
    plan_sid: float,
    act_kvp: float,
    act_mas: float,
    act_sid: float,
    pred_di: float,
    act_di: float,
    pred_snr: float,
    act_snr: float,
    pred_cnr: float,
    act_cnr: float,
    pred_noise_sd: float,
    act_noise_sd: float,
):
    delta_di = act_di - pred_di
    snr_err_pct = ((act_snr - pred_snr) / max(pred_snr, 1e-6)) * 100.0
    cnr_err_pct = ((act_cnr - pred_cnr) / max(pred_cnr, 1e-6)) * 100.0
    noise_err_pct = ((act_noise_sd - pred_noise_sd) / max(pred_noise_sd, 1e-6)) * 100.0

    kvp_diff = act_kvp - plan_kvp
    mas_err_pct = ((act_mas - plan_mas) / max(plan_mas, 1e-6)) * 100.0
    sid_diff = act_sid - plan_sid

    findings = []
    severity = "PASS"

    if abs(kvp_diff) > 2.0 or abs(mas_err_pct) > 10.0 or abs(sid_diff) > 3.0:
        severity = "WARNING"
        findings.append(
            {
                "subsystem": "Console Input / AEC Termination (การตั้งค่าหน้าเครื่อง / ระบบตัดรังสีอัตโนมัติ)",
                "subsystem_en": "Console Input / AEC Termination",
                "status": "MISMATCH",
                "detail": (
                    f"ค่าที่บันทึกใน DICOM รอบที่ 2 ไม่ตรงกับแผนที่ตั้งไว้ "
                    f"(kVp: {act_kvp:.1f} vs {plan_kvp:.1f}, mAs: {act_mas:.2f} vs {plan_mas:.2f} [{mas_err_pct:+.1f}%], "
                    f"SID: {act_sid:.1f} vs {plan_sid:.1f} cm) — ตรวจสอบการคีย์ค่าหน้าตู้หรือการทำงานของเซ็นเซอร์ AEC"
                ),
            }
        )

    if abs(delta_di) > 1.0:
        severity = "FAIL"
        direction = (
            "สูงกว่าที่คาดการณ์ (Over-output)"
            if delta_di > 0
            else "ต่ำกว่าที่คาดการณ์ (Under-output / Tube Aging)"
        )
        findings.append(
            {
                "subsystem": "X-ray Tube & High-Voltage Generator (หลอดเอกซเรย์และชุดกำเนิดไฟฟ้าแรงสูง)",
                "subsystem_en": "X-ray Tube & High-Voltage Generator",
                "status": "FAIL",
                "detail": (
                    f"ค่า Deviation Index จริงคลาดเคลื่อนจากที่คาดการณ์อย่างมีนัยสำคัญ (ΔDI = {delta_di:+.2f} > ±1.00, {direction}) "
                    f"บ่งชี้ความไม่เป็นเชิงเส้นของ mAs/kVp หรือหลอดเอกซเรย์เสื่อมสภาพ แนะนำวัด Air Kerma Output & HVL ด้วย Dosimeter"
                ),
            }
        )
    elif abs(delta_di) > 0.5:
        if severity != "FAIL":
            severity = "WARNING"
        findings.append(
            {
                "subsystem": "X-ray Tube & High-Voltage Generator (หลอดเอกซเรย์และชุดกำเนิดไฟฟ้าแรงสูง)",
                "subsystem_en": "X-ray Tube & High-Voltage Generator",
                "status": "WARNING",
                "detail": (
                    f"ค่า Deviation Index จริงเบี่ยงเบนเล็กน้อยจากแบบจำลอง (ΔDI = {delta_di:+.2f}, เกณฑ์เฝ้าระวัง ±0.50) "
                    f"ควรเฝ้าระวังเสถียรภาพการจ่ายรังสีของหลอด (Tube Output Consistency)"
                ),
            }
        )

    if noise_err_pct > 25.0 or snr_err_pct < -25.0:
        severity = "FAIL"
        findings.append(
            {
                "subsystem": "Image Receptor / Flat-Panel Detector (ฉากรับภาพ / หัวรับรังสี)",
                "subsystem_en": "Image Receptor / Flat-Panel Detector",
                "status": "FAIL",
                "detail": (
                    f"สัญญาณรบกวน (Noise SD) สูงกว่าที่แบบจำลองควอนตัมคาดการณ์ {noise_err_pct:+.1f}% "
                    f"(SNR คลาดเคลื่อน {snr_err_pct:+.1f}%) บ่งชี้ปัญหา Electronic/Dark Noise สูงผิดปกติ, "
                    f"Gain Calibration เพี้ยน หรือ DQE ของฉากรับภาพเสื่อม แนะนำทำ Dark-Field & Flat-Field Calibration"
                ),
            }
        )
    elif noise_err_pct > 15.0 or snr_err_pct < -15.0:
        if severity != "FAIL":
            severity = "WARNING"
        findings.append(
            {
                "subsystem": "Image Receptor / Flat-Panel Detector (ฉากรับภาพ / หัวรับรังสี)",
                "subsystem_en": "Image Receptor / Flat-Panel Detector",
                "status": "WARNING",
                "detail": (
                    f"ค่า Noise จริงสูงกว่าแบบจำลอง {noise_err_pct:+.1f}% (SNR {snr_err_pct:+.1f}%) "
                    f"ควรตรวจสอบ Gain/Offset Calibration และอุณหภูมิของ Detector"
                ),
            }
        )

    if cnr_err_pct < -20.0 and snr_err_pct >= -15.0:
        if severity != "FAIL":
            severity = "WARNING"
        findings.append(
            {
                "subsystem": "Anti-Scatter Grid & Beam Collimation (กริดกันรังสีกระเจิงและคอลลิเมเตอร์)",
                "subsystem_en": "Anti-Scatter Grid & Beam Collimation",
                "status": "WARNING",
                "detail": (
                    f"ความต่างระดับสี (CNR) จริงต่ำกว่าคาดการณ์ {cnr_err_pct:+.1f}% แม้ค่า SNR และปริมาณรังสีอยู่ในเกณฑ์ "
                    f"บ่งชี้รังสีกระเจิง (Scatter) เพิ่มสูงขึ้น ตรวจสอบการใส่ Grid (Grid Cutoff / Misalignment) หรือขนาด Field Collimation"
                ),
            }
        )

    if len(findings) == 0:
        findings.append(
            {
                "subsystem": "All Subsystems Verified (ทั้งแบบจำลองและตัวเครื่องเอกซเรย์ทำงานสอดคล้องกัน)",
                "subsystem_en": "All Subsystems Verified (Simulation & Hardware Consistent)",
                "status": "PASS",
                "detail": (
                    f"ผลการยิงจริงรอบที่ 2 สอดคล้องกับค่าคาดการณ์ของระบบ (ΔDI = {delta_di:+.2f} ≤ ±0.50, "
                    f"SNR Error = {snr_err_pct:+.1f}%, CNR Error = {cnr_err_pct:+.1f}%) ยืนยันว่าทั้งโปรโตคอลใหม่, "
                    f"หลอดเอกซเรย์ และฉากรับภาพทำงานได้อย่างแม่นยำและปลอดภัยต่อผู้ป่วย"
                ),
            }
        )

    summary_metrics = {
        "delta_di": delta_di,
        "snr_err_pct": snr_err_pct,
        "cnr_err_pct": cnr_err_pct,
        "noise_err_pct": noise_err_pct,
        "kvp_diff": kvp_diff,
        "mas_err_pct": mas_err_pct,
        "sid_diff": sid_diff,
        "severity": severity,
    }
    return findings, summary_metrics


# ==============================================================================
# ฟังก์ชันสร้างรายงาน PDF ทางการ ขนาด 1 หน้า A4 (จัดระยะพอดีขอบกระดาษ & รองรับฟอนต์ไทย)
# ==============================================================================
def generate_qc_report_pdf(
    report_mode: str,
    study_info: dict,
    params_rows: list,
    qc_table_rows: list,
    safety_status: tuple,
    quality_status: tuple,
    audit_notes: list,
    img_left: np.ndarray,
    img_left_title: str,
    img_right: np.ndarray = None,
    img_right_title: str = "",
    img_third: np.ndarray = None,
    img_third_title: str = "",
    roi_info: dict = None,
    hist_mode: bool = False,
    institution_name: str = "DEPARTMENT OF RADIOLOGY & MEDICAL PHYSICS",
    equipment_id: str = "RAD-UNIT-01",
    operator_name: str = "",
    physicist_name: str = "",
    report_title: str = "RADIOLOGY QUALITY CONTROL AND\nDOSE OPTIMIZATION REPORT",
    roi_hist_labels: tuple = ("Signal ROI", "Background ROI", "Insert 2 ROI"),
) -> bytes:
    """สร้างเอกสารทางการขนาด 1 หน้า A4 (210 x 297 mm) มาตรฐานฟิสิกส์การแพทย์"""
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 297)
    ax.axis("off")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    NAVY = "#102A43"
    SLATE = "#334E68"
    MUTED = "#627D98"
    BORDER = "#9FB3C8"
    LIGHT_BG = "#F0F4F8"
    ALT_ROW = "#F8FAFC"

    ax.add_patch(patches.Rectangle((0, 0), 210, 297, facecolor="#FFFFFF", edgecolor="none"))
    ax.add_patch(patches.Rectangle((9, 9), 192, 279, facecolor="none", edgecolor=NAVY, linewidth=1.1))
    ax.add_patch(patches.Rectangle((10.2, 10.2), 189.6, 276.6, facecolor="none", edgecolor=BORDER, linewidth=0.45))

    now_dt = datetime.datetime.now()
    now_str = now_dt.strftime("%d %b %Y, %H:%M:%S")
    doc_no = f"QA-RAD-{now_dt.strftime('%Y%m%d-%H%M%S')}"
    inst_raw = (
        institution_name.strip().upper()
        if institution_name.strip()
        else "DEPARTMENT OF RADIOLOGY & MEDICAL PHYSICS"
    )
    inst_display = _wrap_to_mm(ax, renderer, inv, inst_raw, max_width_mm=120.0, fontsize=7.2, fontweight="bold", max_lines=1)
    equip_display = _wrap_to_mm(
        ax,
        renderer,
        inv,
        equipment_id.strip() if equipment_id.strip() else "RAD-UNIT-01",
        max_width_mm=34.0,
        fontsize=6.3,
        fontweight="bold",
        max_lines=1,
    )
    scope_line = _wrap_to_mm(
        ax,
        renderer,
        inv,
        f"Scope: {report_mode} | System ID: {equip_display}",
        max_width_mm=121.0,
        fontsize=6.3,
        fontweight="bold",
        max_lines=1,
    )

    # HEADER
    _draw_text_mm(ax, renderer, inv, 14, 282.2, inst_display, fontsize=7.2, fontweight="bold", color=SLATE, va="center")
    _draw_text_mm(ax, renderer, inv, 14, 275.3, report_title, fontsize=10.6, fontweight="bold", color=NAVY, va="center", linespacing=1.18)
    _draw_text_mm(ax, renderer, inv, 14, 265.4, scope_line, fontsize=6.3, fontweight="bold", color="#1F2933", va="center")
    _draw_text_mm(
        ax,
        renderer,
        inv,
        14,
        261.2,
        "Compliance Basis: ICRP Publ. 103 (ALARA) | IEC 62494-1 | IAEA TRS-457 | AAPM TG-150",
        fontsize=5.9,
        color=MUTED,
        va="center",
    )

    # DOCUMENT CONTROL BOX
    dc_x, dc_y, dc_w, dc_h = 138.0, 259.8, 58.0, 24.5
    ax.add_patch(patches.Rectangle((dc_x, dc_y), dc_w, dc_h, facecolor=ALT_ROW, edgecolor=NAVY, linewidth=0.7))
    ax.add_patch(patches.Rectangle((dc_x, dc_y + dc_h - 5.2), dc_w, 5.2, facecolor=NAVY, edgecolor="none"))
    _draw_text_mm(
        ax,
        renderer,
        inv,
        dc_x + dc_w / 2,
        dc_y + dc_h - 2.6,
        "DOCUMENT CONTROL RECORD",
        fontsize=6.2,
        fontweight="bold",
        color="#FFFFFF",
        ha="center",
        va="center",
    )

    dc_rows = [
        ("Document ID:", doc_no),
        ("Issue Date:", now_str),
        ("Record Class:", "Quality Assurance Audit"),
        ("Page:", "1 of 1 (Controlled Copy)"),
    ]
    for idx, (k, v) in enumerate(dc_rows):
        ry = dc_y + dc_h - 7.8 - idx * 4.4
        if idx > 0:
            ax.plot([dc_x, dc_x + dc_w], [ry + 2.2, ry + 2.2], color="#D9E2EC", linewidth=0.4)
        _draw_text_mm(ax, renderer, inv, dc_x + 2.0, ry, k, fontsize=5.9, fontweight="bold", color=SLATE, va="center")
        _draw_text_mm(ax, renderer, inv, dc_x + dc_w - 2.0, ry, v, fontsize=5.9, color="#102A43", ha="right", va="center")

    ax.plot([13, 197], [257.2, 257.2], color=NAVY, linewidth=1.3)
    ax.plot([13, 197], [256.1, 256.1], color=NAVY, linewidth=0.45)

    # OVERALL QC DISPOSITION BANNER
    overall_pass = (safety_status[0] == "PASS") and (quality_status[0] == "PASS")
    overall_fail = (safety_status[0] == "FAIL") or (quality_status[0] == "FAIL")

    if overall_pass:
        disp_bg, disp_fg, disp_label = (
            "#ECFDF5",
            "#065F46",
            "ACCEPTED - COMPLIANT WITH ALARA & DIAGNOSTIC QUALITY CRITERIA",
        )
    elif overall_fail:
        disp_bg, disp_fg, disp_label = (
            "#FEF2F2",
            "#991B1B",
            "REJECTED - EXCEEDS ACTION LIMIT / BELOW DIAGNOSTIC THRESHOLD",
        )
    else:
        disp_bg, disp_fg, disp_label = (
            "#FFFBEB",
            "#92400E",
            "CONDITIONAL - REQUIRES MEDICAL PHYSICIST REVIEW (TOLERANCE WARNING)",
        )

    ax.add_patch(patches.Rectangle((13, 246.2), 184, 7.8, facecolor=disp_bg, edgecolor=disp_fg, linewidth=0.85))
    _draw_text_mm(ax, renderer, inv, 15.5, 250.1, "OVERALL QC DISPOSITION:", fontsize=6.6, fontweight="bold", color=NAVY, va="center")
    disp_fit = _wrap_to_mm(ax, renderer, inv, disp_label, max_width_mm=140.0, fontsize=6.4, fontweight="bold", max_lines=1)
    _draw_text_mm(ax, renderer, inv, 54.5, 250.1, disp_fit, fontsize=6.4, fontweight="bold", color=disp_fg, va="center")

    # SECTION 1
    _draw_text_mm(
        ax,
        renderer,
        inv,
        13,
        242.2,
        "SECTION 1. EXAMINATION / PHANTOM IDENTIFICATION & EXPOSURE PARAMETERS",
        fontsize=7.8,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.plot([13, 197], [240.3, 240.3], color=NAVY, linewidth=0.75)

    meta_items = list(study_info.items())
    col_w = 184.0 / max(len(meta_items), 1)
    ax.add_patch(patches.Rectangle((13, 230.2), 184, 9.0, facecolor=LIGHT_BG, edgecolor=BORDER, linewidth=0.55))
    for idx, (k, v) in enumerate(meta_items):
        x0 = 13 + idx * col_w
        if idx > 0:
            ax.plot([x0, x0], [230.2, 239.2], color=BORDER, linewidth=0.45)
        k_fit = _wrap_to_mm(ax, renderer, inv, str(k).upper(), max_width_mm=col_w - 3.5, fontsize=5.6, fontweight="bold", max_lines=1)
        v_fit = _wrap_to_mm(ax, renderer, inv, str(v), max_width_mm=col_w - 3.5, fontsize=6.2, fontweight="bold", max_lines=1)
        _draw_text_mm(ax, renderer, inv, x0 + 2.0, 236.6, k_fit, fontsize=5.6, fontweight="bold", color=SLATE, va="center")
        _draw_text_mm(ax, renderer, inv, x0 + 2.0, 232.5, v_fit, fontsize=6.2, fontweight="bold", color="#102A43", va="center")

    tbl1_top = 228.5
    row_h1 = 4.9
    headers1 = ["Acquisition Parameter", "Baseline / Reference", "Evaluated / Planned", "Actual / Variance"]
    col_xs1 = [13, 66, 110, 154, 197]

    ax.add_patch(patches.Rectangle((13, tbl1_top - row_h1), 184, row_h1, facecolor=NAVY, edgecolor=NAVY, linewidth=0.5))
    for i_h, h_txt in enumerate(headers1):
        cx = col_xs1[i_h]
        cw = col_xs1[i_h + 1] - cx
        h_fit = _wrap_to_mm(ax, renderer, inv, h_txt, cw - 3.5, 6.3, "bold", 1)
        _draw_text_mm(ax, renderer, inv, cx + 2.0, tbl1_top - row_h1 / 2, h_fit, fontsize=6.3, fontweight="bold", color="#FFFFFF", va="center")

    for r_idx, row in enumerate(params_rows[:5]):
        y_top = tbl1_top - (r_idx + 1) * row_h1
        bg_c = "#FFFFFF" if r_idx % 2 == 0 else ALT_ROW
        ax.add_patch(patches.Rectangle((13, y_top - row_h1), 184, row_h1, facecolor=bg_c, edgecolor=BORDER, linewidth=0.4))
        for c_i, val in enumerate(row[:4]):
            cx = col_xs1[c_i]
            cw = col_xs1[c_i + 1] - cx
            if c_i > 0:
                ax.plot([cx, cx], [y_top - row_h1, y_top], color="#D9E2EC", linewidth=0.35)
            fw = "bold" if c_i == 0 else "normal"
            val_fit = _wrap_to_mm(ax, renderer, inv, str(val), max_width_mm=cw - 3.5, fontsize=6.1, fontweight=fw, max_lines=1)
            _draw_text_mm(ax, renderer, inv, cx + 2.0, y_top - row_h1 / 2, val_fit, fontsize=6.1, fontweight=fw, color="#102A43", va="center")

    # SECTION 2
    sec2_y = tbl1_top - (min(len(params_rows), 5) + 1) * row_h1 - 4.0
    _draw_text_mm(
        ax,
        renderer,
        inv,
        13,
        sec2_y,
        "SECTION 2. QUANTITATIVE IMAGE QUALITY & DOSIMETRY AUDIT RESULTS",
        fontsize=7.8,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.plot([13, 197], [sec2_y - 1.9, sec2_y - 1.9], color=NAVY, linewidth=0.75)

    tbl2_top = sec2_y - 3.2
    row_h2 = 4.9
    headers2 = ["Quality / Dosimetry Metric", "Measured / Actual", "Tolerance / Predicted Target", "Status"]
    col_xs2 = [13, 72, 108, 173, 197]

    ax.add_patch(patches.Rectangle((13, tbl2_top - row_h2), 184, row_h2, facecolor=NAVY, edgecolor=NAVY, linewidth=0.5))
    for i_h, h_txt in enumerate(headers2):
        cx = col_xs2[i_h]
        cw = col_xs2[i_h + 1] - cx
        h_fit = _wrap_to_mm(ax, renderer, inv, h_txt, cw - 3.5, 6.3, "bold", 1)
        _draw_text_mm(ax, renderer, inv, cx + 2.0, tbl2_top - row_h2 / 2, h_fit, fontsize=6.3, fontweight="bold", color="#FFFFFF", va="center")

    for r_idx, row in enumerate(qc_table_rows[:8]):
        y_top = tbl2_top - (r_idx + 1) * row_h2
        bg_c = "#FFFFFF" if r_idx % 2 == 0 else ALT_ROW
        ax.add_patch(patches.Rectangle((13, y_top - row_h2), 184, row_h2, facecolor=bg_c, edgecolor=BORDER, linewidth=0.4))
        for c_i, val in enumerate(row[:4]):
            cx = col_xs2[c_i]
            cw = col_xs2[c_i + 1] - cx
            if c_i > 0:
                ax.plot([cx, cx], [y_top - row_h2, y_top], color="#D9E2EC", linewidth=0.35)
            txt_color = "#102A43"
            fw = "bold" if c_i in (0, 1) else "normal"
            if c_i == 3:
                fw = "bold"
                if val == "PASS":
                    txt_color = "#065F46"
                elif val == "WARNING":
                    txt_color = "#92400E"
                elif val == "FAIL":
                    txt_color = "#991B1B"
                else:
                    txt_color = SLATE
            val_fit = _wrap_to_mm(ax, renderer, inv, str(val), max_width_mm=cw - 3.5, fontsize=6.1, fontweight=fw, max_lines=1)
            _draw_text_mm(ax, renderer, inv, cx + 2.0, y_top - row_h2 / 2, val_fit, fontsize=6.1, fontweight=fw, color=txt_color, va="center")

    # SECTION 3
    sec3_y = tbl2_top - (min(len(qc_table_rows), 8) + 1) * row_h2 - 4.0
    _draw_text_mm(
        ax,
        renderer,
        inv,
        13,
        sec3_y,
        "SECTION 3. VISUAL VERIFICATION & REGION-OF-INTEREST (ROI) TRACEABILITY",
        fontsize=7.8,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.plot([13, 197], [sec3_y - 1.9, sec3_y - 1.9], color=NAVY, linewidth=0.75)

    img_box_top = sec3_y - 3.2
    img_box_bottom = 83.5
    img_box_h = img_box_top - img_box_bottom
    ax.add_patch(patches.Rectangle((13, img_box_bottom), 184, img_box_h, facecolor=ALT_ROW, edgecolor=BORDER, linewidth=0.55))

    def _draw_roi(target_ax):
        if roi_info is not None:
            r_sz = roi_info["roi_size"]
            sx0 = roi_info.get("sig_x0", roi_info["sig_cx"] - r_sz / 2.0)
            sy0 = roi_info.get("sig_y0", roi_info["sig_cy"] - r_sz / 2.0)
            bx0 = roi_info.get("bg_x0", roi_info["bg_cx"] - r_sz / 2.0)
            by0 = roi_info.get("bg_y0", roi_info["bg_cy"] - r_sz / 2.0)
            target_ax.add_patch(
                patches.Rectangle((sx0, sy0), r_sz, r_sz, linewidth=1.0, edgecolor="#DC2626", facecolor="none")
            )
            target_ax.add_patch(
                patches.Rectangle((bx0, by0), r_sz, r_sz, linewidth=1.0, edgecolor="#0284C7", facecolor="none")
            )
            if "ins2_cx" in roi_info:
                i2x0 = roi_info.get("ins2_x0", roi_info["ins2_cx"] - r_sz / 2.0)
                i2y0 = roi_info.get("ins2_y0", roi_info["ins2_cy"] - r_sz / 2.0)
                target_ax.add_patch(
                    patches.Rectangle((i2x0, i2y0), r_sz, r_sz, linewidth=1.0, edgecolor="#059669", facecolor="none")
                )

    if img_third is not None and img_right is not None:
        panel_specs = [
            (15.5, img_left, _wrap_to_mm(ax, renderer, inv, img_left_title, 54.0, 5.7, "bold", 1)),
            (77.0, img_right, _wrap_to_mm(ax, renderer, inv, img_right_title, 54.0, 5.7, "bold", 1)),
            (138.5, img_third, _wrap_to_mm(ax, renderer, inv, img_third_title, 54.0, 5.7, "bold", 1)),
        ]
        for px, p_img, p_title in panel_specs:
            sub_ax = fig.add_axes(
                [px / 210.0, (img_box_bottom + 4.8) / 297.0, 56.0 / 210.0, (img_box_h - 10.2) / 297.0]
            )
            sub_ax.imshow(p_img, cmap="gray")
            sub_ax.set_title(p_title, fontsize=5.7, fontweight="bold", color=NAVY, pad=2.3)
            sub_ax.axis("off")
            _draw_roi(sub_ax)
    else:
        ax_l = fig.add_axes([15.5 / 210.0, (img_box_bottom + 4.8) / 297.0, 83 / 210.0, (img_box_h - 10.2) / 297.0])
        ax_l.imshow(img_left, cmap="gray")
        ax_l.set_title(_wrap_to_mm(ax, renderer, inv, img_left_title, 80.0, 6.3, "bold", 1), fontsize=6.3, fontweight="bold", color=NAVY, pad=2.5)
        ax_l.axis("off")
        _draw_roi(ax_l)

        if hist_mode and roi_info is not None:
            ax_r = fig.add_axes([113 / 210.0, (img_box_bottom + 9.2) / 297.0, 80 / 210.0, (img_box_h - 15.5) / 297.0])
            ax_r.hist(roi_info["sig_roi"].ravel(), bins=22, alpha=0.68, color="#DC2626", label=roi_hist_labels[0])
            ax_r.hist(roi_info["bg_roi"].ravel(), bins=22, alpha=0.68, color="#0284C7", label=roi_hist_labels[1])
            if "ins2_roi" in roi_info and len(roi_hist_labels) >= 3:
                ax_r.hist(roi_info["ins2_roi"].ravel(), bins=22, alpha=0.68, color="#059669", label=roi_hist_labels[2])
            ax_r.set_title("Figure 1B. ROI Pixel Intensity Distribution", fontsize=6.3, fontweight="bold", color=NAVY, pad=2.5)
            ax_r.set_xlabel("Pixel Intensity (ADU)", fontsize=5.6, color=SLATE)
            ax_r.set_ylabel("Frequency (Pixels)", fontsize=5.6, color=SLATE)
            ax_r.tick_params(labelsize=5.2, colors=SLATE)
            ax_r.spines["top"].set_visible(False)
            ax_r.spines["right"].set_visible(False)
            ax_r.grid(alpha=0.25, linestyle="--")
            ax_r.legend(fontsize=5.0, loc="upper right")
        elif img_right is not None:
            ax_r = fig.add_axes([111.5 / 210.0, (img_box_bottom + 4.8) / 297.0, 83 / 210.0, (img_box_h - 10.2) / 297.0])
            ax_r.imshow(img_right, cmap="gray")
            ax_r.set_title(_wrap_to_mm(ax, renderer, inv, img_right_title, 80.0, 6.3, "bold", 1), fontsize=6.3, fontweight="bold", color=NAVY, pad=2.5)
            ax_r.axis("off")
            _draw_roi(ax_r)

    if roi_info is not None:
        if "ins2_cx" in roi_info:
            roi_caption = (
                f"ImageJ ROI ({roi_info['roi_size']}x{roi_info['roi_size']} px, Centered, ddof=1): "
                f"Insert1(Red)=({roi_info['sig_cx']},{roi_info['sig_cy']}) | "
                f"Bg(Blue)=({roi_info['bg_cx']},{roi_info['bg_cy']}) | "
                f"Insert2(Green)=({roi_info['ins2_cx']},{roi_info['ins2_cy']})"
            )
        else:
            roi_caption = (
                f"Calibrated ImageJ ROI Spec (Centered, Unbiased SD ddof=1): "
                f"Box = {roi_info['roi_size']}x{roi_info['roi_size']} px | "
                f"Signal ROI (Red) = ({roi_info['sig_cx']}, {roi_info['sig_cy']}) | "
                f"Background ROI (Blue) = ({roi_info['bg_cx']}, {roi_info['bg_cy']})"
            )
        roi_cap_fit = _wrap_to_mm(ax, renderer, inv, roi_caption, max_width_mm=178.0, fontsize=5.5, fontweight="bold", max_lines=1)
        _draw_text_mm(
            ax,
            renderer,
            inv,
            105,
            img_box_bottom + 2.0,
            roi_cap_fit,
            fontsize=5.5,
            fontweight="bold",
            color=SLATE,
            ha="center",
            va="center",
        )

    # SECTION 4
    sec4_y = 79.5
    _draw_text_mm(
        ax,
        renderer,
        inv,
        13,
        sec4_y,
        "SECTION 4. REGULATORY COMPLIANCE FINDINGS & AUTHORIZATION SIGN-OFF",
        fontsize=7.8,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    ax.plot([13, 197], [sec4_y - 1.9, sec4_y - 1.9], color=NAVY, linewidth=0.75)

    comp_top = sec4_y - 3.2
    comp_bottom = 44.5
    ax.add_patch(
        patches.Rectangle((13, comp_bottom), 184, comp_top - comp_bottom, facecolor="#FFFFFF", edgecolor=BORDER, linewidth=0.55)
    )

    line1_full = f"1. Radiation Protection & Exposure Index Compliance [{safety_status[0]}]: {safety_status[1]}"
    line1_wrapped = _wrap_to_mm(ax, renderer, inv, line1_full, max_width_mm=176.0, fontsize=6.0, fontweight="bold", max_lines=2)
    _draw_text_mm(
        ax,
        renderer,
        inv,
        15.5,
        comp_top - 2.3,
        line1_wrapped,
        fontsize=6.0,
        fontweight="bold",
        color="#1F2933",
        va="top",
        linespacing=1.22,
    )

    line2_full = f"2. Diagnostic Quality & Hardware Verification [{quality_status[0]}]: {quality_status[1]}"
    line2_wrapped = _wrap_to_mm(ax, renderer, inv, line2_full, max_width_mm=176.0, fontsize=6.0, fontweight="bold", max_lines=2)
    _draw_text_mm(
        ax,
        renderer,
        inv,
        15.5,
        comp_top - 9.2,
        line2_wrapped,
        fontsize=6.0,
        fontweight="bold",
        color="#1F2933",
        va="top",
        linespacing=1.22,
    )

    ax.plot([15, 195], [comp_top - 16.0, comp_top - 16.0], color="#D9E2EC", linewidth=0.5)
    _draw_text_mm(
        ax,
        renderer,
        inv,
        15.5,
        comp_top - 18.2,
        "3. Technical Methodology & Subsystem Diagnostic Notes:",
        fontsize=6.1,
        fontweight="bold",
        color=NAVY,
        va="center",
    )

    cur_note_y = comp_top - 21.0
    for note_line in audit_notes[:3]:
        wrapped_note = _wrap_to_mm(ax, renderer, inv, f"- {note_line}", max_width_mm=174.0, fontsize=5.6, fontweight="normal", max_lines=2)
        n_lines = wrapped_note.count("\n") + 1
        if cur_note_y - (n_lines * 2.8) < comp_bottom + 0.8:
            break
        _draw_text_mm(ax, renderer, inv, 17.0, cur_note_y, wrapped_note, fontsize=5.6, color=SLATE, va="top", linespacing=1.18)
        cur_note_y -= n_lines * 2.8 + 1.1

    # กล่องลงนามรับรอง
    sig_top = 42.2
    sig_bottom = 13.5
    sig_h = sig_top - sig_bottom
    ax.add_patch(patches.Rectangle((13, sig_bottom), 184, sig_h, facecolor=ALT_ROW, edgecolor=NAVY, linewidth=0.7))
    ax.add_patch(patches.Rectangle((13, sig_top - 4.8), 184, 4.8, facecolor=LIGHT_BG, edgecolor=BORDER, linewidth=0.5))
    _draw_text_mm(
        ax,
        renderer,
        inv,
        15.5,
        sig_top - 2.4,
        "OFFICIAL QUALITY ASSURANCE VERIFICATION & SIGN-OFF RECORD",
        fontsize=6.3,
        fontweight="bold",
        color=NAVY,
        va="center",
    )

    op_disp = _wrap_to_mm(ax, renderer, inv, operator_name.strip() if operator_name.strip() else "____________________", 41.0, 5.7, "normal", 1)
    ph_disp = _wrap_to_mm(ax, renderer, inv, physicist_name.strip() if physicist_name.strip() else "____________________", 41.0, 5.7, "normal", 1)

    roles = [
        ("PERFORMED BY (TECHNOLOGIST)", f"Name / ID: {op_disp}", "Date: ______________________"),
        ("VERIFIED BY (MEDICAL PHYSICIST)", f"Name / Lic.: {ph_disp}", "Date: ______________________"),
        ("APPROVED BY (SAFETY OFFICER)", "Name / Title: ________________", "Date: ______________________"),
    ]
    col_sw = 184.0 / 3.0
    for r_i, (role_title, field1, field2) in enumerate(roles):
        rx0 = 13 + r_i * col_sw
        if r_i > 0:
            ax.plot([rx0, rx0], [sig_bottom, sig_top - 4.8], color=BORDER, linewidth=0.55)
        _draw_text_mm(ax, renderer, inv, rx0 + col_sw / 2, sig_top - 7.3, role_title, fontsize=5.5, fontweight="bold", color=NAVY, ha="center", va="center")
        _draw_text_mm(ax, renderer, inv, rx0 + 2.2, sig_bottom + 13.0, "Signature: ____________________", fontsize=5.7, color=SLATE, va="center")
        _draw_text_mm(ax, renderer, inv, rx0 + 2.2, sig_bottom + 7.8, field1, fontsize=5.7, color=SLATE, va="center")
        _draw_text_mm(ax, renderer, inv, rx0 + 2.2, sig_bottom + 2.8, field2, fontsize=5.7, color=SLATE, va="center")

    _draw_text_mm(
        ax,
        renderer,
        inv,
        13,
        11.4,
        f"CONFIDENTIAL - QUALITY ASSURANCE AUDIT RECORD | {doc_no} | Generated electronically; valid upon authorized signature.",
        fontsize=5.4,
        color=MUTED,
        va="center",
    )

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", dpi=300)
    plt.close(fig)
    return pdf_buf.getvalue()


# ==============================================================================
# ฟังก์ชันของโหมด 1: คำนวณเม็ดซ่า Poisson Noise บนภาพจริง
# ==============================================================================
def apply_physics_noise(img_2d, dose_percent, distance_cm=100.0, ref_distance_cm=100.0):
    if np.isclose(dose_percent, 100.0) and np.isclose(distance_cm, ref_distance_cm):
        return img_2d.copy()

    dose_level = dose_percent / 100.0
    distance_factor = (ref_distance_cm / max(distance_cm, 1.0)) ** 2
    effective_dose = dose_level * distance_factor

    base_photons = 300.0
    photons = np.clip((img_2d / 255.0) * base_photons * effective_dose, 0.001, None)
    sampled = np.random.poisson(photons)
    simulated = (sampled / (base_photons * max(effective_dose, 1e-4))) * 255.0
    return np.clip(simulated, 0, 255)


# ==============================================================================
# ฟังก์ชันของโหมด 2: จำลองแฟนธอมเลือกได้ทั้ง 2 วัสดุ (Single-Insert) และ 3 วัสดุ (Multi-Insert)
# ==============================================================================
def simulate_phantom_radiograph(
    num_materials: int,
    img_size: int,
    pixel_size_mm: float,
    bg_material: str,
    thickness_cm: float,
    obj_shape: str,
    ins1_material: str,
    ins2_material: str,
    obj_diameter_mm: float,
    kvp: float,
    mas: float,
    sid_cm: float,
    grid_on: bool,
    filtration_mm: float,
    rel_dose_pct: float,
    photon_scale: float,
    seed: int,
):
    rng = np.random.default_rng(seed)

    base_photons_per_pixel = 1200.0 * photon_scale
    n0 = (
        base_photons_per_pixel
        * ((kvp / 100.0) ** 2)
        * mas
        * ((100.0 / sid_cm) ** 2)
        * (rel_dose_pct / 100.0)
    )

    mu_al = 0.085 * ((120.0 / kvp) ** 0.8)
    t_filter = np.exp(-mu_al * filtration_mm)

    mu_bg = get_linear_attenuation(bg_material, kvp)
    mu_ins1 = get_linear_attenuation(ins1_material, kvp)
    mu_ins2 = get_linear_attenuation(ins2_material, kvp) if num_materials == 3 else 0.0

    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    radius_px = (obj_diameter_mm / 2.0) / pixel_size_mm
    obj_thickness_cm = min(obj_diameter_mm / 10.0, thickness_cm)

    primary_bg = n0 * t_filter * np.exp(-mu_bg * thickness_cm)
    primary_ins1 = n0 * t_filter * np.exp(
        -(mu_bg * (thickness_cm - obj_thickness_cm) + mu_ins1 * obj_thickness_cm)
    )

    if num_materials == 2:
        cx1, cy1 = img_size // 2, img_size // 2
        cx2, cy2 = None, None
        if obj_shape == "Circle":
            mask1 = ((x_grid - cx1) ** 2 + (y_grid - cy1) ** 2) <= radius_px**2
        else:
            mask1 = (np.abs(x_grid - cx1) <= radius_px) & (np.abs(y_grid - cy1) <= radius_px)
        mask2 = np.zeros_like(mask1, dtype=bool)
        primary_ins2 = 0.0
        primary_map = np.where(mask1, primary_ins1, primary_bg)
    else:
        cx1, cy1 = int(round(img_size * 0.27)), img_size // 2
        cx2, cy2 = int(round(img_size * 0.73)), img_size // 2
        if obj_shape == "Circle":
            mask1 = ((x_grid - cx1) ** 2 + (y_grid - cy1) ** 2) <= radius_px**2
            mask2 = ((x_grid - cx2) ** 2 + (y_grid - cy2) ** 2) <= radius_px**2
        else:
            mask1 = (np.abs(x_grid - cx1) <= radius_px) & (np.abs(y_grid - cy1) <= radius_px)
            mask2 = (np.abs(x_grid - cx2) <= radius_px) & (np.abs(y_grid - cy2) <= radius_px)
        primary_ins2 = n0 * t_filter * np.exp(
            -(mu_bg * (thickness_cm - obj_thickness_cm) + mu_ins2 * obj_thickness_cm)
        )
        primary_map = np.full((img_size, img_size), primary_bg, dtype=np.float64)
        primary_map = np.where(mask1, primary_ins1, primary_map)
        primary_map = np.where(mask2, primary_ins2, primary_map)

    scatter_to_primary = 0.04 * thickness_cm
    if grid_on:
        grid_primary_trans = 0.75
        scatter_to_primary *= 0.20
    else:
        grid_primary_trans = 1.00

    expected_photons = primary_map * grid_primary_trans * (1.0 + scatter_to_primary)

    detected_image = rng.poisson(np.maximum(expected_photons, 0.0)).astype(np.float64)
    image_uint16 = np.clip(detected_image, 0, 65535).astype(np.uint16)

    esd_mgy = (
        0.12
        * ((kvp / 100.0) ** 2)
        * mas
        * ((100.0 / sid_cm) ** 2)
        * (rel_dose_pct / 100.0)
    )
    bg_only_mask = ~(mask1 | mask2)
    mean_bg_theoretical = float(np.mean(expected_photons[bg_only_mask]))
    ei = mean_bg_theoretical * 0.1
    target_ei = 250.0
    di = 10.0 * np.log10(max(ei, 1e-6) / target_ei)

    physics_summary = {
        "N0_incident": n0,
        "mu_bg": mu_bg,
        "mu_ins1": mu_ins1,
        "mu_ins2": mu_ins2,
        "primary_bg": primary_bg,
        "primary_ins1": primary_ins1,
        "primary_ins2": primary_ins2,
        "ESD_mGy": esd_mgy,
        "EI": ei,
        "DI": di,
        "cx1": cx1,
        "cy1": cy1,
        "cx2": cx2,
        "cy2": cy2,
    }
    return image_uint16, physics_summary


def create_dicom_bytes(
    image_uint16: np.ndarray,
    pixel_size_mm: float,
    kvp: float,
    ma: float,
    exp_ms: float,
    sid_cm: float,
    num_materials: int,
    bg_material: str,
    ins1_material: str,
    ins2_material: str,
    thickness_cm: float,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    now = datetime.datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.Modality = "CR"
    ds.Manufacturer = "Virtual Radiography Lab"
    if num_materials == 2:
        ds.PatientName = f"Phantom2Mat^{bg_material}_{ins1_material}_{int(thickness_cm)}cm"
        ds.PatientID = "VRL-2MAT-001"
    else:
        ds.PatientName = f"Phantom3Mat^{bg_material}_{ins1_material}_{ins2_material}_{int(thickness_cm)}cm"
        ds.PatientID = "VRL-3MAT-001"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    ds.Rows, ds.Columns = image_uint16.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = [str(pixel_size_mm), str(pixel_size_mm)]
    ds.KVP = str(kvp)
    ds.XRayTubeCurrent = str(int(round(ma)))
    ds.ExposureTime = str(int(round(exp_ms)))
    ds.Exposure = str(round((ma * exp_ms) / 1000.0, 2))
    ds.DistanceSourceToDetector = str(sid_cm * 10.0)
    ds.PixelData = image_uint16.tobytes()

    buffer = io.BytesIO()
    ds.save_as(buffer, write_like_original=False)
    return buffer.getvalue()


# ==============================================================================
# เมนูด้านข้าง (Sidebar) — เรียงลำดับ 1 -> 2 -> 3 อย่างถูกต้อง
# ==============================================================================
with st.sidebar:
    st.header("🧭 เลือกโหมดการทำงาน")
    workspace_mode = st.radio(
        "โหมดหลัก:",
        [
            "📂 1. วิเคราะห์ & จำลองลดรังสี (1st Shot)",
            "🧪 2. จำลอง Phantom & X-ray (Virtual Lab)",
            "🎯 3. ตรวจรับผลยิงจริงรอบ 2 (Post-Exposure QA)",
        ],
    )
    if "post_qa_plan" in st.session_state:
        p_plan = st.session_state["post_qa_plan"]
        st.success(
            f"📌 มีแผนคาดการณ์พร้อมตรวจรับผลรอบ 2:\n"
            f"**ID: {p_plan['patient_id']}** | **{p_plan['plan_kvp']:.0f} kVp, {p_plan['plan_mas']:.1f} mAs**"
        )

    st.divider()
    with st.expander("🏛️ ข้อมูลหัวรายงานทางการ (PDF Metadata)", expanded=False):
        rpt_institution = st.text_input(
            "ชื่อหน่วยงาน / โรงพยาบาล",
            value="DEPARTMENT OF RADIOLOGY & MEDICAL PHYSICS",
        )
        rpt_equipment = st.text_input("รหัสเครื่องตรวจ (System / Unit ID)", value="RAD-UNIT-01")
        rpt_operator = st.text_input("ชื่อผู้ตรวจ (Operator / Technologist)", value="")
        rpt_physicist = st.text_input("ชื่อนักฟิสิกส์การแพทย์ (Medical Physicist)", value="")

    st.divider()
    if st.button("🔄 ลบไฟล์ทั้งหมด / เริ่มต้นใหม่", use_container_width=True, type="primary"):
        reset_app()


# ==============================================================================
# โหมดที่ 1: วิเคราะห์ไฟล์ DICOM จริงรอบที่ 1 & จำลองคาดการณ์
# ==============================================================================
if workspace_mode.startswith("📂"):
    st.title("Virtual 3D Radiology & DICOM Series Viewer Workbench")
    st.write("ขั้นตอนที่ 1: วิเคราะห์ภาพยิงรอบแรก และจำลองหาพารามิเตอร์ที่ปลอดภัยตามหลัก ALARA ก่อนนำไปยิงจริงรอบ 2")

    uploaded_files = st.file_uploader(
        "📥 ลากโฟลเดอร์ภาพยิงรอบที่ 1 มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์ DICOM / DICOM-RT",
        type=None,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        try:
            with st.spinner("กำลังอ่านข้อมูลและสแกนพารามิเตอร์ DICOM..."):
                dicom_data = load_dicom_series(uploaded_files)

            if dicom_data is not None:
                volume_3d = dicom_data["volume_3d"]
                norm_volume = dicom_data["norm_volume"]
                num_slices = dicom_data["num_slices"]
                modality = dicom_data["modality"]
                patient_id = dicom_data["patient_id"]
                is_rt_file = dicom_data["is_rt_file"]

                c_info, c_btn = st.columns([4, 1])
                with c_info:
                    st.success(
                        f"โหลดข้อมูลรอบที่ 1 สำเร็จ! อ่านภาพ DICOM ได้ **{num_slices} Slices** | "
                        f"Modality: **{modality}** | ID: **{patient_id}**"
                    )
                with c_btn:
                    if st.button("🗑️ ล้างไฟล์ชุดนี้", use_container_width=True):
                        reset_app()

                st.divider()
                col_controls, col_display = st.columns([1.1, 1.9])

                with col_controls:
                    st.subheader("⚙️ ตัวควบคุม Layer & ระดับการปรับลดรังสี")

                    if num_slices > 1:
                        selected_slice = st.slider(
                            "🔍 เลื่อนดูความลึกชั้นภาพ (Slice Depth Index)",
                            min_value=0,
                            max_value=num_slices - 1,
                            value=num_slices // 2,
                        )
                        st.caption(f"กำลังดู Layer ที่: **{selected_slice + 1} จากทั้งหมด {num_slices} Layers**")
                    else:
                        selected_slice = 0

                    dose_percent = st.slider("📉 ระดับปริมาณรังสีจำลอง (%)", min_value=1, max_value=100, value=100)
                    dose_level = dose_percent / 100.0

                    roi_size_real = st.slider("📐 ขนาดกรอบ ROI เทียบ ImageJ (px)", 10, 80, 30, step=2)
                    show_roi_overlay = st.checkbox("แสดงกรอบ ROI บนภาพ", value=True)

                    st.divider()
                    st.subheader("🎛️ พารามิเตอร์การตั้งค่าเครื่องเปรียบเทียบ")

                    base_sid = float(dicom_data["sid"])
                    base_kvp = float(dicom_data["kvp"])
                    base_mas = float(dicom_data["mas"])
                    base_angle = float(dicom_data["angle"])

                    if is_rt_file:
                        st.info("🤖 **ค่า Baseline ล็อกอัตโนมัติจาก DICOM-RT | พิมพ์หรือปรับค่าจำลองได้ทางฝั่งขวา**")
                    else:
                        st.caption("🔒 **ค่า Baseline ล็อกตามภาพต้นฉบับ | พิมพ์ตัวเลขหรือกด +/- ปรับค่าจำลองได้ทางฝั่งขวา**")

                    base_sig = (patient_id, num_slices, base_sid, base_kvp, base_mas, base_angle)
                    if st.session_state.get("last_base_sig") != base_sig:
                        st.session_state["last_base_sig"] = base_sig
                        st.session_state["s_sid"] = float(np.clip(base_sid, 10.0, 300.0))
                        st.session_state["s_kvp"] = float(np.clip(base_kvp, 20.0, 150.0))
                        st.session_state["s_mas"] = float(np.clip(base_mas * dose_level, 0.01, 2000.0))
                        st.session_state["s_ang"] = float(np.clip(base_angle, 0.0, 360.0))
                        st.session_state["last_dose_sync"] = dose_percent
                    elif st.session_state.get("last_dose_sync") != dose_percent:
                        st.session_state["last_dose_sync"] = dose_percent
                        st.session_state["s_mas"] = float(np.clip(base_mas * dose_level, 0.01, 2000.0))

                    col_base, col_sim = st.columns(2)
                    with col_base:
                        st.markdown("##### 🔒 ค่าเริ่มต้น (1st Shot)")
                        st.text_input("ระยะทาง SID ต้นฉบับ", value=f"{base_sid:.1f} cm", disabled=True)
                        st.text_input("แรงดันไฟฟ้า (kVp)", value=f"{base_kvp:.1f} kVp", disabled=True)
                        st.text_input("กระแส-เวลาต้นฉบับ", value=f"{base_mas:.2f} mAs", disabled=True)
                        st.text_input("มุม Gantry", value=f"{base_angle:.1f}°", disabled=True)

                    with col_sim:
                        st.markdown("##### ⚡ จำลองใหม่ (Simulated)")
                        sim_sid = st.number_input(
                            "ระยะทาง SID จำลอง (cm)",
                            min_value=10.0,
                            max_value=300.0,
                            step=0.1,
                            format="%.1f",
                            key="s_sid",
                        )
                        sim_kvp = st.number_input(
                            "แรงดันไฟฟ้าจำลอง (kVp)",
                            min_value=20.0,
                            max_value=150.0,
                            step=0.1,
                            format="%.1f",
                            key="s_kvp",
                        )
                        sim_mas = st.number_input(
                            "กระแส-เวลาจำลอง (mAs)",
                            min_value=0.01,
                            max_value=2000.0,
                            step=0.01,
                            format="%.2f",
                            key="s_mas",
                        )
                        sim_angle = st.number_input(
                            "มุม Gantry จำลอง (°)",
                            min_value=0.0,
                            max_value=360.0,
                            step=0.1,
                            format="%.1f",
                            key="s_ang",
                        )

                    mas_ratio = sim_mas / max(base_mas, 1e-6)
                    kvp_ratio = (sim_kvp / max(base_kvp, 1.0)) ** 2
                    distance_factor = (base_sid / max(sim_sid, 1.0)) ** 2
                    effective_factor = mas_ratio * kvp_ratio * distance_factor
                    effective_dose_pct = effective_factor * 100.0
                    beam_dose_pct = mas_ratio * kvp_ratio * 100.0

                    st.caption(
                        f"📊 ปริมาณรังสีสุทธิที่ฉากรับภาพเทียบ Baseline: **{effective_dose_pct:.1f}%** "
                        f"(mAs: {mas_ratio * 100:.1f}%)"
                    )

                    st.divider()

                    baseline_slice = norm_volume[selected_slice]
                    simulated_slice = apply_physics_noise(
                        baseline_slice,
                        beam_dose_pct,
                        distance_cm=sim_sid,
                        ref_distance_cm=base_sid,
                    )

                    g_sim = compute_global_image_metrics(simulated_slice)
                    snr = g_sim["snr"]
                    cnr = g_sim["cnr"]

                    target_ei = 300.0
                    current_ei = target_ei * effective_factor
                    di = 10.0 * np.log10(effective_factor) if effective_factor > 0 else -99.0

                    st.subheader("📊 แดชบอร์ดคาดการณ์ผล (Predicted)")
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted EI", f"{current_ei:.1f}")
                    m2.metric("Predicted DI", f"{di:+.2f}")

                    m3, m4 = st.columns(2)
                    m3.metric("Predicted SNR", f"{snr:.2f}")
                    m4.metric("Predicted CNR", f"{cnr:.2f}")

                    st.divider()
                    st.markdown("### 🛡️ 1. ผลตรวจมาตรฐานความปลอดภัยทางรังสี")
                    if di > +3.0:
                        safety_status_pdf = ("FAIL", f"Severe Overexposure (DI = {di:+.2f} > +3.00 Action Limit)")
                        st.error("❌ **FAIL: Overexposure รุนแรง (DI > +3.0)**")
                    elif +1.0 < di <= +3.0:
                        safety_status_pdf = ("WARNING", f"Overexposure Caution (+1.00 < DI = {di:+.2f} <= +3.00)")
                        st.warning("⚠️ **WARNING: Overexposure (DI > +1.0)**")
                    elif -1.0 <= di <= +1.0:
                        safety_status_pdf = ("PASS", f"Within ALARA Target Range (-1.00 <= DI = {di:+.2f} <= +1.00)")
                        st.success("✅ **PASS: ปลอดภัยตามหลัก ALARA (-1.0 ≤ DI ≤ +1.0)**")
                    elif -3.0 <= di < -1.0:
                        safety_status_pdf = ("WARNING", f"Underexposure Caution (-3.00 <= DI = {di:+.2f} < -1.00)")
                        st.warning("⚠️ **WARNING: Underexposure (DI < -1.0)**")
                    else:
                        safety_status_pdf = ("FAIL", f"Severe Underexposure (DI = {di:+.2f} < -3.00 Action Limit)")
                        st.error("❌ **FAIL: Underexposure รุนแรง (DI < -3.0)**")

                    st.markdown("### 🔬 2. ผลตรวจความละเอียดและคุณภาพภาพวินิจฉัย")
                    if snr >= 5.0 and cnr >= 3.0:
                        quality_status_pdf = ("PASS", f"Diagnostic Quality Optimal (SNR = {snr:.2f} >= 5.0, CNR = {cnr:.2f} >= 3.0)")
                        st.success("✅ **PASS: คุณภาพภาพวินิจฉัยสมบูรณ์ (SNR ≥ 5.0, CNR ≥ 3.0)**")
                    elif snr >= 3.0 and cnr >= 1.5:
                        quality_status_pdf = ("WARNING", f"Diagnostic Quality Marginal (SNR = {snr:.2f}, CNR = {cnr:.2f})")
                        st.warning("⚠️ **WARNING: ความละเอียดภาพอยู่ในระดับเฝ้าระวัง**")
                    else:
                        quality_status_pdf = ("FAIL", f"Below Diagnostic Threshold (SNR = {snr:.2f}, CNR = {cnr:.2f})")
                        st.error("❌ **FAIL: คุณภาพภาพต่ำกว่าเกณฑ์อ่านโรค**")

                with col_display:
                    st.subheader("🖼️ หน้าจอแสดงผล DICOM 3D Slice Viewer")
                    view_mode = st.radio(
                        "เลือกโหมดมุมมอง:",
                        [
                            "เปรียบเทียบ Layer ปัจจุบัน (Side-by-Side)",
                            "จำลองเม็ดซ่าบน 3 ระนาบ (MPR: Axial/Coronal/Sagittal)",
                        ],
                        horizontal=True,
                    )

                    img_h, img_w = baseline_slice.shape
                    bg_offset_real = max(roi_size_real * 2, img_w // 4)
                    ensure_roi_state("m1_roi", img_w, img_h, roi_size_real, bg_offset_real)

                    if view_mode == "เปรียบเทียบ Layer ปัจจุบัน (Side-by-Side)":
                        render_interactive_roi_viewer(
                            prefix="m1_roi",
                            images=[baseline_slice, simulated_slice],
                            titles=[
                                f"1st Shot Baseline Slice {selected_slice + 1} ({base_mas:.1f} mAs, {base_kvp:.0f} kVp)",
                                f"Predicted Sim Slice {selected_slice + 1} ({sim_mas:.1f} mAs, {sim_kvp:.0f} kVp, {effective_dose_pct:.0f}%)",
                            ],
                            roi_size=roi_size_real,
                            show_roi=show_roi_overlay,
                            vmin=0,
                            vmax=255,
                        )
                    elif view_mode == "จำลองเม็ดซ่าบน 3 ระนาบ (MPR: Axial/Coronal/Sagittal)":
                        if num_slices > 1:
                            axial_base = norm_volume[selected_slice, :, :]
                            coronal_base = norm_volume[:, norm_volume.shape[1] // 2, :]
                            sagittal_base = norm_volume[:, :, norm_volume.shape[2] // 2]

                            axial_sim = apply_physics_noise(axial_base, beam_dose_pct, sim_sid, base_sid)
                            coronal_sim = apply_physics_noise(coronal_base, beam_dose_pct, sim_sid, base_sid)
                            sagittal_sim = apply_physics_noise(sagittal_base, beam_dose_pct, sim_sid, base_sid)

                            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                            ax[0].imshow(axial_sim, cmap="gray", vmin=0, vmax=255)
                            ax[0].set_title(f"Axial Cut ({sim_mas:.1f} mAs)", fontsize=10)
                            ax[0].axis("off")

                            ax[1].imshow(coronal_sim, cmap="gray", vmin=0, vmax=255)
                            ax[1].set_title(f"Coronal Cut ({sim_mas:.1f} mAs)", fontsize=10)
                            ax[1].axis("off")

                            ax[2].imshow(sagittal_sim, cmap="gray", vmin=0, vmax=255)
                            ax[2].set_title(f"Sagittal Cut ({sim_mas:.1f} mAs)", fontsize=10)
                            ax[2].axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.warning("การแสดงผล 3 ระนาบต้องการไฟล์ DICOM มากกว่า 1 Slice")

                    st.divider()
                    st.subheader("🎯 ค่าสถิติและพิกัดกรอบ ROI อิสระ (เทียบ ImageJ 100%)")
                    sig_cx, sig_cy, bg_cx, bg_cy = render_roi_coordinate_inputs(
                        "m1_roi", img_w, img_h, roi_size_real, bg_offset_real
                    )

                    roi_metrics, roi_info = analyze_image_qc(
                        baseline_slice,
                        roi_size=roi_size_real,
                        bg_offset_px=bg_offset_real,
                        sig_cx=sig_cx,
                        sig_cy=sig_cy,
                        bg_cx=bg_cx,
                        bg_cy=bg_cy,
                    )
                    sim_roi_metrics, _ = analyze_image_qc(
                        simulated_slice,
                        roi_size=roi_size_real,
                        bg_offset_px=bg_offset_real,
                        sig_cx=sig_cx,
                        sig_cy=sig_cy,
                        bg_cx=bg_cx,
                        bg_cy=bg_cy,
                    )

                    st.info(
                        f"📍 **พิกัดสำหรับกรอกใน ImageJ (`Edit → Selection → Specify...` ติ๊ก `Centered`)** | ขนาดภาพ `{img_w}×{img_h}` px\n\n"
                        f"- **🔴 กรอบสีแดง (Signal ROI):** `Width = {roi_info['roi_size']}`, `Height = {roi_info['roi_size']}`, "
                        f"`X Center = {roi_info['sig_cx']}`, `Y Center = {roi_info['sig_cy']}`\n"
                        f"- **🔵 กรอบสีฟ้า (Background ROI):** `Width = {roi_info['roi_size']}`, `Height = {roi_info['roi_size']}`, "
                        f"`X Center = {roi_info['bg_cx']}`, `Y Center = {roi_info['bg_cy']}`"
                    )
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Predicted ROI Mean", f"{sim_roi_metrics['Signal Mean']:.2f}")
                    r2.metric("Predicted ROI SD", f"{sim_roi_metrics['Signal SD']:.2f}")
                    r3.metric("Predicted ROI SNR", f"{sim_roi_metrics['SNR']:.2f}")
                    r4.metric("Predicted ROI CNR", f"{sim_roi_metrics['CNR']:.2f}")

                    st.divider()
                    st.subheader("🚀 ส่งค่าพารามิเตอร์ที่เลือกไปตรวจรับผลยิงจริงรอบที่ 2")
                    if st.button(
                        "📌 บันทึกค่าคาดการณ์นี้ส่งไปหน้าตรวจรับผลยิงรอบ 2 (Send to Post-Exposure QA)",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state["post_qa_plan"] = {
                            "patient_id": patient_id,
                            "modality": modality,
                            "slice_idx": selected_slice,
                            "num_slices": num_slices,
                            "roi_size": roi_size_real,
                            "bg_offset": bg_offset_real,
                            "sig_cx": int(roi_info["sig_cx"]),
                            "sig_cy": int(roi_info["sig_cy"]),
                            "bg_cx": int(roi_info["bg_cx"]),
                            "bg_cy": int(roi_info["bg_cy"]),
                            "base_kvp": base_kvp,
                            "base_mas": base_mas,
                            "base_sid": base_sid,
                            "base_angle": base_angle,
                            "plan_kvp": float(sim_kvp),
                            "plan_mas": float(sim_mas),
                            "plan_sid": float(sim_sid),
                            "plan_angle": float(sim_angle),
                            "plan_dose_pct": float(effective_dose_pct),
                            "pred_ei": float(current_ei),
                            "pred_di": float(di),
                            "pred_snr": float(snr),
                            "pred_cnr": float(cnr),
                            "pred_roi_mean": float(sim_roi_metrics["Signal Mean"]),
                            "pred_roi_sd": float(sim_roi_metrics["Signal SD"]),
                            "pred_roi_snr": float(sim_roi_metrics["SNR"]),
                            "pred_roi_cnr": float(sim_roi_metrics["CNR"]),
                            "baseline_slice": baseline_slice.copy(),
                            "simulated_slice": simulated_slice.copy(),
                        }
                        st.success(
                            "✅ **บันทึกค่าคาดการณ์เรียบร้อยแล้ว!** เมื่อนำค่านี้ไปตั้งที่เครื่องจริงและยิงซ้ำรอบที่ 2 เสร็จ "
                            "ให้เลือกเมนู **`🎯 3. ตรวจรับผลยิงจริงรอบ 2 (Post-Exposure QA)`** ทางแถบซ้ายเพื่ออัปโหลดไฟล์ยิงรอบ 2 มาตรวจสอบได้เลย"
                        )

                    snr_row_stat = "PASS" if snr >= 5.0 else ("WARNING" if snr >= 3.0 else "FAIL")
                    cnr_row_stat = "PASS" if cnr >= 3.0 else ("WARNING" if cnr >= 1.5 else "FAIL")
                    qc_pdf_bytes = generate_qc_report_pdf(
                        report_mode="Pre-Optimization Simulation & Dose Planning Audit",
                        study_info={
                            "Modality": modality,
                            "Patient / Study ID": patient_id,
                            "Evaluated Layer": f"Slice {selected_slice + 1} / {num_slices}",
                            "Matrix Size": f"{img_w} x {img_h} px",
                            "Net Receptor Dose": f"{effective_dose_pct:.1f}%",
                        },
                        params_rows=[
                            ["Tube Voltage (kVp)", f"{base_kvp:.1f} kVp", f"{sim_kvp:.1f} kVp", f"{sim_kvp - base_kvp:+.1f} kVp"],
                            ["Tube Current-Time (mAs)", f"{base_mas:.2f} mAs", f"{sim_mas:.2f} mAs", f"{(mas_ratio - 1.0) * 100:+.1f}%"],
                            ["Source-to-Image Distance (SID)", f"{base_sid:.1f} cm", f"{sim_sid:.1f} cm", f"{sim_sid - base_sid:+.1f} cm"],
                            ["Gantry Rotation Angle", f"{base_angle:.1f} deg", f"{sim_angle:.1f} deg", f"{sim_angle - base_angle:+.1f} deg"],
                        ],
                        qc_table_rows=[
                            ["Exposure Index (EI)", f"{current_ei:.1f}", "Target = 300.0 (IEC 62494-1)", "INFO"],
                            ["Deviation Index (DI)", f"{di:+.2f}", "-1.00 <= DI <= +1.00 (ALARA)", safety_status_pdf[0]],
                            ["Global Image SNR", f"{snr:.2f}", "SNR >= 5.00 (Diagnostic Min)", snr_row_stat],
                            ["Global Image CNR", f"{cnr:.2f}", "CNR >= 3.00 (Diagnostic Min)", cnr_row_stat],
                            ["Baseline ROI Mean Intensity", f"{roi_metrics['Signal Mean']:.2f} ADU", "1st Shot Reference Signal", "INFO"],
                            ["Baseline ROI Noise (SD, ddof=1)", f"{roi_metrics['Signal SD']:.2f} ADU", "1st Shot Sample SD (ImageJ)", "INFO"],
                            ["Predicted ROI Noise (SD)", f"{sim_roi_metrics['Signal SD']:.2f} ADU", "Simulated Quantum Noise", "INFO"],
                            ["Predicted ROI SNR", f"{sim_roi_metrics['SNR']:.2f}", "Target Post-Optimization SNR", "INFO"],
                        ],
                        safety_status=safety_status_pdf,
                        quality_status=quality_status_pdf,
                        audit_notes=[
                            f"Quantum mottle modeled via Poisson photon statistics scaled by effective dose factor ({effective_dose_pct:.1f}%) and inverse-square law.",
                            f"ImageJ cross-validation ROI (ddof=1) placed at Signal ({roi_info['sig_cx']}, {roi_info['sig_cy']}) and Background ({roi_info['bg_cx']}, {roi_info['bg_cy']}).",
                            "Deviation Index (DI) computed against target receptor EI = 300.0 per IEC 62494-1 / ICRP Publ. 103 ALARA guidelines.",
                        ],
                        img_left=baseline_slice,
                        img_left_title=f"Figure 1A. 1st Shot Baseline ({base_mas:.1f} mAs, {base_kvp:.0f} kVp)",
                        img_right=simulated_slice,
                        img_right_title=f"Figure 1B. Predicted Sim ({sim_mas:.1f} mAs, {sim_kvp:.0f} kVp)",
                        roi_info=roi_info if show_roi_overlay else None,
                        hist_mode=False,
                        institution_name=rpt_institution,
                        equipment_id=rpt_equipment,
                        operator_name=rpt_operator,
                        physicist_name=rpt_physicist,
                    )
                    st.download_button(
                        label="📥 ดาวน์โหลดรายงานคาดการณ์ก่อนยิงซ้ำ (PDF 1 หน้า A4)",
                        data=qc_pdf_bytes,
                        file_name=f"Pre_Optimization_Plan_{patient_id}_Slice{selected_slice + 1}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            else:
                st.warning("ไม่พบไฟล์ภาพ DICOM ที่ใช้งานได้ในสิ่งที่อัปโหลดเข้ามา")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")

    else:
        st.info(
            "👈 **วิธีใช้งาน:** ลากโฟลเดอร์ภาพ DICOM / DICOM-RT ที่ยิงรอบที่ 1 มาวางใส่ด้านบน "
            "เพื่อจำลองปรับลดรังสี แล้วกดส่งค่าคาดการณ์ไปยัง **โหมด 3 (ตรวจรับผลยิงจริงรอบ 2)**"
        )


# ==============================================================================
# โหมดที่ 2: ห้องปฏิบัติการจำลองเสมือนจริง (กรอกค่า kVp & mAs พร้อมปุ่ม +/- แบบเดียว)
# ==============================================================================
elif workspace_mode.startswith("🧪"):
    st.title("🧪 Virtual Radiography Laboratory (Customizable 2 or 3-Material Phantom)")
    st.caption(
        "เลือกได้ทั้งแบบ **ชุดจับคู่สำเร็จรูปตามตำแหน่งการตรวจ** หรือ **เลือกจำนวนวัสดุเอง (2 วัสดุตรงกลาง / 3 วัสดุซ้าย-ขวา)** "
        "พร้อมกรอกค่า **kVp และ mAs** ด้วยปุ่ม `-` / `+` และเลือกสีกรอบเพื่อคลิกกำหนดจุดบนภาพได้ทันที"
    )

    preset_names = list(PHANTOM_PRESETS.keys())
    selected_preset = st.selectbox(
        "🎯 เลือกหมวดชุดจับคู่สำเร็จรูป (Phantom & Clinical Region Presets):",
        preset_names,
        index=0,
        key="m2_cfg_preset",
    )

    if st.session_state.get("m2_last_preset") != selected_preset:
        st.session_state["m2_last_preset"] = selected_preset
        p_cfg = PHANTOM_PRESETS[selected_preset]
        if p_cfg is not None:
            st.session_state["m2_cfg_num_mat"] = (
                "3 วัสดุ (Multi-Insert: ตัวกลาง + ซ้าย + ขวา)"
                if p_cfg["num_mat"] == 3
                else "2 วัสดุ (Single-Insert: ตัวกลาง + วัตถุตรงกลาง)"
            )
            st.session_state["m2_cfg_bg"] = p_cfg["bg"]
            st.session_state["m2_cfg_ins1"] = p_cfg["ins1"]
            st.session_state["m2_cfg_ins2"] = p_cfg["ins2"]
            st.session_state["m2_cfg_thick"] = float(p_cfg["thick"])
            st.session_state["m2_cfg_kvp"] = float(p_cfg["kvp"])
            st.session_state["m2_cfg_mas"] = float(round((p_cfg["ma"] * p_cfg["ms"]) / 1000.0, 2))
            st.session_state["m2_cfg_sid"] = float(p_cfg["sid"])
            st.session_state["m2_cfg_grid"] = bool(p_cfg["grid"])
        for rk in ["m2_roi_shape", "m2_roi_last_click", "m2_roi_chart_ver"]:
            if rk in st.session_state:
                del st.session_state[rk]

    st.session_state.setdefault("m2_cfg_num_mat", "3 วัสดุ (Multi-Insert: ตัวกลาง + ซ้าย + ขวา)")
    st.session_state.setdefault("m2_cfg_bg", "Soft Tissue")
    st.session_state.setdefault("m2_cfg_ins1", "Cortical Bone")
    st.session_state.setdefault("m2_cfg_ins2", "Air")
    st.session_state.setdefault("m2_cfg_thick", 15.0)
    st.session_state.setdefault("m2_cfg_kvp", 100.0)
    st.session_state.setdefault("m2_cfg_mas", 112.50)
    st.session_state.setdefault("m2_cfg_sid", 100.0)
    st.session_state.setdefault("m2_cfg_grid", True)

    if PHANTOM_PRESETS[selected_preset] is not None:
        st.info(f"📋 **รายละเอียดชุดทดสอบ:** {PHANTOM_PRESETS[selected_preset]['desc']}")

    col_controls, col_workspace = st.columns([1.05, 1.45], gap="large")

    with col_controls:
        tab_phantom, tab_console, tab_sim = st.tabs(
            ["1️⃣ Phantom & Materials", "2️⃣ X-ray Console", "3️⃣ Noise & ROI Size"]
        )

        with tab_phantom:
            st.subheader("กำหนดโครงสร้างและจำนวนวัสดุในแฟนธอม")
            num_mat_choice = st.radio(
                "🧩 เลือกจำนวนวัสดุในแบบจำลอง:",
                [
                    "3 วัสดุ (Multi-Insert: ตัวกลาง + ซ้าย + ขวา)",
                    "2 วัสดุ (Single-Insert: ตัวกลาง + วัตถุตรงกลาง)",
                ],
                key="m2_cfg_num_mat",
            )
            num_materials = 3 if num_mat_choice.startswith("3 วัสดุ") else 2

            img_size = st.selectbox("Image Size (Pixels)", [256, 512, 1024], index=0, key="m2_cfg_imgsize")
            pixel_size_mm = st.number_input("Pixel Size (mm)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

            bg_opts = ["Soft Tissue", "Water", "PMMA", "Adipose (Fat)"]
            bg_material = st.selectbox(
                "🔵 1. ตัวกลางหลัก (Background Slab)",
                bg_opts,
                key="m2_cfg_bg",
            )
            st.caption(
                f"ρ = {MATERIALS_DB[bg_material]['density']} g/cm³ | "
                f"μ_ref = {MATERIALS_DB[bg_material]['mu_ref']} cm⁻¹ ({MATERIALS_DB[bg_material]['desc']})"
            )

            thickness_cm = st.slider(
                "Phantom Thickness (cm) — ความหนาอวัยวะ (Beer-Lambert Law)",
                min_value=1.0,
                max_value=30.0,
                step=1.0,
                key="m2_cfg_thick",
            )

            st.divider()
            ins1_opts = [
                "Cortical Bone",
                "Trabecular Bone",
                "PTFE (Teflon)",
                "Iodine (15 mg/mL)",
                "Aluminum",
                "Lung",
                "Air",
            ]
            ins1_label = (
                "🔴 2. วัตถุทดสอบฝั่งซ้าย (Insert 1: High-Z / Target)"
                if num_materials == 3
                else "🔴 2. วัตถุทดสอบตรงกลาง (Center Target Insert)"
            )
            ins1_material = st.selectbox(
                ins1_label,
                ins1_opts,
                key="m2_cfg_ins1",
            )
            st.caption(
                f"ρ = {MATERIALS_DB[ins1_material]['density']} g/cm³ | "
                f"μ_ref = {MATERIALS_DB[ins1_material]['mu_ref']} cm⁻¹ ({MATERIALS_DB[ins1_material]['desc']})"
            )

            if num_materials == 3:
                ins2_opts = ["Air", "Lung", "Adipose (Fat)", "HDPE", "Water", "PMMA"]
                ins2_material = st.selectbox(
                    "🟢 3. วัตถุทดสอบฝั่งขวา (Insert 2: Low-Density / Low-Contrast)",
                    ins2_opts,
                    key="m2_cfg_ins2",
                )
                st.caption(
                    f"ρ = {MATERIALS_DB[ins2_material]['density']} g/cm³ | "
                    f"μ_ref = {MATERIALS_DB[ins2_material]['mu_ref']} cm⁻¹ ({MATERIALS_DB[ins2_material]['desc']})"
                )
            else:
                ins2_material = st.session_state.get("m2_cfg_ins2", "Air")

            st.divider()
            obj_shape = st.selectbox("Object Shape (รูปร่างวัตถุทดสอบ)", ["Circle", "Square"])
            obj_diameter_mm = st.slider("Object Diameter / Width (mm)", 15.0, 65.0, 44.0, step=2.0)

        with tab_console:
            st.subheader("Virtual X-ray Console")
            st.caption("พิมพ์ตัวเลขหรือกดปุ่ม `-` / `+` เพื่อปรับค่าพารามิเตอร์หลักได้โดยตรง")

            c_k1, c_k2 = st.columns(2)
            with c_k1:
                kvp = st.number_input(
                    "แรงดันไฟฟ้า (kVp)",
                    min_value=40.0,
                    max_value=150.0,
                    step=1.0,
                    format="%.1f",
                    key="m2_cfg_kvp",
                )
            with c_k2:
                mas = st.number_input(
                    "กระแส-เวลา (mAs)",
                    min_value=0.1,
                    max_value=1000.0,
                    step=1.0,
                    format="%.2f",
                    key="m2_cfg_mas",
                )

            ma = 250.0 if mas >= 2.5 else 100.0
            exp_ms = max((mas * 1000.0) / ma, 1.0)
            st.caption(f"ℹ️ บันทึกลง DICOM Header อัตโนมัติเทียบเท่า **{ma:.0f} mA × {exp_ms:.0f} ms**")

            st.divider()
            sid_cm = st.number_input(
                "ระยะทาง SID: Source-to-Image Distance (cm)",
                min_value=50.0,
                max_value=200.0,
                step=5.0,
                format="%.1f",
                key="m2_cfg_sid",
            )

            grid_on = st.toggle("Anti-Scatter Grid (เปิดใช้เมื่อแฟนธอมหนา ≥ 10 cm)", key="m2_cfg_grid")

            with st.expander("⚙️ ตัวเลือกขั้นสูง (Relative Dose & Beam Filtration)", expanded=False):
                rel_dose_pct = st.number_input(
                    "Relative Dose (%) — ตัวคูณปรับปริมาณรังสี (ค่ามาตรฐาน 100%)",
                    min_value=10.0,
                    max_value=200.0,
                    value=100.0,
                    step=10.0,
                    format="%.0f",
                )
                filtration_mm = st.number_input(
                    "Beam Filtration (mm Al) — แผ่นกรองรังสีพลังงานต่ำ (ค่ามาตรฐาน 2.5 mm Al)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.5,
                    step=0.5,
                    format="%.1f",
                )

        with tab_sim:
            st.subheader("Noise Simulation & ROI Size")
            st.text_input("Noise Model", value="Poisson (Quantum Mottle)", disabled=True)
            photon_scale = st.slider("Photon Scale Factor", 0.2, 5.0, 1.0, step=0.2)
            seed = st.number_input("Random Seed (ล็อกผลสุ่มให้ทำซ้ำได้)", value=42, step=1)

            st.divider()
            st.markdown("**ตั้งค่าขนาดกรอบ ROI (คลิกย้ายบนภาพหรือกรอก X, Y ได้ทางขวา)**")
            roi_size = st.slider("ROI Size (Width × Height px)", min_value=10, max_value=60, value=20, step=2)

    sim_image, phys = simulate_phantom_radiograph(
        num_materials=num_materials,
        img_size=img_size,
        pixel_size_mm=pixel_size_mm,
        bg_material=bg_material,
        thickness_cm=thickness_cm,
        obj_shape=obj_shape,
        ins1_material=ins1_material,
        ins2_material=ins2_material,
        obj_diameter_mm=obj_diameter_mm,
        kvp=kvp,
        mas=mas,
        sid_cm=sid_cm,
        grid_on=grid_on,
        filtration_mm=filtration_mm,
        rel_dose_pct=rel_dose_pct,
        photon_scale=photon_scale,
        seed=int(seed),
    )

    is_3mat = num_materials == 3
    default_bg_offset_v = max(int(roi_size * 2), img_size // 4)
    ensure_roi_state("m2_roi", img_size, img_size, roi_size, default_bg_offset_v, has_third_roi=is_3mat)

    if is_3mat:
        roi_labels_mode2 = (
            f"🔴 กรอบสีแดง ({ins1_material})",
            f"🔵 กรอบสีฟ้า ({bg_material} Bg)",
            f"🟢 กรอบสีเขียว ({ins2_material})",
        )
        viewer_title = (
            f"3-Material Phantom: {ins1_material} (L) | {bg_material} (Bg) | {ins2_material} (R) "
            f"({kvp:.0f} kVp, {mas:.2f} mAs)"
        )
    else:
        roi_labels_mode2 = (
            f"🔴 กรอบสีแดง ({ins1_material})",
            f"🔵 กรอบสีฟ้า ({bg_material} Bg)",
            "",
        )
        viewer_title = (
            f"2-Material Phantom: {ins1_material} in {bg_material} ({thickness_cm:.0f} cm, {kvp:.0f} kVp, {mas:.2f} mAs)"
        )

    with col_workspace:
        view_tab1, view_tab2 = st.tabs(
            [
                f"🖼️ Image Preview & {num_materials}-Material QC",
                "📊 Histogram & Physics Summary",
            ]
        )

        with view_tab1:
            render_interactive_roi_viewer(
                prefix="m2_roi",
                images=[sim_image],
                titles=[viewer_title],
                roi_size=roi_size,
                show_roi=True,
                has_third_roi=is_3mat,
                roi_labels=roi_labels_mode2,
            )

            coords_out = render_roi_coordinate_inputs(
                "m2_roi",
                img_size,
                img_size,
                roi_size,
                default_bg_offset_v,
                has_third_roi=is_3mat,
                labels=roi_labels_mode2,
            )

            if is_3mat:
                sig_cx, sig_cy, bg_cx, bg_cy, ins2_cx, ins2_cy = coords_out
                metrics, roi_info = analyze_image_qc(
                    sim_image,
                    roi_size=roi_size,
                    bg_offset_px=default_bg_offset_v,
                    sig_cx=sig_cx,
                    sig_cy=sig_cy,
                    bg_cx=bg_cx,
                    bg_cy=bg_cy,
                    ins2_cx=ins2_cx,
                    ins2_cy=ins2_cy,
                )
            else:
                sig_cx, sig_cy, bg_cx, bg_cy = coords_out
                metrics, roi_info = analyze_image_qc(
                    sim_image,
                    roi_size=roi_size,
                    bg_offset_px=default_bg_offset_v,
                    sig_cx=sig_cx,
                    sig_cy=sig_cy,
                    bg_cx=bg_cx,
                    bg_cy=bg_cy,
                )

            v_di = phys["DI"]
            v_bg_snr = metrics["Background SNR"]
            v_sig_snr = metrics["SNR"]
            v_cnr1 = metrics["CNR"]
            v_cnr2 = metrics.get("Insert2 CNR", 0.0)

            if v_di > +3.0:
                v_safety_status = ("FAIL", f"Severe Overexposure (DI = {v_di:+.2f} > +3.00 Action Limit)")
            elif +1.0 < v_di <= +3.0:
                v_safety_status = ("WARNING", f"Overexposure Caution (+1.00 < DI = {v_di:+.2f} <= +3.00)")
            elif -1.0 <= v_di <= +1.0:
                v_safety_status = ("PASS", f"Within ALARA Target Range (-1.00 <= DI = {v_di:+.2f} <= +1.00)")
            elif -3.0 <= v_di < -1.0:
                v_safety_status = ("WARNING", f"Underexposure Caution (-3.00 <= DI = {v_di:+.2f} < -1.00)")
            else:
                v_safety_status = ("FAIL", f"Severe Underexposure (DI = {v_di:+.2f} < -3.00 Action Limit)")

            if is_3mat:
                if v_bg_snr >= 5.0 and v_cnr1 >= 3.0 and v_cnr2 >= 3.0:
                    v_quality_status = (
                        "PASS",
                        f"Optimal Multi-Tissue Contrast (Bg SNR={v_bg_snr:.1f}, CNR1={v_cnr1:.1f}, CNR2={v_cnr2:.1f})",
                    )
                elif v_bg_snr >= 3.0 and (v_cnr1 >= 1.5 or v_cnr2 >= 1.5):
                    v_quality_status = (
                        "WARNING",
                        f"Marginal Contrast/SNR (Bg SNR={v_bg_snr:.1f}, CNR1={v_cnr1:.1f}, CNR2={v_cnr2:.1f})",
                    )
                else:
                    v_quality_status = (
                        "FAIL",
                        f"Below Diagnostic Threshold (Bg SNR={v_bg_snr:.1f}, CNR1={v_cnr1:.1f}, CNR2={v_cnr2:.1f})",
                    )

                st.info(
                    f"📍 **พิกัดสำหรับกรอกใน ImageJ (`Edit → Selection → Specify...` ติ๊ก `Centered`, ขนาด `{roi_size}×{roi_size}` px)**\n\n"
                    f"- **🔴 กรอบสีแดง ({ins1_material}):** `X Center = {roi_info['sig_cx']}`, `Y Center = {roi_info['sig_cy']}`\n"
                    f"- **🔵 กรอบสีฟ้า ({bg_material}):** `X Center = {roi_info['bg_cx']}`, `Y Center = {roi_info['bg_cy']}`\n"
                    f"- **🟢 กรอบสีเขียว ({ins2_material}):** `X Center = {roi_info['ins2_cx']}`, `Y Center = {roi_info['ins2_cy']}`"
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ESD (mGy)", f"{phys['ESD_mGy']:.3f}")
                m2.metric("Exposure Index (EI)", f"{phys['EI']:.1f}")
                m3.metric("Deviation Index (DI)", f"{phys['DI']:+.2f}")
                m4.metric(f"Bg SNR ({bg_material})", f"{metrics['Background SNR']:.2f}")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric(f"🔴 {ins1_material} Mean", f"{metrics['Signal Mean']:.1f}")
                m6.metric(f"🔴 CNR ({ins1_material})", f"{metrics['CNR']:.2f}")
                m7.metric(f"🟢 {ins2_material} Mean", f"{metrics['Insert2 Mean']:.1f}")
                m8.metric(f"🟢 CNR ({ins2_material})", f"{metrics['Insert2 CNR']:.2f}")
            else:
                if v_sig_snr >= 5.0 and v_cnr1 >= 3.0:
                    v_quality_status = ("PASS", f"Diagnostic Quality Optimal (SNR = {v_sig_snr:.2f} >= 5.0, CNR = {v_cnr1:.2f} >= 3.0)")
                elif v_sig_snr >= 3.0 and v_cnr1 >= 1.5:
                    v_quality_status = ("WARNING", f"Diagnostic Quality Marginal (SNR = {v_sig_snr:.2f}, CNR = {v_cnr1:.2f})")
                else:
                    v_quality_status = ("FAIL", f"Below Diagnostic Threshold (SNR = {v_sig_snr:.2f}, CNR = {v_cnr1:.2f})")

                st.info(
                    f"📍 **พิกัดสำหรับกรอกใน ImageJ (`Edit → Selection → Specify...` ติ๊ก `Centered`, ขนาด `{roi_size}×{roi_size}` px)**\n\n"
                    f"- **🔴 กรอบสีแดง ({ins1_material}):** `X Center = {roi_info['sig_cx']}`, `Y Center = {roi_info['sig_cy']}`\n"
                    f"- **🔵 กรอบสีฟ้า ({bg_material}):** `X Center = {roi_info['bg_cx']}`, `Y Center = {roi_info['bg_cy']}`"
                )

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("ESD (mGy)", f"{phys['ESD_mGy']:.3f}")
                m2.metric("Exposure Index (EI)", f"{phys['EI']:.1f}")
                m3.metric("Deviation Index (DI)", f"{phys['DI']:+.2f}")
                m4.metric("ROI SNR", f"{metrics['SNR']:.2f}")
                m5.metric("ROI CNR", f"{metrics['CNR']:.2f}")

                m6, m7, m8, m9 = st.columns(4)
                m6.metric("Signal Mean (Obj)", f"{metrics['Signal Mean']:.1f}")
                m7.metric("Signal SD (ddof=1)", f"{metrics['Signal SD']:.1f}")
                m8.metric("Background Mean", f"{metrics['Background Mean']:.1f}")
                m9.metric("Background SD", f"{metrics['Background SD']:.1f}")

            dicom_bytes = create_dicom_bytes(
                image_uint16=sim_image,
                pixel_size_mm=pixel_size_mm,
                kvp=kvp,
                ma=ma,
                exp_ms=exp_ms,
                sid_cm=sid_cm,
                num_materials=num_materials,
                bg_material=bg_material,
                ins1_material=ins1_material,
                ins2_material=ins2_material,
                thickness_cm=thickness_cm,
            )

            if is_3mat:
                v_snr_stat = "PASS" if v_bg_snr >= 5.0 else ("WARNING" if v_bg_snr >= 3.0 else "FAIL")
                v_cnr1_stat = "PASS" if v_cnr1 >= 3.0 else ("WARNING" if v_cnr1 >= 1.5 else "FAIL")
                v_cnr2_stat = "PASS" if v_cnr2 >= 3.0 else ("WARNING" if v_cnr2 >= 1.5 else "FAIL")

                v_pdf_bytes = generate_qc_report_pdf(
                    report_mode="Virtual Lab (3-Material Multi-Insert Phantom QC)",
                    study_info={
                        "Background Slab": f"{bg_material} ({thickness_cm:.0f} cm)",
                        "Left / Right Insert": f"{ins1_material} / {ins2_material}",
                        "Insert Geometry": f"{obj_shape} ({obj_diameter_mm:.0f} mm)",
                        "Grid / Filter": f"{'Grid ON' if grid_on else 'No Grid'} / {filtration_mm:.1f} mm Al",
                        "Matrix / Seed": f"{img_size}x{img_size} / Seed {int(seed)}",
                    },
                    params_rows=[
                        ["Tube Voltage (kVp)", "100.0 kVp (Ref)", f"{kvp:.1f} kVp", f"mu_bg = {phys['mu_bg']:.3f} cm^-1"],
                        ["Exposure (mAs)", "112.50 mAs (Ref)", f"{mas:.2f} mAs", f"{ma:.0f} mA x {exp_ms:.0f} ms"],
                        ["Source-to-Image Distance (SID)", "100.0 cm (Ref)", f"{sid_cm:.1f} cm", f"InvSq = {(100.0 / sid_cm) ** 2:.2f}x"],
                        ["Relative Dose & Filtration", "100% / 2.5 mm Al", f"{rel_dose_pct:.0f}% / {filtration_mm:.1f} mm Al", f"N0 = {phys['N0_incident']:.0f} ph/px"],
                        ["Entrance Surface Dose (ESD)", "1.350 mGy (Ref)", f"{phys['ESD_mGy']:.4f} mGy", f"EI = {phys['EI']:.1f} (Target=250)"],
                    ],
                    qc_table_rows=[
                        ["Exposure Index (EI)", f"{phys['EI']:.1f}", "Target = 250.0 (Phantom Ref)", "INFO"],
                        ["Deviation Index (DI)", f"{phys['DI']:+.2f}", "-1.00 <= DI <= +1.00 (ALARA)", v_safety_status[0]],
                        [f"Background SNR ({bg_material})", f"{v_bg_snr:.2f}", "SNR >= 5.00 (Diagnostic Min)", v_snr_stat],
                        [f"Left Insert CNR ({ins1_material})", f"{v_cnr1:.2f}", "CNR >= 3.00 (Insert 1 Target)", v_cnr1_stat],
                        [f"Right Insert CNR ({ins2_material})", f"{v_cnr2:.2f}", "CNR >= 3.00 (Insert 2 Target)", v_cnr2_stat],
                        [f"Left Insert Mean / SD", f"{metrics['Signal Mean']:.1f} / {metrics['Signal SD']:.1f}", f"{ins1_material} ROI (ddof=1)", "INFO"],
                        [f"Right Insert Mean / SD", f"{metrics['Insert2 Mean']:.1f} / {metrics['Insert2 SD']:.1f}", f"{ins2_material} ROI (ddof=1)", "INFO"],
                        ["Entrance Surface Dose (ESD)", f"{phys['ESD_mGy']:.4f} mGy", "Calculated Incident Air Kerma", "INFO"],
                    ],
                    safety_status=v_safety_status,
                    quality_status=v_quality_status,
                    audit_notes=[
                        f"3-material Beer-Lambert simulation: {thickness_cm:.0f} cm {bg_material} (mu={phys['mu_bg']:.4f}) with {ins1_material} (mu={phys['mu_ins1']:.4f}) & {ins2_material} (mu={phys['mu_ins2']:.4f}).",
                        f"Quantum mottle generated via Poisson RNG (Seed={int(seed)}, Photon Scale={photon_scale:.1f}x) with {'Anti-Scatter Grid ON' if grid_on else 'No Grid'}.",
                        f"ImageJ ROIs ({roi_size}x{roi_size} px, ddof=1): Insert1=({roi_info['sig_cx']},{roi_info['sig_cy']}), Bg=({roi_info['bg_cx']},{roi_info['bg_cy']}), Insert2=({roi_info['ins2_cx']},{roi_info['ins2_cy']}).",
                    ],
                    img_left=sim_image,
                    img_left_title=f"Figure 1A. 3-Material Phantom ({kvp:.0f} kVp, {mas:.2f} mAs, SID {sid_cm:.0f} cm)",
                    roi_info=roi_info,
                    hist_mode=True,
                    institution_name=rpt_institution,
                    equipment_id=rpt_equipment,
                    operator_name=rpt_operator,
                    physicist_name=rpt_physicist,
                    roi_hist_labels=(ins1_material, bg_material, ins2_material),
                )
                dcm_fname = f"phantom3mat_{bg_material}_{ins1_material}_{ins2_material}_{int(thickness_cm)}cm_{int(kvp)}kVp.dcm"
                pdf_fname = f"QC_Report_3Mat_{bg_material}_{int(kvp)}kVp_{int(sid_cm)}SID.pdf"
            else:
                v_snr_stat = "PASS" if v_sig_snr >= 5.0 else ("WARNING" if v_sig_snr >= 3.0 else "FAIL")
                v_cnr_stat = "PASS" if v_cnr1 >= 3.0 else ("WARNING" if v_cnr1 >= 1.5 else "FAIL")

                v_pdf_bytes = generate_qc_report_pdf(
                    report_mode="Virtual Lab (2-Material Single-Insert Phantom QC)",
                    study_info={
                        "Background Medium": f"{bg_material} ({thickness_cm:.0f} cm)",
                        "Center Target Insert": f"{ins1_material} ({obj_shape}, {obj_diameter_mm:.0f} mm)",
                        "Scatter / Filtration": f"{'Grid ON' if grid_on else 'No Grid'} / {filtration_mm:.1f} mm Al",
                        "Matrix / Pixel Size": f"{img_size}x{img_size} / {pixel_size_mm} mm",
                        "RNG Seed / Scale": f"Seed {int(seed)} / {photon_scale:.1f}x",
                    },
                    params_rows=[
                        ["Tube Voltage (kVp)", "100.0 kVp (Ref)", f"{kvp:.1f} kVp", f"mu_bg = {phys['mu_bg']:.3f} cm^-1"],
                        ["Exposure (mAs)", "96.00 mAs (Ref)", f"{mas:.2f} mAs", f"{ma:.0f} mA x {exp_ms:.0f} ms"],
                        ["Source-to-Image Distance (SID)", "100.0 cm (Ref)", f"{sid_cm:.1f} cm", f"InvSq = {(100.0 / sid_cm) ** 2:.2f}x"],
                        ["Relative Dose & Filtration", "100% / 2.5 mm Al", f"{rel_dose_pct:.0f}% / {filtration_mm:.1f} mm Al", f"N0 = {phys['N0_incident']:.0f} ph/px"],
                        ["Entrance Surface Dose (ESD)", "1.152 mGy (Ref)", f"{phys['ESD_mGy']:.4f} mGy", f"EI = {phys['EI']:.1f} (Target=250)"],
                    ],
                    qc_table_rows=[
                        ["Exposure Index (EI)", f"{phys['EI']:.1f}", "Target = 250.0 (Phantom Ref)", "INFO"],
                        ["Deviation Index (DI)", f"{phys['DI']:+.2f}", "-1.00 <= DI <= +1.00 (ALARA)", v_safety_status[0]],
                        ["ROI Signal-to-Noise (SNR)", f"{metrics['SNR']:.2f}", "SNR >= 5.00 (Diagnostic Min)", v_snr_stat],
                        ["ROI Contrast-to-Noise (CNR)", f"{metrics['CNR']:.2f}", "CNR >= 3.00 (Diagnostic Min)", v_cnr_stat],
                        ["Target ROI Mean / SD (ddof=1)", f"{metrics['Signal Mean']:.2f} / {metrics['Signal SD']:.2f}", f"{ins1_material} ROI", "INFO"],
                        ["Background Mean / SD (ddof=1)", f"{metrics['Background Mean']:.2f} / {metrics['Background SD']:.2f}", f"{bg_material} ROI", "INFO"],
                        ["Entrance Surface Dose (ESD)", f"{phys['ESD_mGy']:.4f} mGy", "Calculated Incident Air Kerma", "INFO"],
                    ],
                    safety_status=v_safety_status,
                    quality_status=v_quality_status,
                    audit_notes=[
                        f"2-material Beer-Lambert simulation: {thickness_cm:.0f} cm {bg_material} (mu={phys['mu_bg']:.4f}) with center {ins1_material} (mu={phys['mu_ins1']:.4f}).",
                        f"Quantum mottle generated via Poisson RNG (Seed={int(seed)}, Photon Scale={photon_scale:.1f}x) with {'Anti-Scatter Grid ON' if grid_on else 'No Grid'}.",
                        f"ImageJ ROIs ({roi_size}x{roi_size} px, ddof=1): Signal=({roi_info['sig_cx']},{roi_info['sig_cy']}), Background=({roi_info['bg_cx']},{roi_info['bg_cy']}).",
                    ],
                    img_left=sim_image,
                    img_left_title=f"Figure 1A. 2-Material Phantom ({kvp:.0f} kVp, {mas:.2f} mAs, SID {sid_cm:.0f} cm)",
                    roi_info=roi_info,
                    hist_mode=True,
                    institution_name=rpt_institution,
                    equipment_id=rpt_equipment,
                    operator_name=rpt_operator,
                    physicist_name=rpt_physicist,
                    roi_hist_labels=(ins1_material, bg_material),
                )
                dcm_fname = f"phantom2mat_{bg_material}_{ins1_material}_{int(thickness_cm)}cm_{int(kvp)}kVp.dcm"
                pdf_fname = f"QC_Report_2Mat_{bg_material}_{int(kvp)}kVp_{int(sid_cm)}SID.pdf"

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label=f"💾 Download {num_materials}-Material DICOM (.dcm)",
                    data=dicom_bytes,
                    file_name=dcm_fname,
                    mime="application/dicom",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    label="📄 Download Official QC & Dose Report (PDF 1 หน้า A4)",
                    data=v_pdf_bytes,
                    file_name=pdf_fname,
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )

        with view_tab2:
            st.subheader(f"Pixel Intensity Histogram ({num_materials} Materials Comparison)")
            fig_hist, ax_hist = plt.subplots(figsize=(6.5, 3.4))
            ax_hist.hist(
                roi_info["sig_roi"].ravel(),
                bins=25,
                alpha=0.65,
                color="#FF4B4B",
                label=f"{ins1_material} (Mean={metrics['Signal Mean']:.1f})",
            )
            ax_hist.hist(
                roi_info["bg_roi"].ravel(),
                bins=25,
                alpha=0.65,
                color="#00A6D6",
                label=f"{bg_material} Bg (Mean={metrics['Background Mean']:.1f})",
            )
            if is_3mat and "ins2_roi" in roi_info:
                ax_hist.hist(
                    roi_info["ins2_roi"].ravel(),
                    bins=25,
                    alpha=0.65,
                    color="#10B981",
                    label=f"{ins2_material} (Mean={metrics['Insert2 Mean']:.1f})",
                )
            ax_hist.set_xlabel("Pixel Intensity (ADU)")
            ax_hist.set_ylabel("Pixel Count")
            ax_hist.legend()
            st.pyplot(fig_hist)
            plt.close(fig_hist)

            st.subheader(f"สรุปค่าตัวแปรทางฟิสิกส์ของวัสดุทั้ง {num_materials} ชนิด")
            if is_3mat:
                st.table(
                    {
                        "พารามิเตอร์ทางฟิสิกส์": [
                            "Incident Photons (N₀)",
                            f"Linear Attenuation μ — {bg_material} (ตัวกลางหลัก @ {kvp:.0f} kVp)",
                            f"Linear Attenuation μ — {ins1_material} (ฝั่งซ้าย @ {kvp:.0f} kVp)",
                            f"Linear Attenuation μ — {ins2_material} (ฝั่งขวา @ {kvp:.0f} kVp)",
                            "Entrance Surface Dose (ESD)",
                            "Exposure Index (EI) / Deviation Index (DI)",
                        ],
                        "ค่าที่คำนวณได้": [
                            f"{phys['N0_incident']:.2f} photons/px",
                            f"{phys['mu_bg']:.4f} cm⁻¹",
                            f"{phys['mu_ins1']:.4f} cm⁻¹",
                            f"{phys['mu_ins2']:.4f} cm⁻¹",
                            f"{phys['ESD_mGy']:.4f} mGy",
                            f"{phys['EI']:.2f} (DI = {phys['DI']:+.2f})",
                        ],
                    }
                )
            else:
                st.table(
                    {
                        "พารามิเตอร์ทางฟิสิกส์": [
                            "Incident Photons (N₀)",
                            f"Linear Attenuation μ — {bg_material} (ตัวกลางหลัก @ {kvp:.0f} kVp)",
                            f"Linear Attenuation μ — {ins1_material} (วัตถุตรงกลาง @ {kvp:.0f} kVp)",
                            "Entrance Surface Dose (ESD)",
                            "Exposure Index (EI) / Deviation Index (DI)",
                        ],
                        "ค่าที่คำนวณได้": [
                            f"{phys['N0_incident']:.2f} photons/px",
                            f"{phys['mu_bg']:.4f} cm⁻¹",
                            f"{phys['mu_ins1']:.4f} cm⁻¹",
                            f"{phys['ESD_mGy']:.4f} mGy",
                            f"{phys['EI']:.2f} (DI = {phys['DI']:+.2f})",
                        ],
                    }
                )


# ==============================================================================
# โหมดที่ 3: ตรวจรับผลยิงจริงรอบที่ 2 (Post-Exposure Verification & Equipment QA)
# ==============================================================================
else:
    st.title("🎯 Post-Exposure Verification & Equipment QA (ตรวจรับผลยิงจริงรอบ 2)")
    st.write(
        "เปรียบเทียบ **ภาพและค่าที่ยิงได้จริงจากเครื่องเอกซเรย์รอบที่ 2** กับ **ค่าที่แบบจำลองคาดการณ์ไว้** "
        "เพื่อยืนยันความปลอดภัยผู้ป่วยและตรวจจับความผิดปกติของหลอดเอกซเรย์, ชุดกำเนิดไฟฟ้าแรงสูง, AEC และฉากรับภาพ"
    )

    has_saved_plan = "post_qa_plan" in st.session_state

    if has_saved_plan:
        use_saved_plan = st.checkbox(
            f"✅ ใช้ข้อมูลยิงรอบที่ 1 และแผนคาดการณ์ที่ส่งมาจากโหมดที่ 1 (Patient ID: {st.session_state['post_qa_plan']['patient_id']})",
            value=True,
        )
    else:
        use_saved_plan = False
        st.info(
            "💡 **ยังไม่มีค่าคาดการณ์ที่ส่งมาจากโหมดที่ 1** — คุณสามารถอัปโหลดทั้งไฟล์ยิงรอบที่ 1 (Baseline) และรอบที่ 2 (Actual) ด้านล่างนี้ได้โดยตรง"
        )

    up_col1, up_col2 = st.columns(2)
    shot1_data = None
    if not use_saved_plan:
        with up_col1:
            files_shot1 = st.file_uploader(
                "📥 1. อัปโหลดไฟล์ DICOM ยิงรอบที่ 1 (1st Shot Baseline)",
                type=None,
                accept_multiple_files=True,
                key=f"qa_shot1_{st.session_state.uploader_key}",
            )
            if files_shot1:
                shot1_data = load_dicom_series(files_shot1)
    else:
        with up_col1:
            p_saved = st.session_state["post_qa_plan"]
            st.success(
                f"📌 **โหลดข้อมูลรอบที่ 1 & ค่าคาดการณ์สำเร็จ**\n\n"
                f"- **1st Shot:** {p_saved['base_kvp']:.0f} kVp, {p_saved['base_mas']:.2f} mAs, SID {p_saved['base_sid']:.0f} cm\n"
                f"- **แผนปรับค่า:** {p_saved['plan_kvp']:.0f} kVp, {p_saved['plan_mas']:.2f} mAs, SID {p_saved['plan_sid']:.0f} cm ({p_saved['plan_dose_pct']:.1f}%)\n"
                f"- **ค่าคาดการณ์:** EI = `{p_saved['pred_ei']:.1f}`, DI = `{p_saved['pred_di']:+.2f}`, SNR = `{p_saved['pred_snr']:.2f}`, CNR = `{p_saved['pred_cnr']:.2f}`"
            )

    with up_col2:
        files_shot2 = st.file_uploader(
            "📥 2. อัปโหลดไฟล์ DICOM ยิงจริงรอบที่ 2 (2nd Shot Actual Re-Exposure)",
            type=None,
            accept_multiple_files=True,
            key=f"qa_shot2_{st.session_state.uploader_key}",
        )

    ready_shot1 = use_saved_plan or (shot1_data is not None)
    shot2_data = load_dicom_series(files_shot2) if files_shot2 else None

    if ready_shot1 and shot2_data is not None:
        st.divider()

        if use_saved_plan:
            plan = st.session_state["post_qa_plan"]
            patient_id = plan["patient_id"]
            modality = plan["modality"]
            base_kvp = plan["base_kvp"]
            base_mas = plan["base_mas"]
            base_sid = plan["base_sid"]
            base_angle = plan["base_angle"]

            plan_kvp = plan["plan_kvp"]
            plan_mas = plan["plan_mas"]
            plan_sid = plan["plan_sid"]
            plan_angle = plan["plan_angle"]
            plan_dose_pct = plan["plan_dose_pct"]

            pred_ei = plan["pred_ei"]
            pred_di = plan["pred_di"]
            pred_snr = plan["pred_snr"]
            pred_cnr = plan["pred_cnr"]

            slice_1_norm = plan["baseline_slice"]
            slice_pred_norm = plan["simulated_slice"]
            roi_size_qa = plan["roi_size"]
            bg_offset_qa = plan["bg_offset"]
            saved_coords = (
                plan.get("sig_cx", slice_1_norm.shape[1] // 2),
                plan.get("sig_cy", slice_1_norm.shape[0] // 2),
                plan.get("bg_cx", slice_1_norm.shape[1] // 2 + bg_offset_qa),
                plan.get("bg_cy", slice_1_norm.shape[0] // 2),
            )
            default_slice_idx = min(plan["slice_idx"], shot2_data["num_slices"] - 1)
        else:
            patient_id = shot1_data["patient_id"]
            modality = shot1_data["modality"]
            base_kvp = shot1_data["kvp"]
            base_mas = shot1_data["mas"]
            base_sid = shot1_data["sid"]
            base_angle = shot1_data["angle"]
            default_slice_idx = shot1_data["num_slices"] // 2

            st.subheader("🎛️ ระบุค่าพารามิเตอร์ที่ตั้งใจปรับไปยิงรอบที่ 2 (Planned Protocol)")
            pc1, pc2, pc3, pc4 = st.columns(4)
            plan_kvp = pc1.number_input("Planned kVp", min_value=20.0, max_value=150.0, value=float(shot2_data["kvp"]), step=0.1, format="%.1f")
            plan_mas = pc2.number_input("Planned mAs", min_value=0.01, max_value=2000.0, value=float(shot2_data["mas"]), step=0.01, format="%.2f")
            plan_sid = pc3.number_input("Planned SID (cm)", min_value=10.0, max_value=300.0, value=float(shot2_data["sid"]), step=0.1, format="%.1f")
            roi_size_qa = pc4.slider("ขนาด ROI (px)", 10, 80, 30, step=2)
            plan_angle = float(shot2_data["angle"])

            s1_idx = min(default_slice_idx, shot1_data["num_slices"] - 1)
            slice_1_norm = shot1_data["norm_volume"][s1_idx]
            bg_offset_qa = max(roi_size_qa * 2, slice_1_norm.shape[1] // 4)
            saved_coords = None

            mas_r = plan_mas / max(base_mas, 1e-6)
            kvp_r = (plan_kvp / max(base_kvp, 1.0)) ** 2
            dist_f = (base_sid / max(plan_sid, 1.0)) ** 2
            eff_f = mas_r * kvp_r * dist_f
            plan_dose_pct = eff_f * 100.0
            beam_pct = mas_r * kvp_r * 100.0

            slice_pred_norm = apply_physics_noise(slice_1_norm, beam_pct, plan_sid, base_sid)
            g_p = compute_global_image_metrics(slice_pred_norm)

            pred_ei = 300.0 * eff_f
            pred_di = 10.0 * np.log10(max(eff_f, 1e-6))
            pred_snr = g_p["snr"]
            pred_cnr = g_p["cnr"]

        if shot2_data["num_slices"] > 1:
            sel_s2 = st.slider(
                "🔍 เลือก Layer ของภาพยิงจริงรอบที่ 2 เพื่อเทียบให้ตรงตำแหน่งเดิม",
                0,
                shot2_data["num_slices"] - 1,
                min(default_slice_idx, shot2_data["num_slices"] - 1),
            )
        else:
            sel_s2 = 0

        slice_2_actual_norm = shot2_data["norm_volume"][sel_s2]
        h2, w2 = slice_2_actual_norm.shape

        if slice_2_actual_norm.shape != slice_1_norm.shape:
            st.warning(
                f"⚠️ ขนาด Matrix ของรอบที่ 2 ({w2}×{h2}) "
                f"ต่างจากรอบที่ 1 ({slice_1_norm.shape[1]}×{slice_1_norm.shape[0]}) ระบบจะปรับพิกัดให้อยู่ภายในขอบเขตของภาพ"
            )

        ensure_roi_state("m3_roi", w2, h2, roi_size_qa, bg_offset_qa, init_coords=saved_coords)

        st.subheader("🖼️ 1. ภาพเปรียบเทียบ 3 ระยะ & เลือกสีกรอบคลิกกำหนดจุด ROI บนภาพ")
        render_interactive_roi_viewer(
            prefix="m3_roi",
            images=[slice_1_norm, slice_pred_norm, slice_2_actual_norm],
            titles=[
                f"1st Shot Baseline ({base_mas:.1f} mAs, {base_kvp:.0f} kVp)",
                f"Predicted Sim ({plan_mas:.1f} mAs, {plan_kvp:.0f} kVp)",
                f"2nd Shot Actual ({float(shot2_data['mas']):.1f} mAs, {float(shot2_data['kvp']):.0f} kVp)",
            ],
            roi_size=roi_size_qa,
            show_roi=True,
            vmin=0,
            vmax=255,
        )

        qa_sig_cx, qa_sig_cy, qa_bg_cx, qa_bg_cy = render_roi_coordinate_inputs(
            "m3_roi", w2, h2, roi_size_qa, bg_offset_qa
        )

        act_kvp = float(shot2_data["kvp"])
        act_mas = float(shot2_data["mas"])
        act_sid = float(shot2_data["sid"])
        act_angle = float(shot2_data["angle"])

        g_act = compute_global_image_metrics(slice_2_actual_norm)
        act_snr = g_act["snr"]
        act_cnr = g_act["cnr"]

        r_act, roi_info_act = analyze_image_qc(
            slice_2_actual_norm,
            roi_size=roi_size_qa,
            bg_offset_px=bg_offset_qa,
            sig_cx=qa_sig_cx,
            sig_cy=qa_sig_cy,
            bg_cx=qa_bg_cx,
            bg_cy=qa_bg_cy,
        )
        r_pred, _ = analyze_image_qc(
            slice_pred_norm,
            roi_size=roi_size_qa,
            bg_offset_px=bg_offset_qa,
            sig_cx=qa_sig_cx,
            sig_cy=qa_sig_cy,
            bg_cx=qa_bg_cx,
            bg_cy=qa_bg_cy,
        )
        r_base, _ = analyze_image_qc(
            slice_1_norm,
            roi_size=roi_size_qa,
            bg_offset_px=bg_offset_qa,
            sig_cx=qa_sig_cx,
            sig_cy=qa_sig_cy,
            bg_cx=qa_bg_cx,
            bg_cy=qa_bg_cy,
        )

        pred_roi_mean = r_pred["Signal Mean"]
        pred_roi_sd = r_pred["Signal SD"]
        pred_roi_snr = r_pred["SNR"]
        pred_roi_cnr = r_pred["CNR"]

        if shot2_data["tag_ei"] is not None:
            act_ei = shot2_data["tag_ei"]
            ei_source_note = "DICOM Tag (0018,1411)"
        else:
            act_eff_factor = (
                (act_mas / max(base_mas, 1e-6))
                * ((act_kvp / max(base_kvp, 1.0)) ** 2)
                * ((base_sid / max(act_sid, 1.0)) ** 2)
            )
            sig_scale = r_act["Signal Mean"] / max(r_base["Signal Mean"], 1e-3)
            act_ei = 300.0 * act_eff_factor * (0.5 + 0.5 * sig_scale)
            ei_source_note = "Derived from Output & Receptor Signal"

        if shot2_data["tag_di"] is not None:
            act_di = shot2_data["tag_di"]
        else:
            act_di = 10.0 * np.log10(max(act_ei, 1e-6) / 300.0)

        act_dose_pct = (act_ei / 300.0) * 100.0

        diag_findings, diag_summary = diagnose_post_exposure_qa(
            plan_kvp=plan_kvp,
            plan_mas=plan_mas,
            plan_sid=plan_sid,
            act_kvp=act_kvp,
            act_mas=act_mas,
            act_sid=act_sid,
            pred_di=pred_di,
            act_di=act_di,
            pred_snr=pred_snr,
            act_snr=act_snr,
            pred_cnr=pred_cnr,
            act_cnr=act_cnr,
            pred_noise_sd=pred_roi_sd,
            act_noise_sd=r_act["Signal SD"],
        )

        st.divider()
        st.subheader("📋 2. สรุปผลการตรวจรับรอบที่ 2 และวินิจฉัยสถานะเครื่องเอกซเรย์")
        if diag_summary["severity"] == "PASS":
            st.success(
                f"✅ **VERIFIED (PASS): ผลยิงจริงรอบที่ 2 ตรงตามที่ระบบคาดการณ์ไว้!**\n\n"
                f"ค่าเบี่ยงเบนดัชนีรังสี `ΔDI = {diag_summary['delta_di']:+.2f}` (อยู่ในเกณฑ์ `±0.50`) | "
                f"ความคลาดเคลื่อน `SNR = {diag_summary['snr_err_pct']:+.1f}%` | `CNR = {diag_summary['cnr_err_pct']:+.1f}%` — "
                f"ทั้งโปรโตคอลใหม่, หลอดเอกซเรย์ และฉากรับภาพทำงานสอดคล้องกันอย่างสมบูรณ์"
            )
        elif diag_summary["severity"] == "WARNING":
            st.warning(
                f"⚠️ **CONDITIONAL (WARNING): พบความคลาดเคลื่อนระดับเฝ้าระวังระหว่างผลคาดการณ์กับผลยิงจริง**\n\n"
                f"`ΔDI = {diag_summary['delta_di']:+.2f}` | `SNR Error = {diag_summary['snr_err_pct']:+.1f}%` | "
                f"`Noise Error = {diag_summary['noise_err_pct']:+.1f}%` — กรุณาตรวจสอบรายการวินิจฉัยอุปกรณ์ด้านล่าง"
            )
        else:
            st.error(
                f"❌ **REJECTED / HARDWARE ALERT (FAIL): ผลยิงจริงเบี่ยงเบนจากแบบจำลองเกินเกณฑ์มาตรฐาน!**\n\n"
                f"`ΔDI = {diag_summary['delta_di']:+.2f}` | `SNR Error = {diag_summary['snr_err_pct']:+.1f}%` | "
                f"`Noise Error = {diag_summary['noise_err_pct']:+.1f}%` — พบข้อบ่งชี้ความผิดปกติของตัวเครื่อง/หัวยิง/ฉากรับภาพ"
            )

        for item in diag_findings:
            if item["status"] == "PASS":
                st.info(f"🟢 **{item['subsystem']}**: {item['detail']}")
            elif item["status"] in ("WARNING", "MISMATCH"):
                st.warning(f"🟡 **{item['subsystem']}**: {item['detail']}")
            else:
                st.error(f"🔴 **{item['subsystem']}**: {item['detail']}")

        st.divider()
        st.subheader("📊 3. ตารางเปรียบเทียบค่าพารามิเตอร์และตัวชี้วัดคุณภาพ (Predicted vs. Actual 2nd Shot)")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            "Actual Exposure Index (EI)",
            f"{act_ei:.1f}",
            delta=f"{act_ei - pred_ei:+.1f} vs Pred ({pred_ei:.1f})",
        )
        k2.metric(
            "Actual Deviation Index (DI)",
            f"{act_di:+.2f}",
            delta=f"ΔDI {diag_summary['delta_di']:+.2f} vs Pred ({pred_di:+.2f})",
            delta_color="inverse" if abs(diag_summary["delta_di"]) > 0.5 else "normal",
        )
        k3.metric(
            "Actual Global SNR",
            f"{act_snr:.2f}",
            delta=f"{diag_summary['snr_err_pct']:+.1f}% vs Pred ({pred_snr:.2f})",
        )
        k4.metric(
            "Actual Global CNR",
            f"{act_cnr:.2f}",
            delta=f"{diag_summary['cnr_err_pct']:+.1f}% vs Pred ({pred_cnr:.2f})",
        )

        st.table(
            {
                "รายการตรวจสอบ (Metric / Parameter)": [
                    "Tube Voltage (kVp)",
                    "Tube Current-Time (mAs)",
                    "Source-to-Image Distance (SID)",
                    "Net Receptor Dose (%)",
                    f"Exposure Index (EI) [{ei_source_note}]",
                    "Deviation Index (DI)",
                    "Global Image SNR",
                    "Global Image CNR",
                    f"ROI Signal Mean @ ({qa_sig_cx}, {qa_sig_cy})",
                    "ROI Noise SD (ddof=1, ImageJ)",
                    "ROI SNR",
                    "ROI CNR",
                ],
                "1️⃣ ยิงรอบที่ 1 (Baseline)": [
                    f"{base_kvp:.1f} kVp",
                    f"{base_mas:.2f} mAs",
                    f"{base_sid:.1f} cm",
                    "100.0%",
                    "300.0",
                    "+0.00",
                    f"{compute_global_image_metrics(slice_1_norm)['snr']:.2f}",
                    f"{compute_global_image_metrics(slice_1_norm)['cnr']:.2f}",
                    f"{r_base['Signal Mean']:.2f}",
                    f"{r_base['Signal SD']:.2f}",
                    f"{r_base['SNR']:.2f}",
                    f"{r_base['CNR']:.2f}",
                ],
                "2️⃣ ค่าที่ระบบคาดการณ์ (Predicted)": [
                    f"{plan_kvp:.1f} kVp",
                    f"{plan_mas:.2f} mAs",
                    f"{plan_sid:.1f} cm",
                    f"{plan_dose_pct:.1f}%",
                    f"{pred_ei:.1f}",
                    f"{pred_di:+.2f}",
                    f"{pred_snr:.2f}",
                    f"{pred_cnr:.2f}",
                    f"{pred_roi_mean:.2f}",
                    f"{pred_roi_sd:.2f}",
                    f"{pred_roi_snr:.2f}",
                    f"{pred_roi_cnr:.2f}",
                ],
                "3️⃣ ยิงจริงรอบที่ 2 (Actual)": [
                    f"{act_kvp:.1f} kVp",
                    f"{act_mas:.2f} mAs",
                    f"{act_sid:.1f} cm",
                    f"{act_dose_pct:.1f}%",
                    f"{act_ei:.1f}",
                    f"{act_di:+.2f}",
                    f"{act_snr:.2f}",
                    f"{act_cnr:.2f}",
                    f"{r_act['Signal Mean']:.2f}",
                    f"{r_act['Signal SD']:.2f}",
                    f"{r_act['SNR']:.2f}",
                    f"{r_act['CNR']:.2f}",
                ],
                "ความคลาดเคลื่อน (Actual vs Predicted)": [
                    f"{diag_summary['kvp_diff']:+.1f} kVp",
                    f"{diag_summary['mas_err_pct']:+.1f}%",
                    f"{diag_summary['sid_diff']:+.1f} cm",
                    f"{act_dose_pct - plan_dose_pct:+.1f}%",
                    f"{act_ei - pred_ei:+.1f}",
                    f"ΔDI = {diag_summary['delta_di']:+.2f}",
                    f"{diag_summary['snr_err_pct']:+.1f}%",
                    f"{diag_summary['cnr_err_pct']:+.1f}%",
                    f"{((r_act['Signal Mean'] - pred_roi_mean) / max(pred_roi_mean, 1e-3)) * 100:+.1f}%",
                    f"{diag_summary['noise_err_pct']:+.1f}%",
                    f"{((r_act['SNR'] - pred_roi_snr) / max(pred_roi_snr, 1e-3)) * 100:+.1f}%",
                    f"{((r_act['CNR'] - pred_roi_cnr) / max(pred_roi_cnr, 1e-3)) * 100:+.1f}%",
                ],
            }
        )

        fig_bar, ax_bar = plt.subplots(1, 2, figsize=(11, 3.3))
        labels_m = ["Global SNR", "Global CNR", "ROI SNR", "ROI CNR"]
        pred_vals = [pred_snr, pred_cnr, pred_roi_snr, pred_roi_cnr]
        act_vals = [act_snr, act_cnr, r_act["SNR"], r_act["CNR"]]
        x_pos = np.arange(len(labels_m))
        w_bar = 0.34

        ax_bar[0].bar(x_pos - w_bar / 2, pred_vals, width=w_bar, color="#334E68", label="Predicted (Simulation)")
        ax_bar[0].bar(x_pos + w_bar / 2, act_vals, width=w_bar, color="#0284C7", label="Actual (2nd Shot)")
        ax_bar[0].set_xticks(x_pos)
        ax_bar[0].set_xticklabels(labels_m, fontsize=8.5)
        ax_bar[0].set_title("Image Quality Metrics: Predicted vs. Actual", fontsize=9.5, fontweight="bold")
        ax_bar[0].grid(axis="y", alpha=0.25, linestyle="--")
        ax_bar[0].legend(fontsize=8)

        labels_n = ["ROI Noise SD (ADU)", "Exposure Index (EI / 10)"]
        pred_n = [pred_roi_sd, pred_ei / 10.0]
        act_n = [r_act["Signal SD"], act_ei / 10.0]
        x_pos2 = np.arange(len(labels_n))
        ax_bar[1].bar(x_pos2 - w_bar / 2, pred_n, width=w_bar, color="#627D98", label="Predicted (Simulation)")
        ax_bar[1].bar(x_pos2 + w_bar / 2, act_n, width=w_bar, color="#D97706", label="Actual (2nd Shot)")
        ax_bar[1].set_xticks(x_pos2)
        ax_bar[1].set_xticklabels(labels_n, fontsize=8.5)
        ax_bar[1].set_title("Noise & Detector Exposure Consistency", fontsize=9.5, fontweight="bold")
        ax_bar[1].grid(axis="y", alpha=0.25, linestyle="--")
        ax_bar[1].legend(fontsize=8)

        st.pyplot(fig_bar)
        plt.close(fig_bar)

        st.divider()
        st.subheader("📄 4. ส่งออกรายงานตรวจรับผลยิงจริงรอบที่ 2 (Official 1-Page A4 PDF)")

        di_status_row = "PASS" if abs(diag_summary["delta_di"]) <= 0.5 else ("WARNING" if abs(diag_summary["delta_di"]) <= 1.0 else "FAIL")
        snr_status_row = "PASS" if abs(diag_summary["snr_err_pct"]) <= 15.0 else ("WARNING" if abs(diag_summary["snr_err_pct"]) <= 25.0 else "FAIL")
        cnr_status_row = "PASS" if abs(diag_summary["cnr_err_pct"]) <= 15.0 else ("WARNING" if abs(diag_summary["cnr_err_pct"]) <= 25.0 else "FAIL")
        noise_status_row = "PASS" if diag_summary["noise_err_pct"] <= 15.0 else ("WARNING" if diag_summary["noise_err_pct"] <= 25.0 else "FAIL")

        safety_tuple_pdf = (
            "PASS" if (-1.0 <= act_di <= 1.0 and abs(diag_summary["delta_di"]) <= 0.5) else ("WARNING" if abs(act_di) <= 3.0 else "FAIL"),
            f"Actual DI = {act_di:+.2f} vs Predicted DI = {pred_di:+.2f} (Delta DI = {diag_summary['delta_di']:+.2f}, EI = {act_ei:.1f})",
        )
        quality_tuple_pdf = (
            diag_summary["severity"],
            f"{diag_findings[0]['subsystem_en']} [{diag_findings[0]['status']}] (SNR Err {diag_summary['snr_err_pct']:+.1f}%, Noise Err {diag_summary['noise_err_pct']:+.1f}%)",
        )

        post_qa_pdf = generate_qc_report_pdf(
            report_mode="Post-Exposure 2nd-Shot Verification & Hardware QA",
            study_info={
                "Modality": modality,
                "Patient / Study ID": patient_id,
                "Evaluated Layer": f"Slice {sel_s2 + 1} / {shot2_data['num_slices']}",
                "Matrix Size": f"{w2} x {h2} px",
                "Actual / Plan Dose": f"{act_dose_pct:.1f}% / {plan_dose_pct:.1f}%",
            },
            params_rows=[
                ["Tube Voltage (kVp)", f"{base_kvp:.1f} kVp", f"{plan_kvp:.1f} kVp", f"{act_kvp:.1f} kVp ({diag_summary['kvp_diff']:+.1f})"],
                ["Tube Current-Time (mAs)", f"{base_mas:.2f} mAs", f"{plan_mas:.2f} mAs", f"{act_mas:.2f} mAs ({diag_summary['mas_err_pct']:+.1f}%)"],
                ["Source-to-Image Distance", f"{base_sid:.1f} cm", f"{plan_sid:.1f} cm", f"{act_sid:.1f} cm ({diag_summary['sid_diff']:+.1f})"],
                ["Gantry Rotation Angle", f"{base_angle:.1f} deg", f"{plan_angle:.1f} deg", f"{act_angle:.1f} deg ({act_angle - plan_angle:+.1f})"],
            ],
            qc_table_rows=[
                ["Exposure Index (EI)", f"{act_ei:.1f}", f"Predicted = {pred_ei:.1f} (IEC 62494-1)", di_status_row],
                ["Deviation Index (DI)", f"{act_di:+.2f}", f"Predicted = {pred_di:+.2f} (dDI = {diag_summary['delta_di']:+.2f})", di_status_row],
                ["Global Image SNR", f"{act_snr:.2f}", f"Predicted = {pred_snr:.2f} ({diag_summary['snr_err_pct']:+.1f}%)", snr_status_row],
                ["Global Image CNR", f"{act_cnr:.2f}", f"Predicted = {pred_cnr:.2f} ({diag_summary['cnr_err_pct']:+.1f}%)", cnr_status_row],
                ["ROI Signal Mean", f"{r_act['Signal Mean']:.2f} ADU", f"Predicted = {pred_roi_mean:.2f} ADU", "INFO"],
                ["ROI Noise SD (ddof=1)", f"{r_act['Signal SD']:.2f} ADU", f"Predicted = {pred_roi_sd:.2f} ADU ({diag_summary['noise_err_pct']:+.1f}%)", noise_status_row],
                ["ROI SNR", f"{r_act['SNR']:.2f}", f"Predicted = {pred_roi_snr:.2f}", snr_status_row],
                ["ROI CNR", f"{r_act['CNR']:.2f}", f"Predicted = {pred_roi_cnr:.2f}", cnr_status_row],
            ],
            safety_status=safety_tuple_pdf,
            quality_status=quality_tuple_pdf,
            audit_notes=[
                f"Subsystem Verification Result: {diag_findings[0]['subsystem_en']} -> {diag_findings[0]['status']}.",
                f"2nd-shot actual protocol ({act_kvp:.1f} kVp, {act_mas:.2f} mAs, SID {act_sid:.1f} cm) evaluated against planned ({plan_kvp:.1f} kVp, {plan_mas:.2f} mAs, SID {plan_sid:.1f} cm).",
                f"ImageJ ROI (ddof=1) verified at Signal ({roi_info_act['sig_cx']}, {roi_info_act['sig_cy']}) and Background ({roi_info_act['bg_cx']}, {roi_info_act['bg_cy']}).",
            ],
            img_left=slice_1_norm,
            img_left_title=f"Fig 1A. 1st Shot ({base_mas:.1f} mAs)",
            img_right=slice_pred_norm,
            img_right_title=f"Fig 1B. Predicted ({plan_mas:.1f} mAs)",
            img_third=slice_2_actual_norm,
            img_third_title=f"Fig 1C. 2nd Actual ({act_mas:.1f} mAs)",
            roi_info=roi_info_act,
            institution_name=rpt_institution,
            equipment_id=rpt_equipment,
            operator_name=rpt_operator,
            physicist_name=rpt_physicist,
            report_title="POST-EXPOSURE VERIFICATION &\nRADIOLOGY EQUIPMENT QA REPORT",
        )

        st.download_button(
            label="📥 ดาวน์โหลด POST-EXPOSURE VERIFICATION & EQUIPMENT QA REPORT (PDF 1 หน้า A4)",
            data=post_qa_pdf,
            file_name=f"Post_Exposure_Verification_QA_{patient_id}_Slice{sel_s2 + 1}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    else:
        st.warning("⏳ กรุณาอัปโหลดไฟล์ DICOM ให้ครบทั้งรอบที่ 1 (หรือกดส่งมาจากโหมดที่ 1) และไฟล์ที่ยิงซ้ำรอบที่ 2 เพื่อเริ่มการตรวจสอบ")
