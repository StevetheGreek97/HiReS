import streamlit as st
import subprocess
import threading
import json
import psutil
import sys
from pathlib import Path

from HiReS.source.config import Settings


# =============================================================
# SESSION STATE
# =============================================================
if "proc" not in st.session_state:
    st.session_state.proc = None
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "status" not in st.session_state:
    st.session_state.status = "Idle"


# =============================================================
# PROCESS KILLER
# =============================================================
def kill_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    for child in parent.children(recursive=True):
        try: child.kill()
        except: pass
    try: parent.kill()
    except: pass


# =============================================================
# LISTENER THREAD — reads output from subprocess
# =============================================================
def listen_to_process(proc):
    for raw in iter(proc.stdout.readline, b""):
        if not raw:
            break

        line = raw.decode().strip()

        # Debug
        print("SUBPROCESS:", line)

        if line.startswith("PROGRESS"):
            _, frac, *msg = line.split(" ", 2)
            st.session_state.progress = float(frac)
            st.session_state.status = " ".join(msg)
            continue

        if line == "DONE":
            st.session_state.progress = 1.0
            st.session_state.status = "Finished!"
            st.session_state.proc = None
            return

    # If subprocess died unexpectedly
    err = proc.stderr.read().decode()
    if err:
        st.session_state.status = "Error"
        st.session_state.proc = None
        st.error(f"Subprocess crashed:\n\n{err}")


# =============================================================
# UI HEADER
# =============================================================
st.title("🔬 HiReS Pipeline Configuration (Subprocess Mode)")

if st.session_state.proc is not None:
    st.warning("⚠ Pipeline is running. Avoid changing settings.")

st.divider()


# =============================================================
# YOUR ORIGINAL UI LAYOUT (unchanged)
# =============================================================

# 1. INPUT & OUTPUT
with st.expander("📂 **1. Input & Output Paths**", expanded=True):
    source = st.text_input(
        "📁 Input file or directory",
        "/media/steve/UHH_EXT/Pictures/transfer_3118497_files_928f0b81/S.vetulus/D2-1_control.tif",
        key="source"
    )
    model_path = st.text_input(
        "🧠 YOLO model (.pt)",
        "/home/steve/Desktop/best.pt",
        key="model_path"
    )
    output_dir = st.text_input(
        "💾 Output directory",
        "/home/steve/Desktop/rrrresults/",
        key="output_dir"
    )
    st.divider()

# 2. CHUNKING
with st.expander("🧩 **2. Chunking & Preprocessing**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        chunk_w = st.number_input("Chunk width", 128, 4096, 1024, key="chunk_w")
    with c2:
        chunk_h = st.number_input("Chunk height", 128, 4096, 1024, key="chunk_h")
    overlap = st.number_input("Chunk overlap (px)", 0, min(chunk_w, chunk_h)-1, 300, key="overlap")
    st.divider()

# 3. INFERENCE
with st.expander("🧠 **3. Inference Settings**", expanded=True):
    imgsz = st.number_input("Inference image size (imgsz)", 256, 4096, 1024, key="imgsz")
    conf_thres = st.number_input("Confidence threshold", 0.0, 1.0, 0.5,
                                 step=0.001, format="%.3f", key="conf_thres")
    st.divider()

# 4. FILTERING
with st.expander("🧹 **4. Filtering**", expanded=True):
    edge_threshold = st.number_input("Edge confidence threshold", -0.01, 1.0, 0.001,
                                     step=0.001, key="edge_threshold")
    iou_thresh = st.number_input("NMS IoU threshold", 0.0, 1.0, 0.7,
                                 step=0.01, key="iou_thresh")

# 5. VISUALIZATION
with st.expander("🎨 **5. Visualization Outputs**", expanded=True):
    sg = st.checkbox("Segmentation mask", True, key="sg")
    if sg:
        st.markdown("**Segmentation Mask Options:**")
        alpha = st.slider("Mask transparency (alpha)", 0.0, 1.0, 0.5,
                          step=0.05, key="alpha_slider")
    else:
        alpha = None
    pol = st.checkbox("Polygon", True, key="pol")
    bb = st.checkbox("Bounding box", True, key="bb")

# 6. METRICS
with st.expander("💾 **6. Output Saving Options**", expanded=True):
    enable_metric_selection = st.checkbox("Customize metrics to save", False, key="custom_metrics")
    if enable_metric_selection:
        metrics_to_save = st.multiselect(
            "Metrics to save",
            ["all", "area", "perimeter", "obb_width", "obb_length", "circularity", "eccentricity"],
            default=["all"],
            key="metrics_multiselect")
    else:
        metrics_to_save = ["all"]

    save_crops = st.checkbox("Enable object crop saving", True, key="save_crops")
    st.divider()

st.divider()


# =============================================================
# PROGRESS DISPLAY
# =============================================================
progress_bar = st.progress(st.session_state.progress)
status_box = st.write(f"Status: {st.session_state.status}")


# =============================================================
# RUN BUTTON
# =============================================================
if st.button("🚀 Run pipeline", disabled=st.session_state.proc is not None):

    st.write("DEBUG: Run clicked")

    # PATH TO SUBPROCESS
    script_path = Path("/home/steve/Documents/Packages/hireseg/HiReS/ui/run_pipeline_subprocess.py")
    st.write("DEBUG: Subprocess script:", str(script_path))

    if not script_path.exists():
        st.error("❌ Subprocess script not found.")
        st.stop()

    # Build config dictionary
    cfg = Settings(
        source=st.session_state.source,
        model_path=st.session_state.model_path,
        output_dir=st.session_state.output_dir,
        conf=st.session_state.conf_thres,
        imgsz=st.session_state.imgsz,
        device="cpu",
        chunk_size=(int(st.session_state.chunk_w), int(st.session_state.chunk_h)),
        overlap=int(st.session_state.overlap),
        edge_threshold=float(st.session_state.edge_threshold),
        iou_thresh=float(st.session_state.iou_thresh),
    )

    st.session_state.progress = 0.0

    # Launch subprocess
    PY = sys.executable  # ALWAYS use this!
    st.write("DEBUG: Using Python executable:", PY)

    proc = subprocess.Popen(
        [PY, "-u", str(script_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    st.write("DEBUG: Subprocess PID:", proc.pid)

    st.session_state.proc = proc

    # Send config
    proc.stdin.write(json.dumps(cfg.__dict__).encode())
    proc.stdin.close()

    # Start background listener
    threading.Thread(target=listen_to_process, args=(proc,), daemon=True).start()

    st.rerun()


# =============================================================
# STOP BUTTON
# =============================================================
if st.session_state.proc is not None:
    if st.button("⛔ Stop"):
        kill_tree(st.session_state.proc.pid)
        st.session_state.proc = None
        st.session_state.progress = 0.0
        st.session_state.status = "Stopped"
        st.rerun()
