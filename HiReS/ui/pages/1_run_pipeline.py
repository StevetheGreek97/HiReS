import streamlit as st
from typing import List, Dict, Tuple

from HiReS.source.pipeline import Pipeline
from HiReS.source.config import Settings

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "is_running" not in st.session_state:
    st.session_state.is_running = False

def param_changed():
    """
    Called whenever a parameter widget changes.
    If a run is active, reset the session to avoid inconsistent state.
    """
    if st.session_state.get("is_running", False):
        # Optional: you won't see this because we rerun immediately,
        # but it's useful if you remove st.rerun() for debugging.
        st.toast(
            "Parameters were changed while the pipeline is running. "
            "Resetting the session."
        )
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------
# Simple multi-checkbox component (kept for later if needed)
# ------------------------------------------------------------
def multi_checkbox(label: str, options: List[str], key: str) -> Dict[str, bool]:
    """Render a simple multi-checkbox selector."""
    st.markdown(f"#### {label}")
    return {opt: st.checkbox(opt, key=f"{key}_{opt}") for opt in options}


# ------------------------------------------------------------
# Page Title
# ------------------------------------------------------------
st.title("🔬 HiReS Pipeline Configuration")
st.write("Configure your high-resolution segmentation pipeline below.")

st.divider()

# ========================================================
# 1. INPUT & OUTPUT
# ========================================================
with st.expander("📂 **1. Input & Output Paths**", expanded=True):
    source = st.text_input(
        "📁 Input file or directory",
        value="/media/steve/UHH_EXT/Pictures/transfer_3118497_files_928f0b81/S.vetulus/D2-1_control.tif",
        on_change=param_changed,
    )
    model_path = st.text_input(
        "🧠 YOLO model (.pt)",
        value="/home/steve/Desktop/best.pt",
        on_change=param_changed,
    )
    output_dir = st.text_input(
        "💾 Output directory",
        value="/home/steve/Desktop/rrrresults/",
        on_change=param_changed,
    )
    st.divider()

# ========================================================
# 2. CHUNKING
# ========================================================
with st.expander("🧩 **2. Chunking & Preprocessing**", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        chunk_w = st.number_input("Chunk width", 
                                  128, 
                                  4096, 
                                  1024,
                                  on_change=param_changed,)
    with c2:
        chunk_h = st.number_input("Chunk height", 
                                  128, 
                                  4096, 
                                  1024,
                                  on_change=param_changed,)

    overlap = st.number_input(
        "Chunk overlap (px)",
        min_value=0,
        max_value=min(chunk_w, chunk_h) - 1,
        value=300,
        on_change=param_changed,
    )
    st.divider()

# ========================================================
# 3. INFERENCE SETTINGS
# ========================================================
with st.expander("🧠 **3. Inference Settings**", expanded=True):
    imgsz = st.number_input("Inference image size (imgsz)", 
                            256, 
                            4096, 
                            1024,
                            on_change=param_changed,)

    conf_thres = st.number_input(
        "Confidence threshold",
        min_value=0.000,
        max_value=1.000,
        value=0.5,
        step=0.001,
        format="%.3f",
        on_change=param_changed,
    )
    st.divider()

# ========================================================
# 4. FILTERING
# ========================================================
with st.expander("🧹 **4. Filtering**", expanded=True):
    edge_threshold = st.number_input(
        "Edge confidence threshold",
        min_value=-0.01,
        max_value=1.0,
        value=0.001,
        step=0.001,
        key="edge_threshold",
        on_change=param_changed,
    )
    iou_thresh = st.number_input(
        "NMS IoU threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        key="iou_thresh",
        on_change=param_changed,
    )

# ========================================================
# 5. VISUALIZATION OPTIONS
# ========================================================
with st.expander("🎨 **5. Visualization Outputs**", expanded=True):

    sg = st.checkbox("Segmentation mask", 
                     value=True, 
                     key="sg",
                     on_change=param_changed,)

    if sg:
        st.markdown("**Segmentation Mask Options:**")
        alpha = st.slider(
            "Mask transparency (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="alpha_slider",
            on_change=param_changed,
        )
    else:
        alpha = None

    pol = st.checkbox("Polygon", 
                      value=True, 
                      key="pol",
                      on_change=param_changed,)
    bb = st.checkbox("Bounding box", 
                     value=True, 
                     key="bb",
                     on_change=param_changed,)

# ========================================================
# 6. RESULT SAVING OPTIONS
# ========================================================
with st.expander("💾 **6. Output Saving Options**", expanded=True):

    enable_metric_selection = st.checkbox(
        "Customize metrics to save",
        value=False,
        key="custom_metrics",
        on_change=param_changed,
    )

    if enable_metric_selection:
        metrics_to_save = st.multiselect(
            "Metrics to save",
            options=[
                "all",
                "area",
                "perimeter",
                "obb_width",
                "obb_length",
                "circularity",
                "eccentricity",
            ],
            default=["all"],
            key="metrics_multiselect",
            on_change=param_changed,
        )
    else:
        metrics_to_save = ["all"]

    save_crops = st.checkbox("Enable object crop saving", 
                             value=True, 
                             key="save_crops",
                             on_change=param_changed,)
    st.divider()

st.divider()

# ========================================================
# RUN BUTTON
# ========================================================
run_clicked = st.button(
    "🚀 Run pipeline",
    use_container_width=True,

)

# placeholders for progress UI
progress_bar = st.progress(0.0, text="")
status_box = st.empty()

# ========================================================
# RUN LOGIC – one click starts the pipeline, guard prevents double-run
# ========================================================
if run_clicked:
    if st.session_state.is_running:
        st.warning("The pipeline was interrupted please wait until it finishes.")
        st.session_state.clear()
        #st.rerun()
    else:
        st.session_state.is_running = True
        hires_cfg = Settings(
            source=source,
            model_path=model_path,
            output_dir=output_dir,
            conf=conf_thres,
            imgsz=imgsz,
            device="cpu",  # or "cuda:0"
            chunk_size=(int(chunk_w), int(chunk_h)),
            overlap=int(overlap),
            edge_threshold=float(edge_threshold),
            iou_thresh=float(iou_thresh),
        )

        st.markdown("### HiReS Settings")
        st.json(hires_cfg.__dict__, expanded=False)

        def progress_cb(frac: float, message: str):
            progress_bar.progress(frac, text=message)
            status_box.write(message)

        st.info("Starting HiReS pipeline…")

        try:
            pipeline = Pipeline(hires_cfg)
            result = pipeline.run(
                workers=1,
                debug=False,
                progress_cb=progress_cb,
            )

            progress_bar.progress(1.0, text="Done ✅")
            st.success("✔ Pipeline finished successfully!")
            st.markdown("### Output paths")
            st.write(result)

        except Exception as e:
            progress_bar.progress(0.0, text="Error")
            st.error(f"Pipeline failed: {e}")
        finally:
            st.session_state.is_running = False

