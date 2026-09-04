import streamlit as st
import streamlit.components.v1 as components
import json
import os
import math
import numpy as np
from PIL import Image, ImageDraw
import plotly.graph_objects as go

def keyboard_navigation():
    components.html(
        """
        <script>
        const doc = window.parent.document;
        
        function handleKeyDown(e) {
            const active = doc.activeElement;
            if (active && (
                active.tagName === 'INPUT' || 
                active.tagName === 'TEXTAREA' || 
                active.tagName === 'SELECT' ||
                active.isContentEditable ||
                active.getAttribute('role') === 'combobox' ||
                active.getAttribute('role') === 'slider' ||
                active.getAttribute('aria-expanded') !== null
            )) {
                return;
            }
            
            const key = e.key.toLowerCase();
            const buttons = Array.from(doc.querySelectorAll('button'));
            
            if (e.key === 'ArrowRight') {
                const btn = buttons.find(btn => btn.textContent.includes('Save & Next'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (e.key === 'ArrowLeft') {
                const btn = buttons.find(btn => btn.textContent.includes('Previous'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (key === 'r') {
                const btn = buttons.find(btn => btn.textContent.includes('Reset 0'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (key === 'n') {
                const btn = buttons.find(btn => btn.textContent.includes('Next Step'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (key === 'd' || e.key === 'Delete') {
                const btn = buttons.find(btn => btn.textContent.includes('Delete'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (e.key === '1') {
                const btn = buttons.find(btn => btn.textContent.includes('Pedestrian'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (e.key === '2') {
                const btn = buttons.find(btn => btn.textContent.includes('Stop Sign'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (e.key === '3') {
                const btn = buttons.find(btn => btn.textContent.includes('Avoid Pothole'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            } else if (e.key === '4') {
                const btn = buttons.find(btn => btn.textContent.includes('Bypass Cones'));
                if (btn) {
                    e.preventDefault();
                    btn.click();
                }
            }
        }
        
        if (!window.parent.__keyboard_nav_attached__) {
            doc.addEventListener('keydown', handleKeyDown);
            window.parent.__keyboard_nav_attached__ = true;
        }
        </script>
        """,
        height=0,
        width=0,
    )

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="VLM Dataset Mobile & Desktop Annotator",
    layout="wide",
    page_icon="📱",
    initial_sidebar_state="collapsed"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(APP_DIR, "vlm_dataset.json")
UPDATED_DATASET_FILE = os.path.join(APP_DIR, "vlm_dataset_updated.json")
NPZ_FILE = os.path.join(APP_DIR, "npz", "icp_cache_trimmed_2d.npz")
TOTAL_EPISODE_STEPS = 2695

PRESET_CATEGORIES = {
    "🚸 Pedestrians & Hazards": [
        "Stop and yield for pedestrian crossing ahead.",
        "Slow down and proceed with caution for pedestrian near the road.",
        "Halt vehicle until pedestrian safely finishes crossing."
    ],
    "🛑 Stop Signs & Signals": [
        "Come to a complete stop at the stop sign ahead.",
        "Slow down and prepare to stop at the upcoming stop symbol/intersection.",
        "Pause at the stop line, scan for traffic, and then proceed."
    ],
    "🕳️ Potholes & Road Damage": [
        "Steer slightly left to avoid the pothole in the road.",
        "Steer slightly right to bypass the pothole ahead.",
        "Reduce speed to safely navigate over uneven road surface / potholes."
    ],
    "🚧 Cones & Construction": [
        "Navigate carefully around the construction cones blocking the lane.",
        "Steer left to bypass traffic cones.",
        "Steer right to bypass traffic cones.",
        "Slow down and follow the cone-guided detour path."
    ],
    "🏎️ Standard Navigation": [
        "Navigate forward along the path, maintaining a central position.",
        "Follow the curvature of the road ahead.",
        "Veer slightly left to follow the open corridor.",
        "Veer slightly right to align with the clear path.",
        "Maintain target speed and follow the forward trajectory."
    ]
}

# 1. Load Dataset into Session State
def load_dataset():
    if os.path.exists(UPDATED_DATASET_FILE):
        with open(UPDATED_DATASET_FILE, "r") as f:
            return json.load(f)
    elif os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    else:
        st.error("No dataset file found! Please check vlm_dataset.json.")
        return []

if "data" not in st.session_state:
    st.session_state.data = load_dataset()

# Handle query parameters for idx restoration
if "idx" not in st.session_state:
    url_idx = st.query_params.get("idx")
    if url_idx is not None:
        try:
            val = int(url_idx)
            st.session_state.idx = val
        except ValueError:
            st.session_state.idx = 0
    else:
        st.session_state.idx = 0

idx = st.session_state.idx
data = st.session_state.data
num_samples = len(data)

if num_samples == 0:
    st.warning("Dataset is empty.")
    st.stop()

# Bounds check for idx
if st.session_state.idx >= num_samples:
    st.session_state.idx = max(0, num_samples - 1)
elif st.session_state.idx < 0:
    st.session_state.idx = 0

idx = st.session_state.idx

# Sync/set the query param so the URL matches current state
st.query_params["idx"] = str(idx)

current_sample = data[idx]
current_step = current_sample.get("step", idx * 10)

# Ensure waypoint format: list of 5 [x, y] coordinates
waypoints = current_sample.get("waypoints", [[0.0, 0.0]] * 5)
normalized_wps = []
for wp in waypoints[:5]:
    if isinstance(wp, (list, tuple)) and len(wp) >= 2:
        normalized_wps.append([float(wp[0]), float(wp[1])])
    elif isinstance(wp, (int, float)):
        normalized_wps.append([float(wp), 0.0])
    else:
        normalized_wps.append([0.0, 0.0])
while len(normalized_wps) < 5:
    normalized_wps.append([0.0, 0.0])
current_sample["waypoints"] = normalized_wps

# Helper function to generate 2D Relative Waypoint Plot with Origin at (0,0)
def build_2d_relative_plot(wps):
    fig = go.Figure()
    
    # Origin marker: Robot at (0, 0)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers+text",
        marker=dict(size=22, color="#00E5FF", symbol="triangle-up", line=dict(width=2, color="white")),
        name="Robot Origin (0,0)",
        text=["🤖 Robot (0,0)"],
        textposition="bottom center"
    ))
    
    # Forward X and Lateral Y coordinates
    xs = [wp[0] for wp in wps]
    ys = [wp[1] for wp in wps]
    
    path_x = [0] + xs
    path_y = [0] + ys
    
    # Trajectory Path Line
    fig.add_trace(go.Scatter(
        x=path_y, y=path_x,
        mode="lines",
        line=dict(color="#00FF88", width=4),
        name="Trajectory Path"
    ))
    
    # Waypoint Node Circles with labels
    fig.add_trace(go.Scatter(
        x=ys, y=xs,
        mode="markers+text",
        marker=dict(size=14, color="#FF3366", symbol="circle", line=dict(width=2, color="white")),
        text=[f"WP{i+1}: ({xs[i]:.2f}, {ys[i]:.2f})m" for i in range(len(xs))],
        textposition="top center",
        name="Waypoints (1..5)"
    ))
    
    # Dynamic Autoscale calculation based on current waypoints
    max_fwd = max(3.0, max([abs(x) for x in xs]) + 1.0)
    max_lat = max(2.0, max([abs(y) for y in ys]) + 1.0)
    
    fig.update_layout(
        title="🧭 2D Mirrored Ego Waypoints (0,0)",
        xaxis_title="← Left (+Y)  |  Lateral (m)  |  Right (-Y) →",
        yaxis_title="Forward Distance X (m) [ ↑ Ahead ]",
        xaxis=dict(
            range=[max_lat, -max_lat],  # MIRRORED: Left on screen = Left (+Y) in robot frame
            zeroline=True, zerolinewidth=2, zerolinecolor="white", gridcolor="#333344"
        ),
        yaxis=dict(
            range=[-0.8, max_fwd],      # AUTOSCALED FORWARD AXIS
            zeroline=True, zerolinewidth=2, zerolinecolor="white", gridcolor="#333344"
        ),
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

# Callbacks and State Management Helpers
def save_current_state():
    curr_idx = st.session_state.idx
    if curr_idx >= len(st.session_state.data):
        return
        
    # Sync prompt
    prompt_key = f"prompt_{curr_idx}"
    if prompt_key in st.session_state:
        st.session_state.data[curr_idx]["prompt"] = st.session_state[prompt_key]
        
    # Sync waypoints
    updated_wps = []
    for i in range(5):
        x_val = st.session_state.get(f"wp_x_{curr_idx}_{i}", st.session_state.data[curr_idx]["waypoints"][i][0])
        y_val = st.session_state.get(f"wp_y_{curr_idx}_{i}", st.session_state.data[curr_idx]["waypoints"][i][1])
        updated_wps.append([float(x_val), float(y_val)])
    st.session_state.data[curr_idx]["waypoints"] = updated_wps
    
    # Save files
    with open(UPDATED_DATASET_FILE, "w") as f:
        json.dump(st.session_state.data, f, indent=2)
    with open(DATASET_FILE, "w") as f:
        json.dump(st.session_state.data, f, indent=2)

def reset_waypoints_callback():
    curr_idx = st.session_state.idx
    st.session_state.data[curr_idx]["waypoints"] = [[0.0, 0.0] for _ in range(5)]
    for i in range(5):
        st.session_state[f"wp_x_{curr_idx}_{i}"] = 0.0
        st.session_state[f"wp_y_{curr_idx}_{i}"] = 0.0
        st.session_state[f"slider_x_desk_{curr_idx}_{i}"] = 0.0
        st.session_state[f"slider_y_desk_{curr_idx}_{i}"] = 0.0
    save_current_state()
    st.toast("Reset all 5 waypoints to (0.00, 0.00)!", icon="🔄")

def advance_step_callback():
    curr_idx = st.session_state.idx
    wps = st.session_state.data[curr_idx]["waypoints"]
    shifted_wps = [list(w) for w in wps[1:]] + [[0.0, 0.0]]
    st.session_state.data[curr_idx]["waypoints"] = shifted_wps
    for i in range(5):
        st.session_state[f"wp_x_{curr_idx}_{i}"] = shifted_wps[i][0]
        st.session_state[f"wp_y_{curr_idx}_{i}"] = shifted_wps[i][1]
        st.session_state[f"slider_x_desk_{curr_idx}_{i}"] = shifted_wps[i][0]
        st.session_state[f"slider_y_desk_{curr_idx}_{i}"] = shifted_wps[i][1]
    save_current_state()
    st.toast("Advanced step: shifted waypoints forward!", icon="➡️")

def save_and_next_callback():
    curr_idx = st.session_state.idx
    save_current_state()
    
    step_num = st.session_state.data[curr_idx].get("step", curr_idx * 10)
    st.toast(f"Saved Step {step_num}!", icon="💾")
    if st.session_state.idx < len(st.session_state.data) - 1:
        st.session_state.idx += 1

def previous_callback():
    if st.session_state.idx > 0:
        st.session_state.idx -= 1

def delete_sample_callback():
    curr_idx = st.session_state.idx
    if len(st.session_state.data) > 0:
        # Remove current sample
        st.session_state.data.pop(curr_idx)
        
        # Save updated datasets
        save_current_state()
        
        # Prevent state bleeding by clearing session state keys for current and shifted indices
        keys_to_clear = []
        for key in list(st.session_state.keys()):
            parts = key.split('_')
            for part in parts:
                if part.isdigit() and int(part) >= curr_idx:
                    keys_to_clear.append(key)
                    break
        for key in keys_to_clear:
            st.session_state.pop(key, None)
            
        # Adjust index to stay in bounds
        st.session_state.idx = max(0, min(curr_idx, len(st.session_state.data) - 1))
        st.toast("Sample deleted successfully!", icon="🗑️")

def reset_from_npz_callback():
    curr_idx = st.session_state.idx
    if os.path.exists(NPZ_FILE):
        try:
            npz = np.load(NPZ_FILE)
            poses = npz["poses"]
            step_idx = st.session_state.data[curr_idx].get("step", 0)
            if step_idx < len(poses):
                p_curr = poses[step_idx][:2, 3]
                
                lookahead = min(step_idx + 15, len(poses) - 1)
                v_motion = poses[lookahead][:2, 3] - p_curr
                if np.linalg.norm(v_motion) < 1e-2:
                    lookahead = min(step_idx + 40, len(poses) - 1)
                    v_motion = poses[lookahead][:2, 3] - p_curr
                
                if np.linalg.norm(v_motion) < 1e-3:
                    R = poses[step_idx][:3, :3]
                    th = math.atan2(R[1, 0], R[0, 0])
                else:
                    th = math.atan2(v_motion[1], v_motion[0])
                
                recalc_wps = []
                for k in range(1, 6):
                    fut_step = min(step_idx + k * 10, len(poses) - 1)
                    p_fut = poses[fut_step][:2, 3]
                    diff_w = p_fut - p_curr
                    
                    fwd = math.cos(th)*diff_w[0] + math.sin(th)*diff_w[1]
                    lat = -math.sin(th)*diff_w[0] + math.cos(th)*diff_w[1]
                    recalc_wps.append([round(float(fwd), 3), round(float(lat), 3)])
                
                st.session_state.data[curr_idx]["waypoints"] = recalc_wps
                for i in range(5):
                    st.session_state[f"wp_x_{curr_idx}_{i}"] = recalc_wps[i][0]
                    st.session_state[f"wp_y_{curr_idx}_{i}"] = recalc_wps[i][1]
                    st.session_state[f"slider_x_desk_{curr_idx}_{i}"] = recalc_wps[i][0]
                    st.session_state[f"slider_y_desk_{curr_idx}_{i}"] = recalc_wps[i][1]
                save_current_state()
                st.toast("Recalculated accurate forward waypoints from NPZ!", icon="✨")
        except Exception as e:
            st.error(f"Error reading NPZ: {e}")

# Additional Widget Interaction Callbacks
def save_preset_callback(prefix, idx):
    preset_key = f"preset_dropdown_{prefix}_{idx}"
    preset_val = st.session_state.get(preset_key, "-- Keep Current / Custom --")
    if preset_val != "-- Keep Current / Custom --":
        p_key = f"prompt_{idx}"
        st.session_state[p_key] = preset_val
        st.session_state.data[idx]["prompt"] = preset_val
        st.session_state[preset_key] = "-- Keep Current / Custom --"
        save_current_state()

def save_prompt_callback(idx):
    p_key = f"prompt_{idx}"
    new_prompt = st.session_state.get(p_key, "")
    st.session_state.data[idx]["prompt"] = new_prompt
    save_current_state()

def slider_x_callback(idx, i, prefix):
    slider_val = st.session_state[f"slider_x_{prefix}_{idx}_{i}"]
    st.session_state[f"wp_x_{idx}_{i}"] = slider_val
    save_current_state()

def number_x_callback(idx, i, prefix):
    num_val = st.session_state[f"wp_x_{idx}_{i}"]
    slider_max_x = max(20.0, abs(num_val) * 2.0 + 5.0)
    clamped_val = max(-2.0, min(float(slider_max_x), float(num_val)))
    st.session_state[f"slider_x_{prefix}_{idx}_{i}"] = clamped_val
    save_current_state()

def slider_y_callback(idx, i, prefix):
    slider_val = st.session_state[f"slider_y_{prefix}_{idx}_{i}"]
    st.session_state[f"wp_y_{idx}_{i}"] = slider_val
    save_current_state()

def number_y_callback(idx, i, prefix):
    num_val = st.session_state[f"wp_y_{idx}_{i}"]
    slider_max_y = max(15.0, abs(num_val) * 2.0 + 5.0)
    clamped_val = max(-float(slider_max_y), min(float(slider_max_y), float(num_val)))
    st.session_state[f"slider_y_{prefix}_{idx}_{i}"] = clamped_val
    save_current_state()

# Sidebar controls & info
with st.sidebar:
    st.title("📱 VLM Annotator")
    st.markdown("---")
    st.subheader("Dataset Info")
    st.write(f"🖼️ **Keyframe Samples**: {num_samples}")
    st.write(f"⏱️ **Total Episode Steps**: {TOTAL_EPISODE_STEPS}")
    st.write(f"📍 **Path**: `{DATASET_FILE}`")
    
    st.markdown("---")
    st.subheader("Telemetry")
    telemetry = current_sample.get("telemetry", {})
    st.metric("VIO X / Y (m)", f"{telemetry.get('vio_x', 0.0):.1f} / {telemetry.get('vio_y', 0.0):.1f}")
    st.metric("Steering / Throttle", f"{telemetry.get('steering', 0.0):.2f} / {telemetry.get('throttle', 0.0):.2f}")
    
    st.markdown("---")
    if st.button("✨ Recalculate NPZ Waypoints", on_click=reset_from_npz_callback, use_container_width=True):
        pass
        
    st.markdown("---")
    st.subheader("⌨️ Keyboard Shortcuts")
    st.markdown(
        """
        - **ArrowLeft**: ⬅️ Previous
        - **ArrowRight**: 💾 Save & Next
        - **R / r**: 🔄 Reset 0
        - **N / n**: ➡️ Next Step
        - **D / d / Delete**: 🗑️ Delete Sample
        - **1**: 🚸 Pedestrian Tag
        - **2**: 🛑 Stop Sign Tag
        - **3**: 🕳️ Avoid Pothole Tag
        - **4**: 🚧 Bypass Cones Tag
        
        *Note: Shortcuts bypass textareas, inputs, and sliders.*
        """
    )

# Top Navigation Bar (Mobile Optimized)
st.title(f"Sample {idx + 1}/{num_samples} (Step {current_step}/{TOTAL_EPISODE_STEPS})")

col_jump1, col_jump2 = st.columns([3, 2])
with col_jump1:
    selected_sample = st.selectbox(
        "Jump to Image / Step:",
        options=list(range(num_samples)),
        format_func=lambda i: f"Sample {i+1}/{num_samples} (Step {data[i].get('step', i*10)})",
        index=idx,
        key="sample_select_dropdown"
    )
    if selected_sample != idx:
        st.session_state.idx = selected_sample
        st.rerun()

with col_jump2:
    progress_val = (current_step / TOTAL_EPISODE_STEPS) if TOTAL_EPISODE_STEPS > 0 else (idx + 1) / num_samples
    st.progress(min(1.0, max(0.0, progress_val)))

st.divider()

# Helper function to render camera image
def render_camera_image(prefix=""):
    img_rel_path = current_sample.get("image_path", "")
    img_full_path = os.path.join(APP_DIR, img_rel_path) if not os.path.isabs(img_rel_path) else img_rel_path
    
    show_overlay = st.checkbox("Overlay Waypoints on Camera Image", value=True, key=f"ov_{prefix}_{idx}")
    
    if os.path.exists(img_full_path):
        base_image = Image.open(img_full_path).convert("RGB")
        if show_overlay:
            overlay_img = base_image.copy()
            draw = ImageDraw.Draw(overlay_img)
            w, h = overlay_img.size
            cx, cy = w // 2, h - 40
            
            wps = current_sample.get("waypoints", [[0.0, 0.0]] * 5)
            img_pts = []
            
            for i, (wx, wy) in enumerate(wps):
                py = cy - int(float(wx) * 45)
                px = cx - int(float(wy) * 60)
                px = max(15, min(w - 15, px))
                py = max(15, min(h - 15, py))
                img_pts.append((px, py))
            
            start_pt = (cx, cy)
            all_pts = [start_pt] + img_pts
            for k in range(len(all_pts) - 1):
                draw.line([all_pts[k], all_pts[k+1]], fill=(0, 255, 150), width=4)
            
            for i, (px, py) in enumerate(img_pts):
                r = 7
                draw.ellipse([px-r, py-r, px+r, py+r], fill=(255, 60, 60), outline=(255, 255, 255), width=2)
                wx, wy = wps[i]
                draw.text((px + 10, py - 8), f"WP{i+1}: ({wx:.1f}, {wy:.1f})m", fill=(255, 255, 0))
            
            st.image(overlay_img, use_container_width=True)
        else:
            st.image(base_image, use_container_width=True)
    else:
        st.error(f"Image not found: {img_full_path}")

# Helper function to render prompt editor
def render_prompt_editor(prefix=""):
    category = st.selectbox(
        "📁 Instruction Category:",
        options=list(PRESET_CATEGORIES.keys()),
        key=f"cat_select_{prefix}_{idx}"
    )
    cat_prompts = PRESET_CATEGORIES[category]
    st.selectbox(
        "⚡ Choose Quick Preset Instruction:",
        options=["-- Keep Current / Custom --"] + cat_prompts,
        key=f"preset_dropdown_{prefix}_{idx}",
        on_change=save_preset_callback,
        args=(prefix, idx)
    )
    
    initial_prompt_val = current_sample.get("prompt", "")

    st.caption("🚀 Quick Scenario Shortcut Tags:")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1:
        if st.button("🚸 Pedestrian", key=f"t1_{prefix}_{idx}", use_container_width=True):
            st.session_state[f"prompt_{idx}"] = "Stop and yield for pedestrian crossing ahead."
            st.session_state.data[idx]["prompt"] = "Stop and yield for pedestrian crossing ahead."
            save_current_state()
            st.rerun()
    with tcol2:
        if st.button("🛑 Stop Sign", key=f"t2_{prefix}_{idx}", use_container_width=True):
            st.session_state[f"prompt_{idx}"] = "Come to a complete stop at the stop sign ahead."
            st.session_state.data[idx]["prompt"] = "Come to a complete stop at the stop sign ahead."
            save_current_state()
            st.rerun()
    with tcol3:
        if st.button("🕳️ Avoid Pothole", key=f"t3_{prefix}_{idx}", use_container_width=True):
            st.session_state[f"prompt_{idx}"] = "Steer slightly left to avoid the pothole in the road."
            st.session_state.data[idx]["prompt"] = "Steer slightly left to avoid the pothole in the road."
            save_current_state()
            st.rerun()
    with tcol4:
        if st.button("🚧 Bypass Cones", key=f"t4_{prefix}_{idx}", use_container_width=True):
            st.session_state[f"prompt_{idx}"] = "Navigate carefully around the construction cones blocking the lane."
            st.session_state.data[idx]["prompt"] = "Navigate carefully around the construction cones blocking the lane."
            save_current_state()
            st.rerun()

    p_key = f"prompt_{idx}"
    st.text_area(
        "VLM Instruction (Editable):",
        value=initial_prompt_val,
        height=85,
        key=p_key,
        on_change=save_prompt_callback,
        args=(idx,)
    )

# Helper function to render waypoints sliders
def render_waypoints_controls(prefix=""):
    current_wps = current_sample.get("waypoints", [[0.0, 0.0]] * 5)
    
    for i in range(5):
        wx_val = float(current_wps[i][0]) if i < len(current_wps) else 0.0
        wy_val = float(current_wps[i][1]) if i < len(current_wps) else 0.0
        
        with st.expander(f"📍 Waypoint {i+1}: ({wx_val:.2f}m, {wy_val:.2f}m)", expanded=(i == 0)):
            col_sl_x, col_sl_y = st.columns(2)
            slider_max_x = max(20.0, abs(wx_val) * 2.0 + 5.0)
            slider_max_y = max(15.0, abs(wy_val) * 2.0 + 5.0)
            
            with col_sl_x:
                st.slider(
                    f"WP {i+1} Forward X (m)", 
                    min_value=-2.0, 
                    max_value=slider_max_x, 
                    value=wx_val, 
                    step=0.05, 
                    key=f"slider_x_{prefix}_{idx}_{i}",
                    on_change=slider_x_callback,
                    args=(idx, i, prefix)
                )
                st.number_input(
                    f"WP {i+1} Fine X", 
                    value=wx_val, 
                    format="%.2f", 
                    step=0.01, 
                    key=f"wp_x_{idx}_{i}",
                    on_change=number_x_callback,
                    args=(idx, i, prefix)
                )
            
            with col_sl_y:
                st.slider(
                    f"WP {i+1} Lateral Y (m)", 
                    min_value=-slider_max_y, 
                    max_value=slider_max_y, 
                    value=wy_val, 
                    step=0.05, 
                    key=f"slider_y_{prefix}_{idx}_{i}",
                    on_change=slider_y_callback,
                    args=(idx, i, prefix)
                )
                st.number_input(
                    f"WP {i+1} Fine Y", 
                    value=wy_val, 
                    format="%.2f", 
                    step=0.01, 
                    key=f"wp_y_{idx}_{i}",
                    on_change=number_y_callback,
                    args=(idx, i, prefix)
                )

# Helper function for action buttons
def render_action_buttons(prefix=""):
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)
    with b_col1:
        st.button("🔄 Reset 0", on_click=reset_waypoints_callback, use_container_width=True, key=f"rst_{prefix}_{idx}")
    with b_col2:
        st.button("➡️ Next Step", on_click=advance_step_callback, use_container_width=True, key=f"nxtst_{prefix}_{idx}")
    with b_col3:
        st.button("💾 Save & Next", on_click=save_and_next_callback, type="primary", use_container_width=True, key=f"svnxt_{prefix}_{idx}")
    with b_col4:
        st.button("⬅️ Previous", on_click=previous_callback, disabled=(idx == 0), use_container_width=True, key=f"prv_{prefix}_{idx}")
    with b_col5:
        st.button("🗑️ Delete", on_click=delete_sample_callback, use_container_width=True, key=f"del_{prefix}_{idx}")

# Render desktop layout side-by-side
left_col, right_col = st.columns([1.2, 1])
with left_col:
    st.subheader("🖼️ Camera Frame")
    render_camera_image(prefix="desk")

with right_col:
    st.subheader("🧭 2D Waypoints Plot")
    fig_2d = build_2d_relative_plot(current_sample["waypoints"])
    st.plotly_chart(fig_2d, use_container_width=True)
    
    st.markdown("---")
    render_prompt_editor(prefix="desk")
    
    st.markdown("---")
    render_waypoints_controls(prefix="desk")
    
    st.markdown("---")
    render_action_buttons(prefix="desk")

st.divider()

with st.expander("🔍 View Current Sample JSON Record"):
    st.json(current_sample)

with st.expander("📥 Download / Export Dataset"):
    dataset_str = json.dumps(st.session_state.data, indent=2)
    st.download_button(
        label="Download vlm_dataset_updated.json",
        data=dataset_str,
        file_name="vlm_dataset_updated.json",
        mime="application/json",
    )

keyboard_navigation()
