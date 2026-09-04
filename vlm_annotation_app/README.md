# VLM Dataset Annotation Tool

This is a standalone annotation directory created for dataset inspection and annotation.

## Directory Structure
```
vlm_annotation_app/
├── app.py                      # Main Streamlit annotation application
├── vlm_dataset.json            # Active JSON dataset (270 samples)
├── vlm_dataset_updated.json    # Auto-saved updated dataset
├── images/                     # 270 camera frame images (frame_000000.jpg ... frame_002690.jpg)
└── npz/
    └── icp_cache_trimmed_2d.npz # 2D odometry poses from VIO/ICP
```

## How to Run

1. Open your terminal in this directory:
   ```bash
   cd /home/riteshrajas/elisa/vlm_annotation_app
   ```

2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- **Visual Frame Inspection**: View camera frames with real-time waypoint overlay.
- **Prompt Instruction Editor**: Update instructions for VLM training.
- **5-Step Waypoints Editing**: Tune Forward (`X`) and Lateral (`Y`) coordinates for all 5 waypoints.
- **Action Buttons**:
  - `🔄 Reset Waypoints to 0`: Reset waypoints to `(0.00, 0.00)`.
  - `➡️ Next Step`: Shift waypoints forward.
  - `💾 Save & Next`: Save edits to JSON and advance to next sample.
  - `⬅️ Previous`: Return to previous sample.
  - `✨ Recalculate Waypoints from NPZ`: Reload ground-truth VIO waypoints from `icp_cache_trimmed_2d.npz`.
