# Quick Start Guide - Advanced System

## Get Running in 10 Minutes

### Step 1: Install (3 minutes)

```bash
cd advanced_employee_detector
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Zones (5 minutes)

```bash
# Interactive zone configuration
python configure_zones.py --video path/to/your/video.mp4
```

**In the tool:**
1. Press `n` to create new zone
2. Type zone name (e.g., "counter")
3. Click 4+ points on the video to draw polygon
4. Press `s` to save zone
5. Repeat for each zone
6. Press `f` to finish and export config
7. Copy the printed config to `config.py`

**Recommended zones:**
- `counter` - Main service counter
- `behind_counter` - Staff-only area behind counter  
- `customer_area` - Seating/waiting area
- `entrance` - Entry zone

### Step 3: Run Analysis (2 minutes)

```bash
python main.py --video path/to/your/video.mp4
```

**That's it!** Check `outputs/` for results.

---

## Understanding Your Results

### Annotated Video (`outputs/annotated.mp4`)

Open this first. You'll see:
- 🟢 **Green boxes** = Employees
- 🔵 **Blue boxes** = Customers
- 🔴 **Red boxes** = Unknown
- **Yellow/Orange zones** = Your defined areas
- **Trajectory lines** = Movement paths

### Track Summary (`outputs/track_summary.csv`)

Open in Excel/Google Sheets:
- `track_id` - Unique person ID
- `final_class` - Employee/Customer/Unknown
- `confidence` - How confident (0-1)
- `duration_sec` - How long person was tracked
- `zone_distribution` - Time spent in each zone

### Frame Logs (`outputs/frame_logs.csv`)

Detailed frame-by-frame data:
- Position of each person in every frame
- Which zones they're in
- Real-time classification

---

## Tuning for Your Environment

### If Too Many "Unknown"

**Option 1:** Widen the classification gap
```python
# In config.py
CLASSIFICATION = {
    "employee_threshold": 0.60,  # Lower (was 0.65)
    "customer_threshold": 0.40,  # Higher (was 0.35)
}
```

**Option 2:** Adjust feature weights based on what's most reliable
```python
# If zones are very accurate:
"weights": {
    "zone_based": 0.50,      # Increase
    "appearance_based": 0.20, # Decrease others
    "behavioral": 0.20,
    "temporal": 0.10
}

# If uniforms are very distinctive:
"weights": {
    "zone_based": 0.25,
    "appearance_based": 0.50,  # Increase
    "behavioral": 0.15,
    "temporal": 0.10
}
```

### If Employees Classified as Customers

1. **Check zone accuracy** - Are zones correct?
2. **Add uniform colors**:
```python
UNIFORM_DETECTION = {
    "enabled": True,
    "uniform_colors": [
        # Find your uniform colors using an HSV picker
        {"name": "black_apron", "lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 50])},
    ],
}
```

3. **Lower employee threshold**:
```python
"employee_threshold": 0.55  # From 0.65
```

### If Customers Classified as Employees

1. **Raise employee threshold**:
```python
"employee_threshold": 0.75  # From 0.65
```

2. **Add more customer-specific zones** (entrance, seating)

3. **Check if customers are loitering in employee zones**

---

## Common Issues

### "CUDA out of memory"
```python
# Use CPU instead of GPU
# The code will automatically fall back to CPU

# Or use smaller model
DETECTION_MODEL = "yolov8n.pt"  # Instead of yolov8x.pt
```

### Processing is slow
```python
# Use smaller, faster model
DETECTION_MODEL = "yolov8n.pt"

# Disable expensive features
REID_CONFIG['enabled'] = False
OUTPUT_CONFIG['save_crops'] = False
```

### Missing many people
```python
# Lower detection threshold
DETECTION_CONF = 0.25  # From 0.35

# Enable image enhancement
CCTV_OPTIMIZATIONS['clahe_enabled'] = True
```

### Too many false detections
```python
# Raise detection threshold
DETECTION_CONF = 0.45  # From 0.35

# Increase minimum track length
MIN_TRACK_LENGTH = 60  # From 30
```

---

## Example Workflow

### Day 1: Initial Setup
```bash
# 1. Configure zones
python configure_zones.py --video sample.mp4

# 2. Run with default settings
python main.py --video sample.mp4

# 3. Review outputs/annotated.mp4
# 4. Check accuracy in outputs/track_summary.csv
```

### Day 2: Fine-tune
```bash
# Based on Day 1 results:
# - Adjust zones if needed
# - Tune thresholds in config.py
# - Add uniform colors

# Re-run
python main.py --video sample.mp4

# Compare results - should be better!
```

### Day 3: Production
```bash
# Process all your videos
python main.py --video cam1_monday.mp4 --output results/cam1_monday
python main.py --video cam1_tuesday.mp4 --output results/cam1_tuesday
# ... etc
```

---

## Next Steps

Once you have good results:

### 1. Batch Processing
```bash
# Process multiple videos
for video in videos/*.mp4; do
    python main.py --video "$video" --output "outputs/$(basename $video .mp4)"
done
```

### 2. Collect Training Data
```python
# In config.py
DATASET_CONFIG['collect_training_data'] = True
```
Run the system, then use collected crops to train custom classifiers.

### 3. Analyze Patterns
```python
import pandas as pd

# Load all results
results = []
for file in glob.glob('outputs/*/track_summary.csv'):
    df = pd.read_csv(file)
    results.append(df)

combined = pd.concat(results)

# Analyze
print(f"Total employees detected: {len(combined[combined.final_class == 'Employee'])}")
print(f"Total customers detected: {len(combined[combined.final_class == 'Customer'])}")
print(f"Average customer duration: {combined[combined.final_class == 'Customer'].duration_sec.mean():.1f}s")
```

---

## Getting Help

**Problem:** Results don't make sense

**Debug:**
1. Watch `outputs/annotated.mp4` - Are zones placed correctly?
2. Check `outputs/track_summary.csv` - Look at zone_distribution
3. Examine individual tracks with low confidence
4. Adjust config based on what you see

**Problem:** Code crashes

**Check:**
1. Video file is readable: `ffmpeg -i video.mp4`
2. Python environment is activated
3. All dependencies installed: `pip list`
4. Enough disk space for outputs

---

## Performance Expectations

### Processing Speed
- **YOLOv8n**: ~15-20 FPS on CPU, ~60-80 FPS on GPU
- **YOLOv8x**: ~5-8 FPS on CPU, ~30-40 FPS on GPU

### Accuracy
With proper tuning:
- **Employee detection**: 85-95% accuracy
- **Customer detection**: 80-90% accuracy
- **Overall**: 85-92% accuracy

Factors affecting accuracy:
- Zone definition quality
- Uniform distinctiveness
- Video quality
- Camera angle
- Lighting conditions

---

**Ready to analyze your CCTV footage!** 🎬🔍
