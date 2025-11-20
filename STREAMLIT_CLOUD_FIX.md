# Streamlit Cloud Deployment Fix

## Issue Fixed
The `ModuleNotFoundError` for `tools.model_runner` has been fixed by:
1. Adding project root to `sys.path` at the start of the app
2. Adding fallback import mechanism for Streamlit Cloud
3. Ensuring all imports work correctly

## Changes Made

### 1. Fixed `webui/streamlit_app.py`
- Added project root to `sys.path` before imports
- Added fallback import mechanism for Streamlit Cloud
- Fixed import paths

### 2. Created `.streamlit/config.toml`
- Streamlit configuration file
- Sets proper server settings

### 3. Created `packages.txt`
- For system-level dependencies (if needed)

## Deployment Checklist

### Required Files in Repository
- ✅ `webui/streamlit_app.py` - Main app (FIXED)
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit config
- ✅ `packages.txt` - System packages (empty, can add if needed)
- ✅ All model files (`models/`, `tools/`, `utils/`, etc.)
- ✅ Config files (`configs/*.yaml`)
- ✅ Checkpoints (if you want to include trained models)

### Streamlit Cloud Setup

1. **Connect Repository**
   - Go to Streamlit Cloud
   - Connect your GitHub repository

2. **Configure App**
   - Main file path: `webui/streamlit_app.py`
   - Python version: 3.10+ (recommended)

3. **Deploy**
   - Streamlit Cloud will automatically:
     - Install dependencies from `requirements.txt`
     - Run the app from `webui/streamlit_app.py`

## Important Notes

### Checkpoints
If you want to use trained models on Streamlit Cloud:
- Upload checkpoints to your repository
- Or use Git LFS for large files
- Or host checkpoints elsewhere and download them

### File Size Limits
- Streamlit Cloud has file size limits
- Large checkpoint files (>100MB) may need Git LFS
- Consider using smaller models or hosting checkpoints externally

### Dependencies
All required packages are in `requirements.txt`:
- torch, torchvision
- streamlit
- PIL, numpy, scikit-image
- opencv-python
- lpips
- pyyaml

## Testing Locally

Before deploying, test locally:
```bash
streamlit run webui/streamlit_app.py
```

## Troubleshooting

### If imports still fail:
1. Check that all files are in the repository
2. Verify `requirements.txt` has all dependencies
3. Check Streamlit Cloud logs for specific errors

### If models don't load:
1. Ensure checkpoints are in the repository
2. Check file paths in the code
3. Verify checkpoint files are accessible

## Next Steps

1. Commit and push all changes to GitHub
2. Deploy on Streamlit Cloud
3. Test the deployed app
4. Monitor logs for any issues

