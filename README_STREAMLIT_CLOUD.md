# Streamlit Cloud Deployment Guide

## ✅ Fixed Issues

The `ModuleNotFoundError` has been fixed! The app now properly handles imports for Streamlit Cloud.

## Changes Made

### 1. Fixed Import Paths
- Added project root to `sys.path` at the start of `webui/streamlit_app.py`
- This ensures all modules (`tools`, `utils`, `models`) can be found

### 2. Created Streamlit Config
- Added `.streamlit/config.toml` for proper configuration

### 3. Created Packages File
- Added `packages.txt` (empty, for system packages if needed)

## Deployment Steps

### 1. Commit and Push to GitHub
```bash
git add .
git commit -m "Fix Streamlit Cloud imports"
git push
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set:
   - **Main file path**: `webui/streamlit_app.py`
   - **Python version**: 3.10 or 3.11
5. Click "Deploy"

### 3. Required Files in Repository

Make sure these are in your GitHub repo:
- ✅ `webui/streamlit_app.py` (FIXED)
- ✅ `requirements.txt`
- ✅ `.streamlit/config.toml`
- ✅ `packages.txt`
- ✅ All Python modules:
  - `tools/model_runner.py`
  - `utils/` (all files)
  - `models/` (all files)
  - `trainer/` (if needed)
- ✅ `configs/*.yaml` files
- ✅ `checkpoints/` (if you want to include trained models)

## Important Notes

### Checkpoints
- If checkpoints are large (>100MB), use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pth"
  git add .gitattributes
  ```
- Or host checkpoints externally and download them in the app

### Dependencies
All dependencies are in `requirements.txt`. Streamlit Cloud will install them automatically.

### File Structure
Your repository should look like:
```
your-repo/
├── webui/
│   └── streamlit_app.py  ← Main app file
├── tools/
│   └── model_runner.py
├── utils/
│   ├── image_utils.py
│   ├── metrics.py
│   └── losses.py
├── models/
│   ├── srgan.py
│   ├── noise2noise.py
│   └── deblurgan.py
├── configs/
│   ├── srgan.yaml
│   ├── noise2noise.yaml
│   └── deblurgan.yaml
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── packages.txt
```

## Testing Before Deployment

Test locally first:
```bash
streamlit run webui/streamlit_app.py
```

If it works locally, it should work on Streamlit Cloud!

## Troubleshooting

### Still getting import errors?
1. Check that all files are committed to GitHub
2. Verify file paths are correct
3. Check Streamlit Cloud logs for specific errors

### Models not loading?
1. Ensure checkpoints are in the repository
2. Check file paths in the code
3. Verify checkpoint files are accessible

### App crashes on startup?
1. Check Streamlit Cloud logs
2. Verify all dependencies in `requirements.txt`
3. Check Python version compatibility

## Success!

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

The import error should now be fixed! 🎉

