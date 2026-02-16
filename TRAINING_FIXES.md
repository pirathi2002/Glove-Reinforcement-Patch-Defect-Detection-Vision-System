# Training Fixes - February 16, 2026

## Overview
Fixed multiple critical issues preventing Anomalib model training on Windows. Training now works successfully with Patchcore and Padim models.

## Issues Fixed

### 1. Incompatible Folder Datamodule Arguments
**Problem:** Anomalib's `Folder` datamodule had varying signatures across versions. The training code tried to pass `task="segmentation"` and other parameters that some versions didn't support.

**Solution:** Implemented fallback chain in `src/train_models.py` `_create_datamodule()`:
- Try full signature with all Anomalib-specific parameters
- Fall back to simpler signature if TypeError occurs
- Final fallback: minimal `Folder(root, normal_dir)` call
- Each fallback is logged for debugging

### 2. Windows Symlink Privilege Error (WinError 1314)
**Problem:** Anomalib's engine tried to create a symbolic link for the `latest` checkpoint directory, which requires administrator privileges on Windows.

**Solution:** Patched `create_versioned_dir` function in `src/train_models.py`:
- Wraps the original function in a try-except
- On WinError 1314, creates a regular versioned directory (`v1`, `v2`, etc.) instead
- Preserves functionality without requiring elevated privileges

### 3. PyTorch Lightning Trainer Receives Anomalib-Only Arguments
**Problem:** Engine was passing Anomalib-specific kwargs (`task`, `image_metrics`, `pixel_metrics`) to PyTorch Lightning's `Trainer.__init__()`, which doesn't recognize them.

**Solution:** Added cleanup in `src/train_models.py` before `engine.fit()`:
```python
for _k in ("task", "image_metrics", "pixel_metrics"):
    engine._cache._cached_args.pop(_k, None)
```

### 4. Model Backbone Attribute Not Found
**Problem:** Code tried to print `model.backbone` which doesn't exist on all model types (Padim, Patchcore, etc.).

**Solution:** Safe attribute lookup with fallbacks:
- Check `backbone`, `backbone_model`, `network`, `feature_extractor` in order
- Display "N/A" if no backbone attribute found
- Wrapped in try-except for robustness

## Files Modified

- **src/train_models.py**
  - Updated `_create_datamodule()` with fallback chain
  - Updated `train()` method with:
    - Symlink workaround patch
    - Trainer args cleanup
    - Safe backbone attribute access

- **config.py**
  - Set `ROI_CONFIG['debug']` to `False` to avoid OpenCV display errors

- **src/preprocessing.py**
  - Added `fast_mode` parameter to `preprocess_training_data()` for faster preprocessing

## Testing

Successfully ran:
```bash
python main.py --train --model patchcore --folder 0
python main.py --train --model padim --folder 0
```

## Checkpoint Location

After training completes, model checkpoints are saved in:
```
models/<model_name>/folder_<idx>/<version>/checkpoints/
```

Example: `models/padim/folder_00/v1/checkpoints/padim.pth`

## Notes

- All fixes are backward compatible
- Graceful degradation if patches fail (training continues with original behavior)
- Extensive logging for debugging
- Works on Windows without requiring administrator privileges for symlinks
