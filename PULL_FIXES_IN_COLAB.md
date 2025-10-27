# How to Pull CUDA Error Fixes in Google Colab

## Step 1: Check Current Branch

```python
!git branch
```

You should see: `claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj`

## Step 2: Pull Latest Changes

```python
# Fetch from remote
!git fetch origin claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj

# Pull the changes
!git pull origin claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj
```

## Step 3: Verify Files Exist

```python
import os

files_to_check = [
    'fix_cuda_error.py',
    'diagnose_cuda_error.py',
    'CUDA_ERROR_FIX.md',
    'CUDA_ERROR_SOLUTION.md',
    'colab_cell_full_training_safe.py'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - NOT FOUND")
```

## Step 4: Run the Fix

```python
!python fix_cuda_error.py
```

## Alternative: Direct Download from GitHub

If git pull doesn't work, download files directly:

```python
import requests

base_url = "https://raw.githubusercontent.com/SSanpui/ESA-NMT/claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj/"

files = {
    'fix_cuda_error.py': 'fix_cuda_error.py',
    'diagnose_cuda_error.py': 'diagnose_cuda_error.py',
    'CUDA_ERROR_FIX.md': 'CUDA_ERROR_FIX.md',
    'CUDA_ERROR_SOLUTION.md': 'CUDA_ERROR_SOLUTION.md',
    'colab_cell_full_training_safe.py': 'colab_cell_full_training_safe.py'
}

for filename, filepath in files.items():
    try:
        url = base_url + filename
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'w') as f:
                f.write(response.text)
            print(f"✅ Downloaded: {filename}")
        else:
            print(f"❌ Failed to download {filename}: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")

print("\n✅ All files downloaded!")
```

## Step 5: Run the Fix

After downloading:

```python
# Run the automated fix
!python fix_cuda_error.py
```

## Expected Output

```
🔧 CUDA Error Fix - Comprehensive Solution
======================================================================

1️⃣ Clearing old checkpoints...
   ✅ Backed up checkpoints to: ./checkpoints_backup
   ✅ Cleared old checkpoints

2️⃣ Clearing PyTorch cache...
   ✅ CUDA cache cleared
   Memory allocated: 0.00 GB

4️⃣ Verifying dataset...
   ✅ File loaded: 27136 rows
   Emotion range: [0, 3]
   ✅ Emotion labels are valid [0-3]

======================================================================
✅ Fix completed!
```

## Step 6: Restart Runtime

After running the fix:
1. Click: **Runtime > Restart runtime**
2. Re-run setup cells (imports, installations)
3. Run training again

## Troubleshooting

### If Git Pull Fails

```python
# Reset local changes and pull
!git reset --hard HEAD
!git clean -fd
!git pull origin claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj
```

### If Branch Doesn't Exist

```python
# Switch to the correct branch
!git checkout claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj
```

### If Still Having Issues

Use the direct download method above - it always works!
