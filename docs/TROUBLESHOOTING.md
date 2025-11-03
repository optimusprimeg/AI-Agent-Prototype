# Troubleshooting Training Issues

## "Terminated" Error Still Occurring?

If you're still seeing:
```
Terminated
```

And the warning:
```
UserWarning: 'pin_memory' argument is set as true but no accelerator is found
```

**You're running the OLD code!** Follow these steps:

### Step 1: Verify Your Code Version

Run this check script:
```bash
python check_version.py
```

If it says you have an OLD version, continue to Step 2.

### Step 2: Update Your Code

```bash
# Pull the latest changes
git pull origin copilot/fix-135869538-1088827371-3115b681-fff6-436b-a386-90ee8910cb52

# Verify you're on the right branch
git branch

# Check the latest commit
git log --oneline -1
# Should show: "548d1d8 Add CPU-optimized training..."
```

### Step 3: Clear Python Cache

Python caches compiled bytecode which can cause it to run old code:

```bash
# Remove all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Remove all .pyc files
find . -name "*.pyc" -delete 2>/dev/null || true
```

### Step 4: Verify Again

```bash
python check_version.py
```

Should now show: ✅ You have the CPU-optimized version!

### Step 5: Run Training

```bash
python src/models/fine_tuned_model.py --quick
```

**Expected output (NEW version):**
```
🚀 Quick training mode enabled
   Batch size will be auto-adjusted for your system

============================================================
System Resources:
  CPU Cores: 4
  RAM: 14.5 GB
  CUDA Available: False
============================================================

Auto-adjusted batch size: 8 (based on 14.5GB RAM)
```

**NOT (OLD version):**
```
🚀 Quick training mode enabled (1 epoch, batch size 64)  # OLD!
   Expected time: 2-3 minutes (GPU) or 8-10 minutes (CPU)
```

## Still Having Issues?

### Check Your Current Code Version

```bash
# See what commit you're on
git log --oneline -3

# Should see:
# 548d1d8 Add CPU-optimized training with auto resource detection and visualizations
# 1de1189 Optimize training speed: reduce default epochs to 3, add --quick mode
# 92632e6 Add comprehensive deliverables summary and final documentation
```

### Verify Pin Memory Setting

```bash
grep "dataloader_pin_memory" src/models/fine_tuned_model.py
```

Should output:
```
dataloader_pin_memory=False,  # Critical: Disable pin_memory for CPU
```

If it says `True` or doesn't exist, you have the wrong version!

### Force Reinstall

If nothing else works:

```bash
# Stash any local changes
git stash

# Hard reset to the correct commit
git reset --hard 548d1d8

# Clear cache again
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Try training
python src/models/fine_tuned_model.py --quick
```

## Key Differences: OLD vs NEW

| Feature | OLD (broken) | NEW (fixed) |
|---------|--------------|-------------|
| Batch size | Fixed 64 | Auto 2-16 |
| Pin memory | True ❌ | False ✅ |
| Message | "batch size 64" | "auto-adjusted" |
| Sequence length | 128 | 64 |
| System info | None | Displayed |
| Pre-tokenization | No | Yes |

## Contact

If you're still stuck after following ALL these steps, please:
1. Run `python check_version.py` and share the output
2. Share the output of `git log --oneline -1`
3. Share the first 5 lines of training output

This will help debug the issue!
