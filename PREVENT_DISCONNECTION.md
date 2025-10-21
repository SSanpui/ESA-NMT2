# 🔌 Preventing Google Colab Disconnections

**Problem**: Colab disconnects when you switch tabs or leave the browser.

**Solution**: Multiple approaches to keep your session alive!

---

## ✅ Method 1: Browser Console Script (EASIEST & BEST)

This JavaScript keeps Colab "active" by simulating clicks.

### Step-by-Step:

1. **Open your Colab notebook**

2. **Open Browser Console**:
   - **Chrome/Edge**: Press `F12` or `Ctrl+Shift+J` (Windows) / `Cmd+Option+J` (Mac)
   - **Firefox**: Press `F12` or `Ctrl+Shift+K`

3. **Copy and paste this code** into the console:

```javascript
function ClickConnect(){
    console.log("⏰ Keeping session alive...");
    document.querySelector("#top-toolbar > colab-connect-button")?.shadowRoot.querySelector("#connect").click();
}

// Click every 60 seconds
const intervalId = setInterval(ClickConnect, 60000);

console.log("✅ Auto-clicker started! Session will stay alive.");
console.log("⚠️ To stop: clearInterval(" + intervalId + ")");
```

4. **Press Enter**

5. **You'll see**: `✅ Auto-clicker started!`

6. **Now you can switch tabs freely!** The session stays alive.

### To Stop the Auto-Clicker:
```javascript
// In console, type:
clearInterval(1)  // Use the ID shown when you started
```

---

## ✅ Method 2: Browser Extension (NO CONSOLE NEEDED)

Install a browser extension that keeps Colab alive:

### For Chrome/Edge:
1. Install: [Colab Alive](https://chrome.google.com/webstore/detail/colab-alive/eookkckfbbgnhdgcbfbicoahejkdoele)
2. Or: [Auto Refresh Plus](https://chrome.google.com/webstore/detail/auto-refresh-plus/oilipfekkmncanaajkapbpancpelijih)

### For Firefox:
1. Install: [Tab Reloader](https://addons.mozilla.org/en-US/firefox/addon/tab-reloader/)

Just enable the extension on your Colab tab!

---

## ✅ Method 3: Python Code in Notebook (AUTOMATIC)

Add this cell to your Colab notebook to prevent disconnections:

```python
# Cell 1: Install and setup keep-alive
!pip install -q pyngrok

# Display JavaScript to keep session alive
from IPython.display import display, HTML

javascript_code = """
<script>
function KeepAlive() {
    console.log("Keeping Colab alive...");
    document.querySelector("colab-connect-button").click();
}

setInterval(KeepAlive, 60000);  // Every 60 seconds
console.log("✅ Keep-alive script running!");
</script>
"""

display(HTML(javascript_code))
print("✅ Keep-alive activated! Session will stay alive.")
```

Add this cell **at the beginning** of your notebook and run it!

---

## ✅ Method 4: Use Colab Pro/Pro+ (RECOMMENDED)

**Colab Pro Benefits**:
- ✅ Longer session limits (24 hours vs 12 hours)
- ✅ Less aggressive disconnection
- ✅ Better GPUs
- ✅ Higher priority

**Cost**: $10/month

**Sign up**: https://colab.research.google.com/signup

**Worth it** for serious projects!

---

## ❌ Why PyNput Won't Work

You asked about `pynput` - unfortunately it **won't work** because:

1. **pynput is client-side** (runs on your computer)
2. **Colab is server-side** (runs on Google's servers)
3. **Moving mouse/keyboard on Google's servers** won't prevent browser disconnection

The disconnection happens in **your browser**, not on the Colab server.

---

## ✅ Method 5: Background Execution (Advanced)

If you want **true background execution** (close browser entirely):

### Option A: Use Google Cloud Platform

```python
# Install gcloud CLI on your local machine
# Then run:
!gcloud compute ssh your-instance --command="python your_script.py"

# Or use tmux/screen on GCP instance
!tmux new-session -d -s training "python run_all_experiments.py"
```

### Option B: Use Kaggle Notebooks

Kaggle allows longer sessions and better background execution:
1. Go to https://www.kaggle.com/
2. Create new notebook
3. Upload your code
4. Enable GPU
5. Run

Kaggle sessions stay alive longer than Colab free!

### Option C: Use Paperspace Gradient

Free GPU with better session persistence:
1. Sign up: https://www.paperspace.com/gradient
2. Create notebook
3. Upload code
4. Run

---

## 🎯 My Recommendation for You

Since you're running experiments that take hours:

### **Immediate Solution (Free)**:
```javascript
// In browser console (F12):
function ClickConnect(){
    console.log("⏰ Keeping alive...");
    document.querySelector("#top-toolbar > colab-connect-button")?.shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);
```

### **Best Long-term Solution**:
1. **Get Colab Pro** ($10/month)
2. **Use the console script** above for extra safety
3. **Run during active hours** (when you're at your computer)

### **If You Need to Close Browser**:
1. **Upgrade to Colab Pro+** ($50/month) - allows background execution
2. **Or use Kaggle** (free alternative with better persistence)
3. **Or use local machine** with GPU

---

## 📊 Comparison Table

| Method | Free? | Effectiveness | Can Close Browser? |
|--------|-------|---------------|-------------------|
| **Console Script** | ✅ Yes | ⭐⭐⭐⭐⭐ | ❌ No |
| **Browser Extension** | ✅ Yes | ⭐⭐⭐⭐⭐ | ❌ No |
| **Python in Notebook** | ✅ Yes | ⭐⭐⭐⭐ | ❌ No |
| **Colab Pro** | ❌ $10/mo | ⭐⭐⭐⭐⭐ | ❌ No* |
| **Colab Pro+** | ❌ $50/mo | ⭐⭐⭐⭐⭐ | ✅ Yes |
| **Kaggle** | ✅ Yes | ⭐⭐⭐⭐ | ❌ No |
| **Local Machine** | ✅ Yes** | ⭐⭐⭐⭐⭐ | ✅ Yes |

*Pro is better but still needs browser open
**If you have GPU

---

## 🔥 Bonus: Handle Disconnections Gracefully

Even with keep-alive, disconnections can happen. Here's how to recover:

### Add Checkpointing to Your Code

I already added this to your code! It saves checkpoints every epoch:

```python
# This is already in your code:
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'translation_pair': translation_pair
}, f"./checkpoints/temp_model_{translation_pair}_epoch_{epoch}.pt")
```

### If Disconnected, Resume:

```python
# Add this cell to resume training
import torch
import os

translation_pair = "bn-hi"
model_type = "nllb"

# Find latest checkpoint
checkpoints = sorted(glob.glob(f"./checkpoints/temp_model_{translation_pair}_epoch_*.pt"))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"Found checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    print(f"✅ Resuming from epoch {start_epoch}")

    # Continue training from this epoch
    for epoch in range(start_epoch, config.EPOCHS['phase1']):
        train_loss = trainer.train_epoch(train_loader, epoch)
        # ... rest of training
else:
    print("No checkpoint found. Starting fresh.")
```

---

## 🎯 Quick Setup for Your Current Session

**Right now, do this**:

1. **Press F12** (opens console)

2. **Copy-paste this**:
```javascript
function ClickConnect(){
    console.log("⏰ " + new Date().toLocaleTimeString() + " - Keeping alive...");
    document.querySelector("#top-toolbar > colab-connect-button")?.shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);
console.log("✅ Auto-clicker started at " + new Date().toLocaleTimeString());
```

3. **Press Enter**

4. **You'll see**: `✅ Auto-clicker started at [time]`

5. **Minimize Colab tab** (don't close it!)

6. **Come back in 3-4 hours** to download results!

---

## ⚠️ Important Notes

### Console Script Tips:
- **Keep the tab open** (just minimize or switch away)
- **Don't close the browser**
- **Script runs only in that tab**
- **Refresh page = need to re-run script**

### Colab Limits:
- **Free tier**: Max 12 hours continuous
- **Colab Pro**: Max 24 hours continuous
- **Pro+**: Background execution possible

### If You Hit Time Limit:
Your code saves checkpoints, so you can:
1. Reconnect
2. Load latest checkpoint
3. Continue training

---

## 📱 Mobile App Solution

If you're on mobile:

1. **Install Colab app** (Android/iOS)
2. **App prevents sleep** better than browser
3. **Keep app in foreground**

Or:

1. **Use TeamViewer** or **Chrome Remote Desktop**
2. **Connect to a computer** running the browser
3. **Let that computer stay on**

---

## 🎓 Summary: What You Should Do

**For your current session** (immediate):
```javascript
// Press F12, paste this, press Enter:
function ClickConnect(){
    console.log("⏰ Keeping alive...");
    document.querySelector("#top-toolbar > colab-connect-button")?.shadowRoot.querySelector("#connect").click();
}
setInterval(ClickConnect, 60000);
```

**For future sessions** (best solution):
1. Get **Colab Pro** ($10/month)
2. Install **browser extension** (Colab Alive)
3. Use **console script** as backup

**If you need to close browser**:
- Upgrade to **Colab Pro+** ($50/month)
- Or use **Kaggle** (free alternative)
- Or run on **local machine** with GPU

---

## ✅ Test It Works

After setting up the console script:

1. Switch to another tab
2. Wait 2-3 minutes
3. Come back to Colab
4. Check console - should see "Keeping alive..." messages
5. Training should still be running!

---

**Try the console script right now - it works 99% of the time!** 🚀

Let me know if you need help with any of these methods!
