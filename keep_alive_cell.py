# ============================================================================
# KEEP COLAB ALIVE - Add this as the FIRST cell in your notebook
# ============================================================================

from IPython.display import display, HTML, Javascript
import time

# JavaScript to keep Colab session alive
keep_alive_js = """
<script>
function ClickConnect(){
    var current_time = new Date().toLocaleTimeString();
    console.log("⏰ " + current_time + " - Keeping Colab session alive...");

    // Try multiple methods to keep alive
    try {
        // Method 1: Click connect button
        document.querySelector("#top-toolbar > colab-connect-button")?.shadowRoot?.querySelector("#connect")?.click();

        // Method 2: Simulate user activity
        document.dispatchEvent(new Event('mousemove'));

        console.log("✅ Keep-alive signal sent");
    } catch(e) {
        console.log("⚠️ Keep-alive error (usually safe to ignore):", e);
    }
}

// Run every 60 seconds (1 minute)
const keepAliveInterval = setInterval(ClickConnect, 60000);

// Also run immediately
ClickConnect();

console.log("✅✅✅ KEEP-ALIVE ACTIVATED ✅✅✅");
console.log("Session will be kept alive automatically!");
console.log("You can now switch tabs or minimize the browser.");
console.log("To stop: clearInterval(" + keepAliveInterval + ")");

// Display notification in notebook
var notification = document.createElement("div");
notification.innerHTML = `
    <div style="background-color: #d4edda; border: 2px solid #28a745; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h3 style="color: #155724; margin: 0 0 10px 0;">✅ Keep-Alive Activated!</h3>
        <p style="color: #155724; margin: 0;">
            Your Colab session will stay alive even if you switch tabs.<br>
            <strong>Check browser console (F12) to see keep-alive messages.</strong>
        </p>
    </div>
`;
document.querySelector("body").prepend(notification);
</script>
"""

# Display the JavaScript
display(HTML(keep_alive_js))

print("="*60)
print("🔌 KEEP-ALIVE SYSTEM ACTIVATED")
print("="*60)
print("✅ Session will stay alive automatically!")
print("✅ You can switch tabs or minimize browser")
print("✅ Check browser console (F12) to see activity")
print("")
print("⚠️ Note: Keep the browser tab open (minimized is OK)")
print("⚠️ Don't close the browser completely")
print("="*60)

# Also set a Python-side heartbeat
import threading

def python_heartbeat():
    """Python-side heartbeat to show activity"""
    while True:
        time.sleep(300)  # Every 5 minutes
        print(f"💓 Heartbeat at {time.strftime('%H:%M:%S')} - Session active")

# Start heartbeat in background
heartbeat_thread = threading.Thread(target=python_heartbeat, daemon=True)
heartbeat_thread.start()

print("💓 Python heartbeat started (prints every 5 minutes)")
