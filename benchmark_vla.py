import requests
import base64
from PIL import Image
import io
import numpy as np
import time

# Create a realistic test image (224x224 like the actual inference)
np.random.seed(42)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
pil_img = Image.fromarray(img)
buf = io.BytesIO()
pil_img.save(buf, format='PNG')
b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

payload = {'image_base64': b64, 'instruction': 'grip the red cube'}

print("=" * 60)
print("OpenVLA 7B Inference Benchmark - GPU (MPS) Performance Test")
print("=" * 60)
print("\nModel: OpenVLA 7B (7 billion parameters)")
print("Device: Apple Metal Performance Shaders (MPS)")
print("Input: 224x224 RGB image\n")

times = []
success_count = 0

for i in range(3):
    print(f"Test {i+1}/3...", end=" ", flush=True)
    start = time.time()
    try:
        r = requests.post('http://127.0.0.1:8000/act', json=payload, timeout=120)
        elapsed = time.time() - start
        result = r.json()
        action = result.get('action', [])
        times.append(elapsed)
        success_count += 1
        print(f"✓ {elapsed:.2f}s - Action shape: {len(action)}")
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"✗ TIMEOUT after {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Error after {elapsed:.2f}s: {str(e)[:40]}")

print("\n" + "=" * 60)
if success_count > 0:
    avg_time = np.mean(times)
    print(f"Successful inferences: {success_count}/3")
    print(f"Average inference time: {avg_time:.2f}s")
    if success_count > 1:
        print(f"Min: {np.min(times):.2f}s, Max: {np.max(times):.2f}s")
    print("=" * 60)
    
    print("\n📊 Performance Analysis:")
    print(f"  • Single inference: ~{avg_time:.1f} seconds")
    print(f"  • Throughput: {1/avg_time:.2f} inferences/second")
    
    if avg_time < 5:
        print("\n✅ EXCELLENT: GPU acceleration is working well!")
        print("   Suitable for real-time control with fast inference")
    elif avg_time < 15:
        print("\n⚠️  MODERATE: GPU is being used but inference is still slow")
        print("   Likely causes:")
        print("   - Model parameters disk-offloading")
        print("   - Memory transfer overhead")
        print("   - MPS implementation limitations (7B model)")
    else:
        print("\n❌ SLOW: Inference is very slow despite GPU")
        print("   Recommendations:")
        print("   - Fall back to CPU for faster inference")
        print("   - Use smaller model (3B instead of 7B)")
        print("   - Implement caching or batch processing")
else:
    print("❌ No successful inferences - server may not be responding")
    print("=" * 60)
