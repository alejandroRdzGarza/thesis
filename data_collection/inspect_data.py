import numpy as np
import cv2

# The path of your desired inspect steps
path = "dataset/episodes/episode_0000/step_0052.npz"
d = np.load(path)

print(f"\nLoaded file: {path}")
print("=" * 60)


for key in d.files:
    val = d[key]

    print(f"\nKey: {key}")
    print(f"  type : {type(val)}")
    print(f"  dtype: {val.dtype}")
    print(f"  shape: {val.shape}")

    if (
        isinstance(val, np.ndarray)
        and val.ndim == 3
        and val.shape[2] == 3
        and val.dtype == np.uint8
    ):
        print("  -> Detected IMAGE, showing preview...")

        # OpenCV expects BGR
        img_bgr = cv2.cvtColor(val, cv2.COLOR_RGB2BGR)
        cv2.imshow(key, img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    else:
        # Avoid printing huge arrays
        if val.size <= 20:
            print(f"  value: {val}")
        else:
            print("  value: <large array, not printed>")

print("\nDone.")
