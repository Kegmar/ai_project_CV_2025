#!/usr/bin/env python3

import os
import re
import tkinter as tk
from PIL import Image, ImageTk
from picamera2 import Picamera2
from libcamera import controls  # for autofocus & AE constraints

# ----------------------------------------------------------------------
# CONFIG: set highlight behavior here (no UI changes needed)
#   "normal"   -> default exposure behavior
#   "highlight"-> preserve highlights more aggressively
#   "shadows"  -> protect shadows more
# ----------------------------------------------------------------------
AE_CONSTRAINT_MODE = "highlight"
# AE_CONSTRAINT_MODE = "normal"
# AE_CONSTRAINT_MODE = "shadows"


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raspberry Pi Camera Module 3 - Full Quality Preview")

        # Size of the preview window (display size, NOT capture size)
        self.display_width = 1280
        self.display_height = 720

        # Label to show the camera frames
        self.preview_label = tk.Label(self.root, bg="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Status label at the bottom
        self.status_label = tk.Label(self.root, text="Press SPACE to save photo", anchor="w")
        self.status_label.pack(fill=tk.X)

        # Initialize camera
        self.picam2 = Picamera2()
        self._configure_camera_full_res()

        self.photo = None
        self.running = True
        self.last_frame = None  # last full-res frame used for preview

        # Bind SPACE to save image
        self.root.bind("<space>", self.on_space)

        # Start updating frames
        self.update_frame()

        # Clean shutdown when window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_camera_full_res(self):
        """
        Configure the camera to use the full sensor resolution of
        the Raspberry Pi Camera Module 3 (12MP: 4608 x 2592).
        """
        full_res = (1920, 1080) # full FoV mode for Cam Module 3

        # IMPORTANT:
        # In Picamera2:
        #   "RGB888" -> B,G,R order
        #   "BGR888" -> R,G,B order (what PIL expects)
        camera_config = self.picam2.create_still_configuration(
            main={"size": full_res, "format": "BGR888"}  # <- THIS is what we want for PIL
        )
        self.picam2.configure(camera_config)
        self.picam2.start()

        # Enable continuous autofocus where supported
        try:
            self.picam2.set_controls({
                "AfMode": controls.AfModeEnum.Continuous,
                # Optional if you want faster AF:
                # "AfSpeed": controls.AfSpeedEnum.Fast,
            })
        except Exception as e:
            print("Autofocus controls not available or failed:", e)

        # ------------------ HIGHLIGHT / SHADOW BIAS ------------------
        # Map the AE_CONSTRAINT_MODE string to the libcamera enum.
        mode_map = {
            "normal": controls.AeConstraintModeEnum.Normal,
            "highlight": controls.AeConstraintModeEnum.Highlight,
            "shadows": controls.AeConstraintModeEnum.Shadows,
        }

        chosen_mode = mode_map.get(
            str(AE_CONSTRAINT_MODE).lower(),
            controls.AeConstraintModeEnum.Normal
        )

        try:
            self.picam2.set_controls({"AeConstraintMode": chosen_mode})
            print(f"AeConstraintMode set to: {AE_CONSTRAINT_MODE}")
        except Exception as e:
            print("Failed to set AeConstraintMode (not supported on this setup?):", e)
        # -------------------------------------------------------------

    def update_frame(self):
        if not self.running:
            return

        try:
            # Capture a full-resolution frame as a NumPy array (R,G,B)
            frame = self.picam2.capture_array("main")
            self.last_frame = frame  # keep full-res frame
        except Exception as e:
            print("Error capturing preview frame:", e)
            self.root.after(30, self.update_frame)
            return

        # Convert to PIL image for preview (explicitly say it's RGB)
        image = Image.fromarray(frame, mode="RGB")

        # Resize to fit the UI window (no crop, preserve aspect ratio)
        image = image.resize(
            (self.display_width, self.display_height),
            Image.LANCZOS
        )

        # Convert to ImageTk for Tkinter
        self.photo = ImageTk.PhotoImage(image)

        # Update label
        self.preview_label.configure(image=self.photo)

        # Schedule the next frame
        self.root.after(30, self.update_frame)

    def on_space(self, event=None):
        """Handle SPACE key: save current frame as next number in Photos folder."""
        try:
            # Fresh full-resolution capture for saving (best quality)
            full_res_frame = self.picam2.capture_array("main")
        except Exception as e:
            # Fallback: use last preview frame if fresh capture fails
            if self.last_frame is None:
                self.status_label.config(text="No frame yet, try again...")
                print("Error capturing save frame:", e)
                return
            full_res_frame = self.last_frame
            print("Using last_frame due to capture error:", e)

        try:
            filepath = self._get_next_photo_path()
            image = Image.fromarray(full_res_frame, mode="RGB")

            # Save at maximum quality (no chroma subsampling)
            image.save(
                filepath,
                "JPEG",
                quality=100,      # max quality
                subsampling=0,    # no chroma subsampling (4:4:4)
                optimize=True
            )

            self.status_label.config(text=f"Saved photo: {os.path.basename(filepath)}")
            print(f"Saved photo: {filepath}")
        except Exception as e:
            self.status_label.config(text=f"Error saving photo: {e}")
            print("Error saving photo:", e)

    def _get_next_photo_path(self):
        """
        Find the next integer filename in:
        /home/piuser/Photo taker/Photos
        -> 1.jpg, 2.jpg, 3.jpg, ...
        """
        photos_dir = "/home/piuser/Photo taker/Photos"
        os.makedirs(photos_dir, exist_ok=True)

        max_n = 20
        pattern = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

        for fname in os.listdir(photos_dir):
            m = pattern.match(fname)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n

        next_n = max_n + 1
        filename = f"{next_n}.jpg"
        return os.path.join(photos_dir, filename)

    def on_close(self):
        self.running = False
        try:
            self.picam2.stop()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1280x720")
    app = CameraApp(root)
    root.mainloop()
