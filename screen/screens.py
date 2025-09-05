import tkinter as tk
from screeninfo import get_monitors, Monitor
from PIL import Image, ImageTk
import io

# Local screens
class Screen:
    def __init__(self, id: int, info: Monitor):
        self.w = info.width
        self.h = info.height

        window = tk.Toplevel()
        window.geometry(f"{info.width}x{info.height}+{info.x}+{info.y}")
        window.overrideredirect(True)
        window.title(f"Monitor {id}")
        window.config(bg="black")
        self.window = window

        text = f"Connected\nL{id}\n{info.width}*{info.height}"
        print(text)

        label = tk.Label(self.window)
        label.config(text=text, font=("Helvetica", 100), fg="white", bg="black")
        label.pack()
        self.label = label

    def show_image(self, img: Image.Image):
        self.photo = ImageTk.PhotoImage(img.resize((self.w, self.h)))
        self.label.config(image=self.photo)

    def destroy(self):
        self.window.destroy()

screens = []

def start_ui():
    root = tk.Tk()
    label = tk.Label(root, text="Hello", font=("Helvetica", 100))
    label.pack()

    screens_reset()
    root.mainloop()

def set_image(id: int, img):
    img=Image.open(io.BytesIO(img))
    screens[id].show_image(img)

def screens_reset():
    for s in screens:
        s.destroy()
    screens.clear()

    monitors = get_monitors()
    for info in monitors:
        if not info.is_primary:
            sid = len(screens)
            screens.append(Screen(sid, info))