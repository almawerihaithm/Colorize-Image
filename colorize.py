import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

# Get the current directory path
base_dir = os.path.dirname(os.path.abspath(__file__))

PROTOTXT = os.path.join(base_dir, 'mo/colorization_deploy_v2.prototxt')
POINTS = os.path.join(base_dir, 'mo/pts_in_hull.npy')
MODEL = os.path.join(base_dir, 'mo/colorization_release_v2.caffemodel')

# Load the Caffe model and points
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    return net, pts

# Function to prepare the model layers with the pre-trained cluster centers
def prepare_model(net, pts):
    class8_ab = net.getLayerId("class8_ab")
    conv8_313_rh = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8_ab).blobs = [pts.astype("float32")]
    net.getLayer(conv8_313_rh).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load and process the image
def process_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized_lab = cv2.resize(lab, (224, 224))
    L_channel = cv2.split(resized_lab)[0]
    L_channel -= 50
    return image, lab, L_channel

# Predict the ab channels using the model
def colorize_image(net, L_channel, original_shape):
    net.setInput(cv2.dnn.blobFromImage(L_channel))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (original_shape[1], original_shape[0]))
    return ab_channel

# Combine the L channel and predicted ab channels to get the colorized image
def merge_channels(lab, ab_channel):
    L_channel = cv2.split(lab)[0]
    colorized_lab = np.concatenate((L_channel[:, :, np.newaxis], ab_channel), axis=2)
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")
    return colorized_bgr

# Function to browse and select an image file
def browse_image():
    global original_image_path, original_img_label
    image_path = filedialog.askopenfilename()
    if image_path:
        original_image_path = image_path
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        original_img_label.config(image=img)
        original_img_label.image = img

# Function to colorize the selected image
def colorize():
    global colorized_img_label, original_image_path, colorized_image
    if original_image_path:
        # Read the image in BGR format
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            save_status_label.config(text="Failed to load image!", fg="red")
            return
        
        # Check if the image is grayscale
        if len(original_image.shape) < 3 or original_image.shape[2] == 1:
            save_status_label.config(text="Selected image is black-and-white! Please select a color image.", fg="red")
            return
        
        # Proceed with colorization
        net, pts = load_model()
        prepare_model(net, pts)
        _, lab, L_channel = process_image(original_image_path)
        ab_channel = colorize_image(net, L_channel, original_image.shape)
        colorized_image = merge_channels(lab, ab_channel)

        # Convert to PIL Image for display
        colorized_pil = Image.fromarray(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
        colorized_pil.thumbnail((300, 300))
        colorized_img = ImageTk.PhotoImage(colorized_pil)
        colorized_img_label.config(image=colorized_img)
        colorized_img_label.image = colorized_img
    else:
        save_status_label.config(text="No image selected!", fg="red")


# Function to save the colorized image
def save_image():
    global colorized_image
    if colorized_image is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
            save_status_label.config(text=f"Image saved at {save_path}", fg="light blue")
        else:
            save_status_label.config(text="Save operation cancelled", fg="orange")
    else:
        save_status_label.config(text="No colorized image to save!", fg="red")

# Function to reset the application
def reset_app():
    global original_image_path, colorized_image
    original_image_path = None
    colorized_image = None
    original_img_label.config(image=placeholder_img)
    colorized_img_label.config(image=placeholder_img)
    save_status_label.config(text="", fg="white")

# Function to exit the application
def exit_app():
    root.destroy()

# Create the Tkinter window
root = tk.Tk()
root.title("Image Colorizer")

# Set the background color
root.config(bg="black")

# Set the font style
font_style = ("Comic Sans MS", 20)

# Labels for displaying images with empty boxes initially
placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (300, 300), "gray"))

original_img_label = tk.Label(root, bg="black", image=placeholder_img)
original_img_label.grid(row=0, column=0, padx=10, pady=10)

colorized_img_label = tk.Label(root, bg="black", image=placeholder_img)
colorized_img_label.grid(row=0, column=1, padx=10, pady=10)

# Buttons
browse_button = tk.Button(root, text="Browse Photo", command=browse_image, bg="light blue", font=font_style)
browse_button.grid(row=1, column=0, pady=10)

colorize_button = tk.Button(root, text="Colorize", command=colorize, bg="light blue", font=font_style)
colorize_button.grid(row=1, column=1, pady=10)

# Save button placed under the other two buttons
save_button = tk.Button(root, text="Save Photo", command=save_image, bg="light blue", font=font_style)
save_button.grid(row=2, column=0, columnspan=2, pady=10)

# Label to show save status
save_status_label = tk.Label(root, text="", bg="black", fg="light blue", font=font_style)
save_status_label.grid(row=3, column=0, columnspan=2)

# Reset and Exit buttons
reset_button = tk.Button(root, text="Reset", command=reset_app, bg="light blue", font=font_style)
reset_button.grid(row=4, column=0, pady=10)

exit_button = tk.Button(root, text="Exit", command=exit_app, bg="light blue", font=font_style)
exit_button.grid(row=4, column=1, pady=10)

# Initialize colorized_image variable
colorized_image = None
original_image_path = None

root.mainloop()
