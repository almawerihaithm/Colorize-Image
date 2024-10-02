
# Image Colorizer Using Deep Learning

This project is a Python-based GUI application that allows users to colorize black-and-white images using a pre-trained deep learning model. The model is based on a Caffe framework, and the GUI is implemented using Tkinter. The colorization model predicts the `a` and `b` color channels in the LAB color space based on the given `L` channel from a grayscale image.

## Features
- Browse and load a grayscale image.
- Use a pre-trained model to colorize the image.
- Display the original and colorized images side by side.
- Save the colorized image to a file.
- Reset the application or exit it easily.

## Prerequisites

Make sure you have the following dependencies installed:

1. **Python 3.x**: The project is built using Python 3.
2. **Tkinter**: Tkinter is used for the graphical user interface.
3. **OpenCV (cv2)**: This is used for image processing and model loading.
4. **NumPy**: Required for array manipulation.
5. **Pillow (PIL)**: Used for image handling in the GUI.

You can install these libraries via `pip`:

```bash
pip install opencv-python-headless numpy Pillow
```

## Pre-trained Model Files

You need the following model files to run the project. Place them in the `mo` directory under the project root:
- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

You can download the model files from [OpenCV's colorization model repository](https://github.com/richzhang/colorization).

## Project Structure

```
├── main.py                  # The main application script
├── mo/                      # Directory containing model files
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
├── README.md                # This file
```

## Usage

1. **Clone the repository** or download the project files.

2. **Prepare the environment** by installing the required dependencies.

3. **Download the model files** (`prototxt`, `caffemodel`, `npy`) and place them in the `mo/` directory.

4. **Run the application** by executing the `main.py` script:

```bash
python main.py
```

5. **Using the application**:
   - Click "Browse Photo" to select an image file.
   - Click "Colorize" to generate a colorized version of the image.
   - You can save the colorized image using the "Save Photo" button.
   - Reset or exit the app using the respective buttons.

## How It Works

1. The application uses a Caffe deep learning model to predict the `a` and `b` color channels based on the lightness (`L`) channel of the grayscale image.
2. The selected image is resized and converted into the LAB color space, where the neural network operates.
3. The predicted `a` and `b` channels are merged back with the `L` channel to produce a fully colorized image.
4. The result is displayed in the GUI and can be saved in various formats.

## Acknowledgments

The pre-trained model and the colorization algorithm are based on the research by Richard Zhang, Phillip Isola, Alexei A. Efros, and their work in image colorization.

- [Link to research paper](https://richzhang.github.io/colorization/)

## License

This project is open-source and available under the [MIT License](LICENSE).
