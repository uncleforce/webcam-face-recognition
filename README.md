# Webcam Face Recognition

Real-time face recognition from your webcam in a single Python script. Drop reference photos into a folder, run one command, and see names and confidence scores drawn live on the video feed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3](https://img.shields.io/badge/Python-3-blue.svg)

> **Privacy notice:** face images are biometric data. Only use this project on people who have given their consent, and read [Responsible Use & Privacy](#responsible-use--privacy) before you start.

## Project Overview

`face_system.py` compares the faces in front of your camera with the photos you place in a `persons/` folder. It is built on [`face_recognition`](https://github.com/ageitgey/face_recognition) (dlib) for detection and matching, and [OpenCV](https://opencv.org/) for capture and display.

It is designed to be easy to run and tweak: one script, one optional JSON config file, and a handful of keyboard shortcuts.

## Key Features

- **Live recognition** from any camera, with a bounding box, name and confidence percentage per face.
- **Three speed/accuracy modes:** `fast`, `balanced` and `accurate` (CNN detector).
- **Simple enrollment:** use one photo per person, or one sub-folder per person with several photos.
- **Encoding cache:** reference photos are encoded once and re-used on later runs (re-encoded automatically when a photo changes).
- **Frame skipping and downscaling** to keep the video smooth on ordinary laptops.
- **Arabic name rendering** on the video window (optional extra dependencies).
- **Runtime controls:** snapshot, pause, reload photos, and adjust the matching tolerance without restarting.
- **Session reports:** a JSON summary of who was recognized and when, plus an optional log file.
- **Optional unknown-face capture:** save crops of unrecognized faces, rate-limited by a cooldown.
- **Color-coded results:** green = recognized, red = `Unknown`, orange = `Low Confidence`.

## Demo

No screenshots are included yet. Pull requests with screenshots are welcome, but please only use images of people who have agreed to appear in them.

## Tech Stack

- Python 3
- [face_recognition](https://github.com/ageitgey/face_recognition) / [dlib](http://dlib.net/)
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- Optional: [Pillow](https://python-pillow.org/), [arabic-reshaper](https://github.com/mpcabd/python-arabic-reshaper), [python-bidi](https://github.com/MeirKriheli/python-bidi) for Arabic text

## Prerequisites

- **Python 3.** The script uses `str.isascii()`, so 3.7 is the minimum the code can run on. The version it has been tested with is listed in [Tested Environments](#tested-environments).
- **A working webcam** (built-in or USB).
- **A C++ build toolchain and CMake.** `dlib` is usually compiled from source during installation, which takes several minutes.

| OS | What to install first |
|----|------------------------|
| macOS | `xcode-select --install`, then `brew install cmake` |
| Windows | [CMake](https://cmake.org/download/) and the *Desktop development with C++* workload from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| Linux (Debian/Ubuntu) | `sudo apt install build-essential cmake` |

## Installation

```bash
git clone https://github.com/uncleforce/webcam-face-recognition.git
cd webcam-face-recognition

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell / cmd)

pip install --upgrade pip
pip install -r requirements.txt

# Optional: Arabic names on the video window
pip install -r requirements-arabic.txt
```

On Windows you can use `python` instead of `python3`.

### Tested Environments

| Environment | What was verified |
|-------------|-------------------|
| macOS and Windows (Python 3 latest at the time of testing) | Live webcam recognition, tested manually by the maintainer |
| Linux, Python 3.13, clean virtual environment | `pip install -r requirements.txt` and `requirements-arabic.txt`, script start-up, photo loading and Arabic label rendering (no camera available) |

Other platforms and Python versions are likely to work but have not been verified.

## Configuration

On first run the script creates a `config.json` next to where you run it, filled with the defaults. A copy of those defaults is provided as [`config.example.json`](config.example.json). Edit `config.json` and restart; missing keys fall back to the defaults.

| Key | Default | Description |
|-----|---------|-------------|
| `tolerance` | `0.5` | Maximum face distance accepted as a match. **Lower = stricter.** |
| `min_conf` | `60` | Minimum confidence (0-100) needed to show a name; otherwise `Low Confidence` is shown. See the note below. |
| `scale` | `0.25` | Frames are shrunk by this factor before detection, for speed. |
| `skip_frames` | `2` | Run recognition on every N-th frame. |
| `mode` | `"fast"` | `fast`, `balanced` or `accurate`. |
| `upsample` | `1` | Times to upsample the image when looking for faces. Higher finds smaller faces but is slower. |
| `camera` | `0` | Camera index. Can be overridden with `--cam`. |
| `width`, `height` | `1280`, `720` | Requested capture resolution. |
| `font_scale` | `0.7` | Label text size. |
| `show_conf` | `true` | Show the confidence percentage next to names. |
| `save_unknown` | `false` | Save crops of `Unknown` faces to `unknown_faces/`. |
| `unknown_cooldown` | `3` | Minimum seconds between two saved unknown faces. |
| `enhance_photos` | `false` | Apply contrast enhancement (CLAHE) to reference photos when loading them. |
| `logging` | `true` | Write events to `logs/face.log`. |
| `font_path` | `""` | Path to a `.ttf` font that supports Arabic, if none is found automatically. |

### Modes

| Mode | Detector | Encoding model | Notes |
|------|----------|----------------|-------|
| `fast` | HOG | small | Default. Works well on a CPU. |
| `balanced` | HOG | large | Better encodings, a bit slower. |
| `accurate` | CNN | large | Expect it to be slow unless dlib was built with CUDA. |

> **Note on `min_conf`:** with the default `tolerance` of `0.5`, the confidence formula stays above 60% until the face distance is almost exactly at the tolerance limit, so `Low Confidence` appears only for matches that are very close to being rejected. Raise `min_conf` if you want it to trigger more often.

## Usage

Run the script **from the project root**: the `persons/`, `logs/`, `screenshots/` and `unknown_faces/` folders, the cache and `config.json` are all created relative to your current directory.

### 1. Add reference photos

Put photos in `persons/`. Two layouts are supported and can be mixed:

```text
persons/
├── sara.jpg              -> "Sara"
├── omar_khan.png         -> "Omar Khan"
└── ahmed/                -> "Ahmed" (all photos in the folder belong to one person)
    ├── 1.jpg
    └── 2.jpg
```

- Use **one sub-folder per person** when you have several photos of the same person. If you name flat files `ahmed_1.jpg` and `ahmed_2.jpg`, they are treated as two different people ("Ahmed 1" and "Ahmed 2").
- Underscores in names become spaces and names are title-cased.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`.
- If a photo contains several faces, the largest one is used.
- Use clear, front-facing photos with good lighting.

### 2. Start the app

```bash
python face_system.py
```

```bash
# Use a different camera
python face_system.py --cam 1

# Use a different photo folder or config file
python face_system.py --persons ./team_photos --config ./my_config.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--persons` | `persons` | Folder with reference photos |
| `--config` | `config.json` | Path to the configuration file |
| `--cam` | from config | Camera index (overrides `camera` in the config) |

> **Language note:** the console messages and code comments are currently written in Arabic. The on-screen interface (HUD and key hints) is in English.

### 3. Keyboard shortcuts

Click on the video window first so it has focus.

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit |
| `S` | Save a snapshot to `screenshots/` |
| `R` | Reload photos from the persons folder |
| `Space` | Pause / resume recognition |
| `+` / `-` | Raise / lower the tolerance (range 0.3-0.7, step 0.02) |
| `C` | Print the current settings in the console |

Changes made with `+`/`-` apply only to the current session and are not written back to `config.json`.

### Output files

| Path | Content |
|------|---------|
| `logs/face.log` | Events log (when `logging` is `true`) |
| `logs/session_<timestamp>.json` | Session summary, written on exit if at least one person was recognized |
| `screenshots/` | Snapshots taken with `S` |
| `unknown_faces/` | Crops of unknown faces (when `save_unknown` is `true`) |
| `.faces_cache.pkl` | Cache of encoded reference photos |

## Project Structure

```text
webcam-face-recognition/
├── face_system.py             # The application
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt           # Core dependencies
├── requirements-arabic.txt    # Optional: Arabic text rendering
├── config.example.json        # Default settings (copy to config.json)
└── .gitignore
```

Created at runtime and excluded from Git: `persons/`, `logs/`, `screenshots/`, `unknown_faces/`, `config.json`, `.faces_cache.pkl`.

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `opencv-python` | Camera capture, drawing, window display | Yes |
| `numpy` | Array operations | Yes |
| `face_recognition` | Face detection and encoding (brings in `dlib` and Pillow) | Yes |
| `setuptools<81` | Keeps `pkg_resources` available for `face_recognition_models` | Yes |
| `pillow` | Draw Arabic text | Optional |
| `arabic-reshaper` | Shape Arabic letters | Optional |
| `python-bidi` | Right-to-left text ordering | Optional |

Use `opencv-python`, not `opencv-python-headless`: the script opens a GUI window.

## Troubleshooting

**`Please install face_recognition_models with this command...` even though it is installed**
`face_recognition_models` needs `pkg_resources`, which newer `setuptools` versions no longer ship. Run `pip install "setuptools<81"` (already included in `requirements.txt`).

**`dlib` fails to build during `pip install`**
Install CMake and a C++ compiler first (see [Prerequisites](#prerequisites)), open a fresh terminal, and retry. On Windows make sure the *Desktop development with C++* workload is installed. The build can take several minutes without printing output.

**The camera does not open**
Close other apps that may be using it (video call apps, browsers) and try `--cam 1` or `--cam 2`. On macOS, allow camera access for your terminal or editor under *System Settings -> Privacy & Security -> Camera*, then restart it.

**A photo is skipped ("no face found")**
Use a sharper, front-facing, well-lit photo where the face is large enough. Try `enhance_photos: true`.

**Everyone is `Unknown`, or the wrong person is matched**
Lower `tolerance` for fewer false matches, raise it if known people are not recognized (or use `+`/`-` live). Adding several photos per person in their own sub-folder helps. The `balanced` mode gives better encodings.

**Arabic names show as `?`**
Install the extras with `pip install -r requirements-arabic.txt`. If no suitable font is found automatically, set `font_path` in `config.json` to a `.ttf` file that supports Arabic.

**Low frame rate**
Increase `skip_frames`, decrease `scale`, lower `width`/`height`, or use `fast` mode. Avoid `accurate` without a CUDA-enabled dlib.

**An error is printed and the app exits**
The script prints only the error message, not the full traceback. When opening an issue, include that message together with your OS, Python version and how you installed `dlib`.

## Roadmap

Ideas for future work (not commitments):

- Replace the pickle cache with a format that is safe to load from untrusted files.
- A `--debug` flag to show full tracebacks.
- Automated tests for the non-camera logic (matching, photo discovery).
- Translated console messages.

## Responsible Use & Privacy

Face images and encodings are **biometric personal data**, and many jurisdictions regulate collecting and processing them.

- Obtain clear consent from everyone whose photo you enroll or whose face you may record.
- Check the privacy and data-protection laws that apply to you before deploying this anywhere other than a personal experiment.
- Everything stays on your machine: the project does not send data over the network. Reference photos, cache, logs, snapshots and unknown-face crops are stored in plain files in the working directory, so protect them accordingly.
- `.gitignore` excludes all of these paths. **Never commit or publish them.**
- Face recognition can be wrong. Do not use it as the only basis for decisions that affect people.
- The underlying models are distributed through third-party packages (`dlib`, `face_recognition_models`) under their own licenses. If you plan commercial use, review those licenses.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

Released under the [MIT License](LICENSE).
