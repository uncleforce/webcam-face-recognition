# -*- coding: utf-8 -*-
# التعرف على الوجوه من الكاميرا
# يقارن الوجوه اللي قدام الكاميرا بالصور اللي في مجلد persons

import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition

# هذي اختيارية، تنفع بس لو أسماء الأشخاص بالعربي
try:
    from PIL import Image, ImageDraw, ImageFont
    import arabic_reshaper
    try:
        from bidi import get_display
    except ImportError:
        from bidi.algorithm import get_display
    HAS_AR = True
except ImportError:
    HAS_AR = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
CACHE_FILE = ".faces_cache.pkl"

# (طريقة كشف الوجه, موديل التشفير)
MODES = {
    "fast": ("hog", "small"),
    "balanced": ("hog", "large"),
    "accurate": ("cnn", "large"),
}

DEFAULT_CFG = {
    "tolerance": 0.5,         # كل ما قلّ الرقم صار التطابق أدق
    "min_conf": 60,           # أقل نسبة ثقة (من 100) عشان نقبل الاسم
    "scale": 0.25,            # نصغّر الإطار قبل المعالجة عشان السرعة
    "skip_frames": 2,         # نعالج كل كم إطار
    "mode": "fast",           # fast / balanced / accurate
    "upsample": 1,
    "camera": 0,
    "width": 1280,
    "height": 720,
    "font_scale": 0.7,
    "show_conf": True,
    "save_unknown": False,
    "unknown_cooldown": 3,    # ثواني بين كل حفظ لوجه مجهول
    "enhance_photos": False,  # تحسين تباين الصور وقت التحميل
    "logging": True,
    "font_path": "",          # خط يدعم العربي لو احتجته
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

RED = (0, 0, 255)
ORANGE = (0, 165, 255)
GREEN = (0, 200, 0)
SPECIAL = {"Unknown": RED, "Low Confidence": ORANGE}


def dist_to_conf(dist, tol):
    # نحوّل المسافة لنسبة ثقة من 100
    if dist > tol:
        lin = (1.0 - dist) / ((1.0 - tol) * 2.0)
        return max(lin, 0.0) * 100
    lin = 1.0 - dist / (tol * 2.0)
    return (lin + (1.0 - lin) * math.pow((lin - 0.5) * 2.0, 0.2)) * 100


def name_from(text):
    return text.replace("_", " ").strip().title()


def list_photos(folder):
    # يقبل صور مباشرة داخل المجلد، أو مجلد فرعي لكل شخص (persons/ahmed/1.jpg)
    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() in IMG_EXT:
            yield name_from(item.stem), item
        elif item.is_dir():
            for f in sorted(item.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    yield name_from(item.name), f


def boost_contrast(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def encode_photo(path, model, enhance):
    img = face_recognition.load_image_file(str(path))

    # الصور الكبيرة تبطّئ التحميل، نصغرها
    h, w = img.shape[:2]
    if max(h, w) > 1000:
        k = 1000 / max(h, w)
        img = cv2.resize(img, (int(w * k), int(h * k)))
    if enhance:
        img = boost_contrast(img)

    locs = face_recognition.face_locations(img)
    if not locs:
        return None

    # لو فيه أكثر من وجه ناخذ الأكبر
    areas = [(b - t) * (r - l) for t, r, b, l in locs]
    best = int(np.argmax(areas))
    encs = face_recognition.face_encodings(img, [locs[best]], model=model)
    return encs[0] if encs else None


class FaceID:
    def __init__(self, persons_dir="persons", cfg_path="config.json", cam=None):
        self.persons_dir = Path(persons_dir)
        self.cfg_path = Path(cfg_path)
        self.cfg = self.read_cfg()
        if cam is not None:
            self.cfg["camera"] = cam

        for d in (self.persons_dir, "screenshots", "unknown_faces", "logs"):
            os.makedirs(d, exist_ok=True)

        self.log = logging.getLogger("faceid")
        self.log.setLevel(logging.INFO)
        if not self.log.handlers:
            h = logging.FileHandler("logs/face.log", encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.log.addHandler(h)
        self.log.disabled = not self.cfg["logging"]

        self.enc = []
        self.names = []
        self.results = []
        self.paused = False
        self.seen = {}
        self.started = datetime.now()
        self.last_unknown = 0
        self._font = None
        self._font_warned = False

        self.load_faces()

    # ---------- الإعدادات ----------

    def read_cfg(self):
        cfg = dict(DEFAULT_CFG)
        if self.cfg_path.exists():
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    cfg.update(json.load(f))
            except Exception as e:
                print(f"ما قدرت أقرأ ملف الإعدادات ({e})، بنمشي على الافتراضي")
        else:
            try:
                with open(self.cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=4, ensure_ascii=False)
            except OSError:
                pass

        if cfg["mode"] not in MODES:
            print(f"الوضع '{cfg['mode']}' مو موجود، بنشتغل على fast")
            cfg["mode"] = "fast"
        return cfg

    # ---------- تحميل الصور ----------

    def load_cache(self, model):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("model") == model and data.get("enhance") == self.cfg["enhance_photos"]:
                return data["files"]
        except Exception:
            pass
        return {}

    def save_cache(self, model, files):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump({"model": model,
                             "enhance": self.cfg["enhance_photos"],
                             "files": files}, f)
        except OSError:
            pass

    def load_faces(self):
        print("نحمّل الصور شوي...")
        _, model = MODES[self.cfg["mode"]]
        old = self.load_cache(model)
        new = {}
        ok = bad = 0

        for name, path in list_photos(self.persons_dir):
            key = str(path)
            st = path.stat()
            stamp = (st.st_mtime, st.st_size)

            if key in old and old[key][0] == stamp:
                enc = old[key][1]
            else:
                try:
                    enc = encode_photo(path, model, self.cfg["enhance_photos"])
                except Exception as e:
                    print(f"  مشكلة بملف {path.name}: {e}")
                    self.log.error("فشل تحميل %s: %s", path, e)
                    bad += 1
                    continue
            new[key] = (stamp, enc)

            if enc is None:
                print(f"  ما لقيت وجه في: {path.name}")
                bad += 1
                continue

            self.enc.append(enc)
            self.names.append(name)
            ok += 1
            print(f"  تمام: {name}")
            self.log.info("تم تحميل %s", name)

        self.save_cache(model, new)
        print(f"\nانتهينا: {ok} صورة تمام، {bad} فشلت")
        print(f"عدد الأشخاص: {len(set(self.names))}\n")

    # ---------- الكشف والمطابقة ----------

    def detect(self, frame):
        detector, model = MODES[self.cfg["mode"]]
        s = self.cfg["scale"]
        tol = self.cfg["tolerance"]

        small = cv2.resize(frame, (0, 0), fx=s, fy=s)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(
            rgb, number_of_times_to_upsample=self.cfg["upsample"], model=detector)
        encs = face_recognition.face_encodings(rgb, locs, model=model)

        out = []
        for (t, r, b, l), enc in zip(locs, encs):
            name, conf = "Unknown", 0.0

            if self.enc:
                dists = face_recognition.face_distance(self.enc, enc)
                i = int(np.argmin(dists))
                if dists[i] <= tol:
                    conf = dist_to_conf(dists[i], tol)
                    name = self.names[i] if conf >= self.cfg["min_conf"] else "Low Confidence"

            if name not in SPECIAL:
                now = datetime.now().strftime("%H:%M:%S")
                rec = self.seen.setdefault(name, {"count": 0, "first": now, "last": now})
                rec["count"] += 1
                rec["last"] = now

            box = tuple(int(v / s) for v in (t, r, b, l))
            out.append({"box": box, "name": name, "conf": conf})
        return out

    def save_unknown(self, frame):
        if not self.cfg["save_unknown"]:
            return
        now = time.time()
        if now - self.last_unknown < self.cfg["unknown_cooldown"]:
            return

        H, W = frame.shape[:2]
        for res in self.results:
            if res["name"] != "Unknown":
                continue
            t, r, b, l = res["box"]
            crop = frame[max(t, 0):min(b, H), max(l, 0):min(r, W)]
            if crop.size == 0:
                continue
            fn = f"unknown_faces/unknown_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(fn, crop)
            self.last_unknown = now

    # ---------- الرسم ----------

    def get_font(self):
        if self._font is not None:
            return self._font
        size = max(12, int(self.cfg["font_scale"] * 30))
        paths = ([self.cfg["font_path"]] if self.cfg["font_path"] else []) + FONT_CANDIDATES

        # لازم التخطيط الأساسي، لأن raqm يقلب الاتجاه مرة ثانية فوق مكتبة bidi
        layout = ImageFont.Layout.BASIC if hasattr(ImageFont, "Layout") else ImageFont.LAYOUT_BASIC
        for p in paths:
            if p and os.path.exists(p):
                try:
                    self._font = ImageFont.truetype(p, size, layout_engine=layout)
                    return self._font
                except Exception:
                    continue
        return None

    def arabic_patch(self, text, bg):
        # OpenCV ما يرسم عربي، فنرسم النص بـ PIL ونلصقه
        font = self.get_font() if HAS_AR else None
        if font is None:
            if not self._font_warned:
                self._font_warned = True
                print("فيه اسم بحروف عربية بس ما قدرت أرسمه على الشاشة.")
                print("نزّل المكتبات: pip install pillow arabic-reshaper python-bidi")
                print("ولو الخط ما انلقى حط مساره في font_path بملف config.json")
            return None

        shaped = get_display(arabic_reshaper.reshape(text))
        l, t, r, b = font.getbbox(shaped)
        pad = 6
        img = Image.new("RGB", (r - l + pad * 2, b - t + pad * 2), (bg[2], bg[1], bg[0]))
        ImageDraw.Draw(img).text((pad - l, pad - t), shaped, font=font, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def put_label(self, frame, text, left, top, bottom, color):
        fs = self.cfg["font_scale"]
        left = max(left, 0)

        patch = None
        if not text.isascii():
            patch = self.arabic_patch(text, color)
            if patch is None:
                text = "".join(c if c.isascii() else "?" for c in text)

        if patch is None:
            (tw, th), base = cv2.getTextSize(text, FONT, fs, 1)
            bar = th + base + 10
            y1 = top - bar if top - bar >= 0 else bottom
            cv2.rectangle(frame, (left, y1), (left + tw + 12, y1 + bar), color, -1)
            cv2.putText(frame, text, (left + 6, y1 + bar - base - 4),
                        FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
            return

        ph, pw = patch.shape[:2]
        H, W = frame.shape[:2]
        y1 = top - ph if top - ph >= 0 else bottom
        x2, y2 = min(left + pw, W), min(y1 + ph, H)
        if x2 > left and y2 > y1:
            frame[y1:y2, left:x2] = patch[:y2 - y1, :x2 - left]

    def draw_faces(self, frame):
        for res in self.results:
            top, right, bottom, left = res["box"]
            name = res["name"]
            color = SPECIAL.get(name, GREEN)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = name
            if self.cfg["show_conf"] and name not in SPECIAL:
                label = f"{name} {res['conf']:.0f}%"
            self.put_label(frame, label, left, top, bottom, color)

    def draw_hud(self, frame, fps):
        h, w = frame.shape[:2]

        top = frame[0:30, :]
        top[:] = cv2.addWeighted(np.zeros_like(top), 0.6, top, 0.4, 0)
        info = (f"People: {len(set(self.names))}  Faces: {len(self.results)}  "
                f"Mode: {self.cfg['mode']}  Tol: {self.cfg['tolerance']:.2f}  FPS: {fps}")
        if self.paused:
            info += "  [PAUSED]"
        cv2.putText(frame, info, (10, 20), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        bottom = frame[h - 28:h, :]
        bottom[:] = cv2.addWeighted(np.zeros_like(bottom), 0.6, bottom, 0.4, 0)
        keys = "Q quit | S snapshot | R reload | SPACE pause | +/- tolerance | C config"
        cv2.putText(frame, keys, (10, h - 9), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ---------- الأزرار ----------

    def on_key(self, key, frame):
        if key in (ord("s"), ord("S")):
            fn = f"screenshots/shot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(fn, frame)
            print(f"انحفظت اللقطة: {fn}")

        elif key in (ord("r"), ord("R")):
            print("نعيد تحميل الصور...")
            self.enc.clear()
            self.names.clear()
            self.results = []
            self.load_faces()

        elif key in (ord("c"), ord("C")):
            print("\nالإعدادات الحالية:")
            for k, v in self.cfg.items():
                print(f"  {k}: {v}")
            print()

        elif key == 32:
            self.paused = not self.paused
            print("وقفنا المعالجة" if self.paused else "كمّلنا المعالجة")

        elif key in (ord("+"), ord("=")):
            self.cfg["tolerance"] = round(min(0.7, self.cfg["tolerance"] + 0.02), 2)
            print(f"الحساسية صارت {self.cfg['tolerance']}")

        elif key in (ord("-"), ord("_")):
            self.cfg["tolerance"] = round(max(0.3, self.cfg["tolerance"] - 0.02), 2)
            print(f"الحساسية صارت {self.cfg['tolerance']}")

    # ---------- التشغيل ----------

    def run(self):
        if not self.enc:
            print("ما فيه ولا وجه محمّل. حط صور في مجلد", self.persons_dir, "وجرب مرة ثانية")
            return False

        cap = cv2.VideoCapture(self.cfg["camera"])
        if not cap.isOpened():
            print("الكاميرا ما فتحت، تأكد انها مو شغالة في برنامج ثاني")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])

        print(f"شغّال على وضع {self.cfg['mode']} - اضغط Q عشان تطلع\n")

        win = "Face Recognition"
        skip = max(1, int(self.cfg["skip_frames"]))
        frame_no = 0
        fps, counter, t0 = 0, 0, time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ما قدرت أقرأ من الكاميرا")
                    break

                if not self.paused and frame_no % skip == 0:
                    self.results = self.detect(frame)
                    self.save_unknown(frame)
                frame_no += 1

                self.draw_faces(frame)

                counter += 1
                if time.time() - t0 >= 1:
                    fps, counter, t0 = counter, 0, time.time()
                self.draw_hud(frame, fps)

                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                if key != 255:
                    self.on_key(key, frame)
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            print("\nوقفتها بيدك")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.finish()

        return True

    def finish(self):
        if not self.seen:
            print("ما تعرفنا على أحد بهالجلسة")
            return

        print("\nاللي تعرفنا عليهم:")
        for name, rec in sorted(self.seen.items(), key=lambda x: -x[1]["count"]):
            print(f"  {name}: {rec['count']} مرة (من {rec['first']} إلى {rec['last']})")

        end = datetime.now()
        data = {
            "start": self.started.isoformat(timespec="seconds"),
            "end": end.isoformat(timespec="seconds"),
            "duration": str(end - self.started).split(".")[0],
            "seen": self.seen,
            "settings": self.cfg,
        }
        fn = f"logs/session_{end:%Y%m%d_%H%M%S}.json"
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"\nحفظنا تقرير الجلسة في {fn}")
        except OSError as e:
            print(f"ما قدرت أحفظ التقرير: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="تعرّف على الوجوه من الكاميرا")
    p.add_argument("--persons", default="persons", help="مجلد صور الأشخاص")
    p.add_argument("--config", default="config.json", help="ملف الإعدادات")
    p.add_argument("--cam", type=int, default=None, help="رقم الكاميرا (الافتراضي من الإعدادات)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        app = FaceID(args.persons, args.config, args.cam)
        return app.run()
    except Exception as e:
        print(f"صار خطأ: {e}")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
