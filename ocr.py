"""
Módulo OCR para leitura de matrículas de veículos.

Tenta as bibliotecas disponíveis pela seguinte ordem:
  1. easyocr  – `pip install easyocr`
  2. pytesseract + Pillow  – `pip install pytesseract Pillow`
             (requer também Tesseract-OCR instalado no sistema)
  3. Entrada manual como fallback

Formatos de matrícula portugueses suportados:
  - Novo (pós-2005): AA-00-BB
  - Antigo: 00-AA-00
  - Mais antigo: AA-00-00
"""

import os
import re
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Padrões de matrícula portuguesa
# ---------------------------------------------------------------------------
_PATTERNS = [
    r"[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{2}",   # AA-00-BB (atual)
    r"\d{2}[-\s]?[A-Z]{2}[-\s]?\d{2}",       # 00-AA-00
    r"\d{2}[-\s]?\d{2}[-\s]?[A-Z]{2}",       # 00-00-AA
    r"[A-Z]{2}[-\s]?\d{2}[-\s]?\d{2}",       # AA-00-00
]

_DIGIT_FIXES = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_LETTER_FIXES = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}
_PLATE_MASKS = ("LLDDLL", "DDLLDD", "DDDDLL", "LLDDDD")
_BASE_DIR = Path(__file__).resolve().parent
_YOLO_CONFIG_DIR = _BASE_DIR / ".ultralytics"
try:
    _YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))
_YOLO_MODEL_CANDIDATES = (
    _BASE_DIR / "yolov8_model" / "best.pt",
    _BASE_DIR / "yolov8_model" / "runs" / "detect" / "train3" / "weights" / "best.pt",
)


def _normalize(text: str) -> str:
    """Converte para maiúsculas, remove caracteres inválidos e insere traços."""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(text) == 6 and "-" not in text:
        text = f"{text[:2]}-{text[2:4]}-{text[4:]}"
    return text


def normalize_license_plate(text: str) -> str | None:
    """Normaliza uma matrícula escrita manualmente ou devolvida pelo OCR."""
    return _extract_plate(text)


def _extract_plate(raw: str) -> str | None:
    """Tenta extrair uma matrícula do texto bruto devolvido pelo OCR."""
    upper = raw.upper()
    for pattern in _PATTERNS:
        matches = re.findall(pattern, upper)
        if matches:
            return _normalize(matches[0])

    compact = re.sub(r"[^A-Z0-9]", "", upper)
    for size in range(6, min(10, len(compact)) + 1):
        for start in range(0, len(compact) - size + 1):
            chunk = compact[start:start + size]
            for i in range(0, len(chunk) - 5):
                candidate = _fit_plate_masks(chunk[i:i + 6])
                if candidate:
                    return candidate
    return None


def _fit_plate_masks(chunk: str) -> str | None:
    for mask in _PLATE_MASKS:
        fixed = []
        ok = True
        for char, expected in zip(chunk, mask):
            if expected == "D":
                if char.isdigit():
                    fixed.append(char)
                elif char in _DIGIT_FIXES:
                    fixed.append(_DIGIT_FIXES[char])
                else:
                    ok = False
                    break
            else:
                if char.isalpha():
                    fixed.append(char)
                elif char in _LETTER_FIXES:
                    fixed.append(_LETTER_FIXES[char])
                else:
                    ok = False
                    break
        if ok:
            plate = "".join(fixed)
            return f"{plate[:2]}-{plate[2:4]}-{plate[4:]}"
    return None


def _image_variant_paths(image_path: str):
    paths = [image_path]
    try:
        from PIL import Image, ImageEnhance, ImageOps  # type: ignore
    except ImportError:
        return paths, None

    tmpdir = tempfile.TemporaryDirectory()
    try:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = img.size
        boxes = [
            (0, 0, width, height),
            (int(width * 0.08), int(height * 0.12), int(width * 0.90), int(height * 0.80)),
            (int(width * 0.08), int(height * 0.18), int(width * 0.84), int(height * 0.78)),
            (int(width * 0.14), int(height * 0.22), int(width * 0.78), int(height * 0.70)),
            (0, int(height * 0.38), width, height),
            (int(width * 0.45), int(height * 0.35), width, int(height * 0.92)),
            (int(width * 0.55), int(height * 0.45), int(width * 0.98), int(height * 0.88)),
        ]

        for index, box in enumerate(boxes):
            crop = img.crop(box)
            scale = max(2, min(4, 1200 // max(1, crop.width)))
            if scale > 1:
                crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
            gray = ImageOps.grayscale(crop)
            gray = ImageOps.autocontrast(gray)
            gray = ImageEnhance.Sharpness(gray).enhance(2.0)
            variant_path = os.path.join(tmpdir.name, f"plate_variant_{index}.png")
            gray.save(variant_path)
            paths.append(variant_path)
    except Exception:
        tmpdir.cleanup()
        return [image_path], None

    return paths, tmpdir


def _find_yolo_model() -> str | None:
    env_path = os.getenv("YOLO_PLATE_MODEL")
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(_YOLO_MODEL_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    return None


def _detect_plate_crop_paths(image_path: str):
    model_path = _find_yolo_model()
    if not model_path:
        return [], None

    try:
        from PIL import Image, ImageEnhance, ImageOps  # type: ignore
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return [], None

    tmpdir = tempfile.TemporaryDirectory()
    crop_paths = []
    try:
        model = YOLO(model_path)
        results = model.predict(image_path, conf=0.25, verbose=False)
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = img.size

        boxes = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((confidence, x1, y1, x2, y2))

        for index, (_, x1, y1, x2, y2) in enumerate(sorted(boxes, reverse=True)[:5]):
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            pad_x = box_w * 0.18
            pad_y = box_h * 0.35
            left = max(0, int(x1 - pad_x))
            top = max(0, int(y1 - pad_y))
            right = min(width, int(x2 + pad_x))
            bottom = min(height, int(y2 + pad_y))
            crop = img.crop((left, top, right, bottom))
            scale = max(2, min(5, 900 // max(1, crop.width)))
            if scale > 1:
                crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
            gray = ImageOps.grayscale(crop)
            gray = ImageOps.autocontrast(gray)
            gray = ImageEnhance.Sharpness(gray).enhance(2.5)
            crop_path = os.path.join(tmpdir.name, f"yolo_plate_{index}.png")
            gray.save(crop_path)
            crop_paths.append(crop_path)
    except Exception:
        tmpdir.cleanup()
        return [], None

    if not crop_paths:
        tmpdir.cleanup()
        return [], None
    return crop_paths, tmpdir


def _ocr_result_score(result, image_size) -> float:
    try:
        box, text, confidence = result
        points = [(float(point[0]), float(point[1])) for point in box]
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        image_width, image_height = image_size
        box_width = max(1.0, max_x - min_x)
        box_height = max(1.0, max_y - min_y)
        relative_width = box_width / max(1.0, float(image_width))
        relative_height = box_height / max(1.0, float(image_height))
        aspect = box_width / box_height
        compact_len = len(re.sub(r"[^A-Z0-9]", "", text.upper()))

        return (
            float(confidence) * 5.0
            + min(relative_width * 3.0, 2.0)
            + min(relative_height * 1.5, 1.0)
            + min(aspect / 5.0, 1.0)
            + min(compact_len / 6.0, 1.0)
        )
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Leitores OCR
# ---------------------------------------------------------------------------

def _read_yolo_then_ocr(image_path: str) -> str | None:
    crop_paths, tmpdir = _detect_plate_crop_paths(image_path)
    if not crop_paths:
        return None

    try:
        for crop_path in crop_paths:
            for reader_fn in (_read_easyocr, _read_pytesseract):
                plate = reader_fn(crop_path)
                if plate:
                    return plate
    finally:
        if tmpdir:
            tmpdir.cleanup()
    return None

def _read_easyocr(image_path: str) -> str | None:
    try:
        import easyocr  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image, ImageOps  # type: ignore
        reader = easyocr.Reader(["pt", "en"], verbose=False)
        paths, tmpdir = _image_variant_paths(image_path)
        candidates = []
        try:
            for path in paths:
                img = ImageOps.grayscale(Image.open(path))
                img_array = np.array(img)
                results = reader.readtext(
                    img_array,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
                    detail=1,
                    paragraph=False,
                )
                texts = [r[1] for r in results]
                for result in results:
                    text = result[1]
                    plate = _extract_plate(text)
                    if plate:
                        candidates.append((_ocr_result_score(result, img.size), plate))
                raw = " ".join(texts)
                plate = _extract_plate(raw)
                if plate:
                    candidates.append((0.5, plate))
        finally:
            if tmpdir:
                tmpdir.cleanup()
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1]
        return None
    except ImportError:
        return None
    except Exception:
        return None


def _read_pytesseract(image_path: str) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        paths, tmpdir = _image_variant_paths(image_path)
        try:
            for path in paths:
                img = Image.open(path)
                for psm in (7, 6, 11, 13):
                    cfg = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
                    raw = pytesseract.image_to_string(img, config=cfg)
                    plate = _extract_plate(raw)
                    if plate:
                        return plate
        finally:
            if tmpdir:
                tmpdir.cleanup()
        return None
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def read_license_plate(image_path: str | None = None, allow_manual: bool = True) -> str | None:
    """
    Lê a matrícula de um veículo via OCR ou entrada manual.

    Args:
        image_path: Caminho para o ficheiro de imagem (opcional).
                    Se None ou se o OCR falhar, pede entrada manual.

    Returns:
        Matrícula no formato XX-XX-XX (ex: "AB-12-CD") ou None se o OCR falhar
        e a entrada manual estiver desativada.
    """
    if image_path and os.path.isfile(image_path):
        print(f"[OCR] A processar imagem: {image_path}")

        readers = [
            (_read_yolo_then_ocr, "yolov8 + OCR"),
            (_read_easyocr, "easyocr"),
            (_read_pytesseract, "pytesseract"),
        ]
        for reader_fn, name in readers:
            plate = reader_fn(image_path)
            if plate:
                print(f"[OCR] Matrícula detetada ({name}): {plate}")
                return plate

        print("[OCR] Não foi possível detetar a matrícula automaticamente.")
    elif image_path:
        print(f"[OCR] Ficheiro não encontrado: {image_path}")

    if not allow_manual:
        return None

    # Fallback: entrada manual
    while True:
        entry = input("Introduza a matrícula (ex: AB-12-CD): ").strip().upper()
        plate = _normalize(entry)
        if re.match(r"^[A-Z0-9]{2}-[A-Z0-9]{2}-[A-Z0-9]{2}$", plate):
            return plate
        print("  Formato inválido. Use XX-XX-XX (letras ou dígitos em cada bloco).")
