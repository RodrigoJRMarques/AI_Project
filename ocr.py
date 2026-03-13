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


# ---------------------------------------------------------------------------
# Padrões de matrícula portuguesa
# ---------------------------------------------------------------------------
_PATTERNS = [
    r"[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{2}",   # AA-00-BB (atual)
    r"\d{2}[-\s]?[A-Z]{2}[-\s]?\d{2}",       # 00-AA-00
    r"[A-Z]{2}[-\s]?\d{2}[-\s]?\d{2}",       # AA-00-00
]


def _normalize(text: str) -> str:
    """Converte para maiúsculas, remove caracteres inválidos e insere traços."""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(text) == 6 and "-" not in text:
        text = f"{text[:2]}-{text[2:4]}-{text[4:]}"
    return text


def _extract_plate(raw: str) -> str | None:
    """Tenta extrair uma matrícula do texto bruto devolvido pelo OCR."""
    upper = raw.upper()
    for pattern in _PATTERNS:
        matches = re.findall(pattern, upper)
        if matches:
            return _normalize(matches[0])
    return None


# ---------------------------------------------------------------------------
# Leitores OCR
# ---------------------------------------------------------------------------

def _read_easyocr(image_path: str) -> str | None:
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["pt", "en"], verbose=False)
        results = reader.readtext(image_path)
        raw = " ".join(r[1] for r in results)
        return _extract_plate(raw)
    except ImportError:
        return None
    except Exception:
        return None


def _read_pytesseract(image_path: str) -> str | None:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        img = Image.open(image_path)
        cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        raw = pytesseract.image_to_string(img, config=cfg)
        return _extract_plate(raw)
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ponto de entrada público
# ---------------------------------------------------------------------------

def read_license_plate(image_path: str | None = None) -> str:
    """
    Lê a matrícula de um veículo via OCR ou entrada manual.

    Args:
        image_path: Caminho para o ficheiro de imagem (opcional).
                    Se None ou se o OCR falhar, pede entrada manual.

    Returns:
        Matrícula no formato XX-XX-XX (ex: "AB-12-CD").
    """
    if image_path and os.path.isfile(image_path):
        print(f"[OCR] A processar imagem: {image_path}")

        for reader_fn, name in [(_read_easyocr, "easyocr"), (_read_pytesseract, "pytesseract")]:
            plate = reader_fn(image_path)
            if plate:
                print(f"[OCR] Matrícula detetada ({name}): {plate}")
                return plate

        print("[OCR] Não foi possível detetar a matrícula automaticamente.")
    elif image_path:
        print(f"[OCR] Ficheiro não encontrado: {image_path}")

    # Fallback: entrada manual
    while True:
        entry = input("Introduza a matrícula (ex: AB-12-CD): ").strip().upper()
        plate = _normalize(entry)
        if re.match(r"^[A-Z0-9]{2}-[A-Z0-9]{2}-[A-Z0-9]{2}$", plate):
            return plate
        print("  Formato inválido. Use XX-XX-XX (letras ou dígitos em cada bloco).")
