from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from algorithms import astar_search, greedy_search, limited_depth_search, uniform_cost_search
from graph import CITIES, GRAPH, get_heuristic
from llm_client import get_city_attractions
from ocr import normalize_license_plate, read_license_plate


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None

        if request.method == "POST":
            result, error = _handle_search(request)

        return render_template(
            "web_index.html",
            cities=CITIES,
            result=result,
            error=error,
            selected=request.form if request.method == "POST" else {},
            algorithms=_algorithm_labels(),
        )

    return app


def _algorithm_labels() -> dict[str, str]:
    return {
        "ucs": "Custo Uniforme",
        "dls": "Profundidade Limitada",
        "greedy": "Procura Sôfrega",
        "astar": "A*",
    }


def _handle_search(form_request):
    start = form_request.form.get("start", "").strip()
    goal = form_request.form.get("goal", "").strip()
    algorithm = form_request.form.get("algorithm", "").strip()

    if start not in CITIES or goal not in CITIES:
        return None, "Escolhe uma cidade de origem e uma cidade de destino válidas."
    if start == goal:
        return None, "A cidade de origem e destino têm de ser diferentes."
    if algorithm not in _algorithm_labels():
        return None, "Escolhe um método de procura."

    image = form_request.files.get("plate_image")
    if image is None or not image.filename:
        return None, "Submete uma imagem da matrícula para efetuar o login por OCR."

    upload_path, upload_error = _save_upload(image)
    if upload_error:
        return None, upload_error

    plate = read_license_plate(str(upload_path), allow_manual=False)
    plate_source = "OCR"
    if not plate:
        manual_plate = normalize_license_plate(form_request.form.get("manual_plate", ""))
        if not manual_plate:
            return None, "Não foi possível reconhecer a matrícula na imagem submetida. Confirma se o EasyOCR está instalado ou usa a matrícula manual de apoio."
        plate = manual_plate
        plate_source = "Introdução manual"

    max_depth = _parse_depth(form_request.form.get("max_depth"))
    path, cost, iterations = _run_algorithm(algorithm, start, goal, max_depth)
    attractions = _build_attractions(path or [])

    return {
        "plate": plate,
        "plate_source": plate_source,
        "start": start,
        "goal": goal,
        "algorithm": _algorithm_labels()[algorithm],
        "path": path,
        "cost": cost,
        "iterations": iterations,
        "uses_heuristic": algorithm in {"greedy", "astar"},
        "attractions": attractions,
        "image_url": f"uploads/{upload_path.name}",
        "max_depth": max_depth,
    }, None


def _save_upload(image):
    suffix = Path(image.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return None, "Usa uma imagem em formato PNG, JPG, JPEG, WEBP ou BMP."

    safe_name = secure_filename(image.filename)
    filename = f"{uuid4().hex}_{safe_name}"
    upload_path = UPLOAD_DIR / filename
    image.save(upload_path)
    return upload_path, None


def _parse_depth(raw_value: str | None) -> int:
    try:
        value = int(raw_value or "10")
    except ValueError:
        return 10
    return max(1, min(value, 30))


def _run_algorithm(algorithm: str, start: str, goal: str, max_depth: int):
    if algorithm == "ucs":
        return uniform_cost_search(GRAPH, start, goal)
    if algorithm == "dls":
        return limited_depth_search(GRAPH, start, goal, max_depth=max_depth)
    if algorithm == "greedy":
        return greedy_search(GRAPH, start, goal, get_heuristic)
    return astar_search(GRAPH, start, goal, get_heuristic)


def _build_attractions(path: list[str]) -> dict[str, str]:
    return {city: get_city_attractions(city) for city in path}


app = create_app()


if __name__ == "__main__":
    app.run(debug=False)
