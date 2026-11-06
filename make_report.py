"""Builds reports/Report.pdf. Run: python make_report.py [--docker]"""
import glob, os, subprocess, sys, datetime
from fpdf import FPDF

# ===== EDIT THESE =====
NAME = "Dhanya R"
ROLL = "24AI021"
COURSE = "SEL Lab"
ASSIGNMENT = "Lab-6,7"
REPO_URL = "https://github.com/DhanyaR-07"
CHALLENGES = [
       ("Docker Desktop could not run: 'The paging file is too small for this operation to complete' (Windows virtual memory exhausted), so docker build/run was not executed.",
        "Not resolved in the time available. Dockerfile, .dockerignore and docker-compose.yml are written and included as untested work; the fix would be to increase the Windows paging file / free RAM and rerun."),
       ("OpenBLAS memory allocation error while running numpy/scikit-learn.",
        "Set OPENBLAS_NUM_THREADS=1 and ran uvicorn without --reload."),
       ("Docker Desktop not found at the default path, and PowerShell vs cmd syntax differences.",
        "Located the install under AppData\\Local\\Programs\\DockerDesktop and used PowerShell syntax."),
   ]
# ======================

sys.path.insert(0, os.getcwd())
os.makedirs("reports/evidence", exist_ok=True)
clean = lambda s: str(s).encode("latin-1", "replace").decode("latin-1")


def run(cmd, timeout=900):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        out = f"$ {cmd}\n{p.stdout}\n{p.stderr}"
        return p.returncode == 0, out
    except Exception as e:
        return False, f"$ {cmd}\nERROR: {e}"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 5, clean(f"{NAME} | ID: {ROLL} | {COURSE}"), new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, clean(ASSIGNMENT), new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, clean(f"{NAME} | {ROLL} | Page {self.page_no()}"), align="C")


pdf = PDF()
pdf.set_auto_page_break(True, 15)
pdf.add_page()


def h(t):
    pdf.ln(2); pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, clean(t), new_x="LMARGIN", new_y="NEXT")


def p(t):
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, clean(t), new_x="LMARGIN", new_y="NEXT")


def code(t, last=60):
    pdf.set_font("Courier", "", 7)
    lines = [l[:115] for l in t.splitlines() if l.strip()][-last:]
    pdf.multi_cell(0, 3.5, clean("\n".join(lines)) or "(no output)", new_x="LMARGIN", new_y="NEXT")


# ---------- run steps ----------
steps = [
    ("Data download", "python -m src.data"),
    ("Training", "python -m src.train"),
    ("Evaluation", "python -m src.evaluate"),
    ("Prediction (separate process)", "python -m src.predict"),
    ("Tests (pytest)", "python -m pytest -v"),
    ("Docker images", "docker image ls"),
    ("Docker containers", "docker ps -a"),
    ("Docker logs", "docker logs ml-api"),
]
if "--docker" in sys.argv:
    steps.insert(5, ("Docker build", "docker build -t ml-inference-api:1.0 ."))
results = []
for name, cmd in steps:
    ok, out = run(cmd)
    results.append((name, cmd, ok, out))
    open(f"reports/evidence/{name.split()[0].lower()}_{len(results)}.txt", "w", encoding="utf-8").write(out)

# ---------- API validation via TestClient ----------
api_rows = []
try:
    from fastapi.testclient import TestClient
    from src.api import app
    V = {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
         "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}
    cases = [("Valid input", V), ("Missing field", {k: v for k, v in V.items() if k != "MedInc"}),
             ("Wrong type", {**V, "HouseAge": "abc"}), ("Invalid business value", {**V, "MedInc": -5})]
    with TestClient(app) as c:
        r = c.get("/health"); api_rows.append(("Health check", r.status_code, r.text[:70]))
        for n, body in cases:
            r = c.post("/predict", json=body); api_rows.append((n, r.status_code, r.text[:70]))
except Exception as e:
    api_rows.append(("API checks", "ERROR", str(e)[:90]))

# ---------- build PDF ----------
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 9, "MLOps Lab Report", new_x="LMARGIN", new_y="NEXT")
p(f"Student: {NAME}    ID: {ROLL}\nCourse: {COURSE}\nGenerated: {datetime.datetime.now():%d %b %Y %H:%M}\nGitHub: {REPO_URL}")

h("1. Objectives")
p("Refactor the house-price ML project into modules (data, train, evaluate, predict), serve the saved "
  "scikit-learn Pipeline through FastAPI (GET /health, POST /predict) and package it with Docker/Compose "
  "so another developer can reproduce it.")

h("2. Architecture")
p("Data -> Cleaning -> Feature Engineering (imputer + scaler) -> Training (RandomForest) -> Evaluation -> "
  "Saved Pipeline (models/model.joblib) -> FastAPI POST /predict -> Docker container.")

h("3. Pipeline stages (input / output / purpose / failure)")
p("Loading: file path -> DataFrame; fails if file/columns missing.\n"
  "Cleaning: raw DataFrame -> clean DataFrame; fails if valid rows are dropped wrongly.\n"
  "Feature engineering: columns -> model-ready features; fails on train/inference mismatch.\n"
  "Training: X_train,y_train -> fitted Pipeline; fails by overfitting or data leakage.\n"
  "Evaluation: held-out data -> metrics; fails if test data leaks into training.\n"
  "Prediction: new records -> predictions; fails on wrong feature names/order.")

h("4. Online vs batch inference")
p("Online: one/few records, immediate response, latency matters (POST /predict). Batch: many records or a CSV, "
  "scheduled or background job, throughput matters. For a CSV I would add an upload endpoint that queues a "
  "background job and writes a results file, or run a separate scheduled batch script using the same pipeline.")

h("5. Docker concepts")
p("Works on one laptop but not another because of different OS, Python and library versions. Containers ship the "
  "same runtime and dependencies. They do not guarantee identical hardware, performance, security, or that data/"
  "models are present (host paths are not visible inside a container unless mounted).")

h("6. Implementation status and evidence")
for name, cmd, ok, out in results:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, clean(f"[{'PASS' if ok else 'FAIL / NOT COMPLETED'}] {name}"), new_x="LMARGIN", new_y="NEXT")
    code(out, 25)

h("7. API validation (FastAPI TestClient)")
code("\n".join(f"{n:<24}{s!s:<8}{b}" for n, s, b in api_rows))

h("8. Results")
mp = "reports/metrics.json"
code(open(mp).read() if os.path.exists(mp) else "metrics.json not found - evaluation not completed")

shots = sorted(glob.glob("reports/screenshots/*.png") + glob.glob("reports/screenshots/*.jpg"))
h("9. Screenshots")
p(f"{len(shots)} screenshot(s) attached on the following pages.")
for s in shots:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, clean(os.path.basename(s)), new_x="LMARGIN", new_y="NEXT")
    pdf.image(s, w=190, h=225, keep_aspect_ratio=True)

pdf.add_page()
h("10. Challenges and resolutions")
for c1, c2 in CHALLENGES:
    p(f"- Problem: {c1}\n  Resolution: {c2}")

h("11. Reflection")
p("I learned how to separate ML code into modules, save one Pipeline for both training and inference, validate "
  "API inputs with Pydantic, and package the service with Docker. Items marked FAIL above were not completed "
  "in the available time and are listed honestly.")

pdf.output("reports/Report.pdf")
print("Created reports/Report.pdf")