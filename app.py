import os
import sqlite3
import json
from datetime import datetime
from uuid import uuid4

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from model.predict import CLASS_NAMES, predict_scan
from utils.report_generator import build_interpretation_text, generate_medical_report

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


def resolve_persist_dir(path: str | None) -> str | None:
    if not path:
        return None
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return None
    return path


PERSIST_DIR = resolve_persist_dir(os.environ.get("PERSIST_DIR"))

DB_DIR = os.path.join(PERSIST_DIR, "database") if PERSIST_DIR else os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "patients.db")

UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
REPORT_DIR = os.path.join(STATIC_DIR, "reports")

PERSIST_UPLOAD_DIR = os.path.join(PERSIST_DIR, "uploads") if PERSIST_DIR else None
PERSIST_REPORT_DIR = os.path.join(PERSIST_DIR, "reports") if PERSIST_DIR else None
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-in-production")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)


class User(UserMixin):
    def __init__(self, user_id: int, username: str, password_hash: str):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_storage_paths() -> None:
    os.makedirs(DB_DIR, exist_ok=True)

    if PERSIST_DIR:
        os.makedirs(PERSIST_UPLOAD_DIR, exist_ok=True)
        os.makedirs(PERSIST_REPORT_DIR, exist_ok=True)

        for link_path, target_path in (
            (UPLOAD_DIR, PERSIST_UPLOAD_DIR),
            (REPORT_DIR, PERSIST_REPORT_DIR),
        ):
            if os.path.islink(link_path):
                continue
            if os.path.exists(link_path):
                continue
            try:
                os.symlink(target_path, link_path)
            except OSError:
                pass

    if not os.path.islink(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
    if not os.path.islink(REPORT_DIR):
        os.makedirs(REPORT_DIR, exist_ok=True)


def init_db() -> None:
    ensure_storage_paths()

    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_uid TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            notes TEXT,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (created_by) REFERENCES users (id)
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            uploaded_by INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            prob_normal REAL NOT NULL,
            prob_adenocarcinoma REAL NOT NULL,
            prob_squamous REAL NOT NULL,
            prob_large_cell REAL DEFAULT 0,
            probability_json TEXT,
            report_path TEXT NOT NULL,
            gradcam_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients (id),
            FOREIGN KEY (uploaded_by) REFERENCES users (id)
        );
        """
    )
    existing_columns = [row["name"] for row in conn.execute("PRAGMA table_info(scans)").fetchall()]
    if "prob_large_cell" not in existing_columns:
        conn.execute("ALTER TABLE scans ADD COLUMN prob_large_cell REAL DEFAULT 0")
    if "probability_json" not in existing_columns:
        conn.execute("ALTER TABLE scans ADD COLUMN probability_json TEXT")
    conn.commit()
    conn.close()


@login_manager.user_loader
def load_user(user_id: str):
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
    conn.close()
    if not row:
        return None
    return User(row["id"], row["username"], row["password_hash"])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_patient_uid() -> str:
    return f"PT-{datetime.now().strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"


@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3 or len(password) < 6:
            flash("Username must be 3+ chars and password must be 6+ chars.", "danger")
            return render_template("register.html")

        conn = get_db_connection()
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            flash("Username already exists.", "warning")
            return render_template("register.html")

        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db_connection()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if row and check_password_hash(row["password_hash"], password):
            login_user(User(row["id"], row["username"], row["password_hash"]))
            flash("Login successful.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


@app.route("/patients/add", methods=["GET", "POST"])
@login_required
def add_patient():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        notes = request.form.get("notes", "").strip()

        if not full_name:
            flash("Patient name is required.", "danger")
            return redirect(url_for("add_patient"))

        try:
            age_value = int(age) if age else None
        except ValueError:
            flash("Age must be a valid number.", "danger")
            return redirect(url_for("add_patient"))

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO patients (patient_uid, full_name, age, gender, phone, notes, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generate_patient_uid(),
                full_name,
                age_value,
                gender,
                phone,
                notes,
                current_user.id,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
        conn.close()
        flash("Patient added successfully.", "success")
        return redirect(url_for("upload_scan"))

    conn = get_db_connection()
    patients = conn.execute(
        "SELECT * FROM patients WHERE created_by = ? ORDER BY created_at DESC",
        (current_user.id,),
    ).fetchall()
    conn.close()
    return render_template("patients.html", patients=patients)


@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db_connection()

    total_patients = conn.execute(
        "SELECT COUNT(*) AS count FROM patients WHERE created_by = ?", (current_user.id,)
    ).fetchone()["count"]

    total_scans = conn.execute(
        "SELECT COUNT(*) AS count FROM scans WHERE uploaded_by = ?", (current_user.id,)
    ).fetchone()["count"]

    distribution_rows = conn.execute(
        """
        SELECT prediction_label, COUNT(*) AS total
        FROM scans
        WHERE uploaded_by = ?
        GROUP BY prediction_label
        """,
        (current_user.id,),
    ).fetchall()

    distribution = {label: 0 for label in CLASS_NAMES}
    for row in distribution_rows:
        distribution[row["prediction_label"]] = row["total"]

    recent_predictions = conn.execute(
        """
        SELECT s.id, s.prediction_label, s.confidence, s.created_at, p.patient_uid, p.full_name
        FROM scans s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.uploaded_by = ?
        ORDER BY s.created_at DESC
        LIMIT 10
        """,
        (current_user.id,),
    ).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_patients=total_patients,
        total_scans=total_scans,
        distribution=distribution,
        recent_predictions=recent_predictions,
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_scan():
    conn = get_db_connection()
    patients = conn.execute(
        "SELECT id, patient_uid, full_name FROM patients WHERE created_by = ? ORDER BY created_at DESC",
        (current_user.id,),
    ).fetchall()

    if request.method == "POST":
        patient_id = request.form.get("patient_id", "").strip()
        file = request.files.get("scan_image")

        if not patient_id:
            flash("Please select a patient.", "danger")
            conn.close()
            return redirect(url_for("upload_scan"))

        patient = conn.execute(
            "SELECT * FROM patients WHERE id = ? AND created_by = ?",
            (patient_id, current_user.id),
        ).fetchone()
        if not patient:
            flash("Patient not found.", "danger")
            conn.close()
            return redirect(url_for("upload_scan"))

        if not file or file.filename == "":
            flash("Please upload a CT scan image.", "danger")
            conn.close()
            return redirect(url_for("upload_scan"))

        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed: png, jpg, jpeg", "danger")
            conn.close()
            return redirect(url_for("upload_scan"))

        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = secure_filename(f"scan_{uuid4().hex}.{ext}")
        abs_image_path = os.path.join(UPLOAD_DIR, filename)
        file.save(abs_image_path)

        relative_image_path = f"uploads/{filename}"

        try:
            prediction = predict_scan(abs_image_path, generate_cam=True, cam_dir=REPORT_DIR)
        except FileNotFoundError as exc:
            flash(str(exc), "danger")
            conn.close()
            return redirect(url_for("upload_scan"))
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")
            conn.close()
            return redirect(url_for("upload_scan"))

        report_filename = f"report_{uuid4().hex}.pdf"
        abs_report_path = os.path.join(REPORT_DIR, report_filename)
        relative_report_path = f"reports/{report_filename}"

        probability_map = prediction["probabilities"]
        interpretation_text = build_interpretation_text(
            prediction["predicted_class"], prediction["confidence"]
        )

        generate_medical_report(
            output_path=abs_report_path,
            patient={
                "patient_uid": patient["patient_uid"],
                "full_name": patient["full_name"],
                "age": patient["age"],
                "gender": patient["gender"],
                "phone": patient["phone"],
            },
            image_path=abs_image_path,
            prediction={
                "label": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "probabilities": probability_map,
                "interpretation": interpretation_text,
                "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        gradcam_rel = None
        if prediction.get("gradcam_path"):
            gradcam_filename = os.path.basename(prediction["gradcam_path"])
            gradcam_rel = f"reports/{gradcam_filename}"

        conn.execute(
            """
            INSERT INTO scans (
                patient_id, uploaded_by, image_path, prediction_label, confidence,
                prob_normal, prob_adenocarcinoma, prob_squamous, prob_large_cell, probability_json,
                report_path, gradcam_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient["id"],
                current_user.id,
                relative_image_path,
                prediction["predicted_class"],
                prediction["confidence"],
                probability_map.get("Normal", 0.0),
                probability_map.get("Adenocarcinoma", 0.0),
                probability_map.get("Squamous Cell Carcinoma", 0.0),
                probability_map.get("Large Cell Carcinoma", 0.0),
                json.dumps(probability_map),
                relative_report_path,
                gradcam_rel,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        scan_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
        conn.commit()

        scan_row = conn.execute(
            """
            SELECT s.*, p.patient_uid, p.full_name, p.age, p.gender
            FROM scans s
            JOIN patients p ON p.id = s.patient_id
            WHERE s.id = ?
            """,
            (scan_id,),
        ).fetchone()
        conn.close()
        probability_map_for_view = probability_map
        if scan_row["probability_json"]:
            try:
                probability_map_for_view = json.loads(scan_row["probability_json"])
            except json.JSONDecodeError:
                pass

        flash("Prediction completed and report generated.", "success")
        return render_template("result.html", scan=scan_row, probability_map=probability_map_for_view)

    conn.close()
    return render_template("upload.html", patients=patients)


@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    records = conn.execute(
        """
        SELECT s.*, p.patient_uid, p.full_name
        FROM scans s
        JOIN patients p ON p.id = s.patient_id
        WHERE s.uploaded_by = ?
        ORDER BY s.created_at DESC
        """,
        (current_user.id,),
    ).fetchall()
    conn.close()
    return render_template("history.html", records=records)


@app.route("/reports/<int:scan_id>/download")
@login_required
def download_report(scan_id: int):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT report_path FROM scans WHERE id = ? AND uploaded_by = ?",
        (scan_id, current_user.id),
    ).fetchone()
    conn.close()

    if not row:
        flash("Report not found.", "danger")
        return redirect(url_for("history"))

    abs_path = os.path.join(BASE_DIR, "static", row["report_path"])
    if not os.path.exists(abs_path):
        flash("Report file is missing on disk.", "danger")
        return redirect(url_for("history"))

    return send_file(abs_path, as_attachment=True)


init_db()


if __name__ == "__main__":
    app.run(debug=True)
