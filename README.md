# Lung Cancer Detection Web Application using CNN

A complete Flask + TensorFlow mini project for:
- user authentication
- patient registration and history
- CT image upload and prediction
- automatic medical PDF report generation
- analytics dashboard with Chart.js
- optional Grad-CAM explainability

## Tech Stack
- Backend: Flask
- ML: TensorFlow/Keras CNN
- Image Processing: OpenCV
- Database: SQLite
- Frontend: HTML/CSS/Bootstrap/JavaScript
- Charts: Chart.js
- PDF: ReportLab

## Dataset
Use Kaggle dataset: **Lung and Colon Cancer Histopathological Images**.
Keep only lung folders:
- `lung_n`
- `lung_aca`
- `lung_scc`

Place dataset in `dataset/` in either format:
1. `dataset/lung_n`, `dataset/lung_aca`, `dataset/lung_scc`
2. `dataset/train/...` and `dataset/test/...` with same lung class folders

## Project Structure
```
lung_cancer_project/
  dataset/
  static/
    css/
    js/
    uploads/
    reports/
  templates/
    base.html
    index.html
    login.html
    register.html
    dashboard.html
    upload.html
    result.html
    history.html
    patients.html
  model/
    train_model.py
    predict.py
    lung_cancer_model.h5   (generated after training)
  database/
    patients.db            (auto-created)
  utils/
    image_preprocess.py
    report_generator.py
  app.py
  requirements.txt
  README.md
```

## Installation (VS Code)
1. Open this folder in VS Code.
2. Open terminal and create virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

Required command from specification:
```powershell
pip install tensorflow flask opencv-python numpy pandas matplotlib scikit-learn pillow reportlab flask-login
```

## Train the CNN Model
Run:
```powershell
python model/train_model.py
```
This saves model at:
- `model/lung_cancer_model.h5`

## Run the Web App
```powershell
python app.py
```
Open browser:
- `http://127.0.0.1:5000`

## Workflow
1. Register user and login.
2. Add patient in **Patients** page.
3. Go to **Upload Scan**, select patient and upload CT image.
4. App predicts one of:
   - Normal
   - Adenocarcinoma
   - Squamous Cell Carcinoma
   - Large Cell Carcinoma (if present in trained dataset)
5. PDF report is auto-generated in `static/reports/`.
6. Use **History** to view previous predictions and download old reports.

## Deploy on Vercel
Deployment files added:
- `vercel.json`
- `api/index.py`
- `.vercelignore`

### Steps
1. Install Vercel CLI:
   ```powershell
   npm i -g vercel
   ```
2. Login:
   ```powershell
   vercel login
   ```
3. From project root deploy:
   ```powershell
   vercel
   ```
4. For production deployment:
   ```powershell
   vercel --prod
   ```

### Important Vercel Limits for This Project
- Vercel Serverless is **ephemeral**:
  - SQLite file (`database/patients.db`) and generated reports/uploads are not persistent.
- Running TensorFlow model inference inside Vercel functions is usually not reliable due to package size/cold-start/runtime constraints.

### Recommended Production Architecture
- Keep this Flask app for UI/API.
- Move persistent data to managed DB (Postgres/Supabase).
- Move uploads/reports to object storage (S3/Cloudinary/Supabase Storage).
- Host model inference on a dedicated ML service (Render/RunPod/Hugging Face/VM) and call it from Flask.

## Notes
- Passwords are securely hashed.
- Upload validation supports `png/jpg/jpeg`.
- Images are resized to `128x128` and normalized before inference.
- Grad-CAM overlay is generated when possible and shown on result page.
- This is an educational project and not a clinical diagnostic replacement.
