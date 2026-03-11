import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DISEASE_DETAILS = {
    "Normal": {
        "type": "No malignant pattern detected",
        "overview": "No clear imaging features of lung carcinoma were identified by the model.",
        "action": "Continue routine follow-up if clinically indicated.",
    },
    "Adenocarcinoma": {
        "type": "Non-small cell lung cancer (NSCLC)",
        "overview": "Adenocarcinoma is the most common NSCLC subtype and may arise in peripheral lung tissue.",
        "action": "Recommend oncologist review, staging workup, and histopathological confirmation.",
    },
    "Squamous Cell Carcinoma": {
        "type": "Non-small cell lung cancer (NSCLC)",
        "overview": "Squamous cell carcinoma is often centrally located and linked with smoking history.",
        "action": "Recommend urgent multidisciplinary evaluation and confirmatory diagnostics.",
    },
    "Large Cell Carcinoma": {
        "type": "Non-small cell lung cancer (NSCLC)",
        "overview": "Large cell carcinoma is a poorly differentiated NSCLC subtype and can be aggressive.",
        "action": "Recommend prompt specialist review and complete staging panel.",
    },
}


def build_interpretation_text(predicted_label: str, confidence: float) -> str:
    confidence_pct = confidence * 100
    details = DISEASE_DETAILS.get(predicted_label, {})

    if predicted_label == "Normal":
        return (
            f"AI analysis suggests no malignant pattern (confidence: {confidence_pct:.2f}%). "
            f"{details.get('action', 'Clinical correlation is advised.')}"
        )

    return (
        f"AI analysis indicates features most consistent with {predicted_label} "
        f"(confidence: {confidence_pct:.2f}%). {details.get('action', 'Clinical confirmation is required.')}"
    )


def _probability_rows(probabilities: dict):
    rows = []
    for label, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        rows.append([f"{label} Probability", f"{prob * 100:.2f}%"])
    return rows


def generate_medical_report(output_path: str, patient: dict, image_path: str, prediction: dict) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    story = []
    title = Paragraph("Lung Cancer Detection Medical Report", styles["Title"])
    subtitle = Paragraph("Project Hospital AI Diagnostic Unit", styles["Heading3"])
    story.extend([title, subtitle, Spacer(1, 12)])

    patient_rows = [
        ["Patient ID", patient.get("patient_uid", "-")],
        ["Patient Name", patient.get("full_name", "-")],
        ["Age", str(patient.get("age") if patient.get("age") is not None else "-")],
        ["Gender", patient.get("gender") or "-"],
        ["Phone", patient.get("phone") or "-"],
        ["Prediction Date", prediction.get("prediction_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
    ]

    patient_table = Table(patient_rows, colWidths=[2.0 * inch, 4.8 * inch])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([patient_table, Spacer(1, 14)])

    story.append(Paragraph("CT Scan Preview", styles["Heading3"]))
    if os.path.exists(image_path):
        scan_image = Image(image_path, width=2.3 * inch, height=2.3 * inch)
        story.extend([scan_image, Spacer(1, 10)])

    probabilities = prediction.get("probabilities", {})
    confidence_pct = prediction.get("confidence", 0.0) * 100
    pred_rows = [
        ["Predicted Class", prediction.get("label", "-")],
        ["Confidence Score", f"{confidence_pct:.2f}%"],
    ] + _probability_rows(probabilities)

    prediction_table = Table(pred_rows, colWidths=[2.8 * inch, 4.0 * inch])
    prediction_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d6e4ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )

    predicted_label = prediction.get("label", "")
    detail = DISEASE_DETAILS.get(predicted_label, {})

    disease_rows = [
        ["Disease Type", detail.get("type", "Unknown")],
        ["Clinical Overview", detail.get("overview", "No additional details available.")],
        ["Suggested Next Step", detail.get("action", "Clinical validation recommended.")],
    ]

    disease_table = Table(disease_rows, colWidths=[2.0 * inch, 4.8 * inch])
    disease_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f7f0ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.extend([Paragraph("Prediction Summary", styles["Heading3"]), prediction_table, Spacer(1, 10)])
    story.extend([Paragraph("Disease Detail", styles["Heading3"]), disease_table, Spacer(1, 10)])

    story.append(Paragraph(f"AI Interpretation: {prediction.get('interpretation', '-')}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    disclaimer = (
        "Disclaimer: This report is generated by an AI model for decision support only. "
        "It must not replace professional medical diagnosis, biopsy, or treatment planning."
    )
    story.append(Paragraph(disclaimer, styles["Italic"]))

    doc.build(story)
