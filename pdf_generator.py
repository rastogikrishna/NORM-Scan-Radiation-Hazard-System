from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from datetime import datetime


def generate_pdf(
    ra,
    th,
    k,
    u235,
    latitude,
    longitude,
    detected_isotope,
    parent_series,
    decay_chain,
    energy_peak,
    dose,
    raeq,
    hex_value,
    hin_value,
    aed,
    elcr,
    risk_level,
    location_name,
    city,
    state,
    country,
    soil_texture,
    soil_ph,
    texture_interpretation,
    ph_interpretation,
    sample_id=None,
    ml_details=None
):

    filename = "reports/Radiation_Report.pdf"

    # Standard professional margins (54pt = 0.75 in)
    doc = SimpleDocTemplate(
        filename,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom modern typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_CENTER,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        'DocHeading',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    content = []

    # Helper function to generate clean tables
    def create_clean_table(data_rows, col_widths=None):
        t = Table(data_rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor("#cbd5e1")),
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ]))
        return t

    # =====================================================
    # TITLE SECTION
    # =====================================================

    content.append(
        Paragraph(
            "ENVIRONMENTAL RADIATION HAZARD ASSESSMENT REPORT",
            title_style
        )
    )

    content.append(
        Paragraph(
            "Python-Based Machine Learning Model for Radiation Hazard Prediction in Environmental Samples",
            subtitle_style
        )
    )

    content.append(
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            body_style
        )
    )

    content.append(Spacer(1, 10))

    # =====================================================
    # 1. INPUT PARAMETERS TABLE
    # =====================================================

    content.append(Paragraph("1. Input Parameters", heading_style))

    input_table = create_clean_table([
        ["Parameter", "Value"],
        ["Radium-226 (Ra-226)", f"{ra} Bq/kg"],
        ["Thorium-232 (Th-232)", f"{th} Bq/kg"],
        ["Potassium-40 (K-40)", f"{k} Bq/kg"],
        ["Uranium-235 (U-235)", f"{u235} Bq/kg"]
    ], col_widths=[252, 252])

    content.append(input_table)
    content.append(Spacer(1, 12))

    # =====================================================
    # 2. LOCATION INFORMATION
    # =====================================================

    content.append(Paragraph("2. Location Information", heading_style))

    loc_rows = [["Detail", "Value"]]
    if sample_id:
        loc_rows.append(["Sample Number", str(sample_id)])
    else:
        loc_rows.append(["Sample Number", "Manual Input Mode"])
    loc_rows.extend([
        ["Location Name", str(location_name) if location_name else "N/A"],
        ["District", str(city) if city else "N/A"],
        ["Region", str(state) if state else "N/A"],
        ["Country", str(country) if country else "N/A"],
        ["Latitude", f"{latitude:.4f}" if latitude is not None else "N/A"],
        ["Longitude", f"{longitude:.4f}" if longitude is not None else "N/A"]
    ])
    loc_table = create_clean_table(loc_rows, col_widths=[252, 252])

    content.append(loc_table)
    content.append(Spacer(1, 12))

    # =====================================================
    # 3. SOIL CHARACTERISTICS
    # =====================================================

    content.append(Paragraph("3. Soil Characteristics", heading_style))

    soil_table = create_clean_table([
        ["Soil Assessment Parameter", "Value"],
        ["Soil Texture", str(soil_texture)],
        ["Soil pH", f"{soil_ph:.1f}" if soil_ph is not None else "N/A"]
    ], col_widths=[252, 252])

    content.append(soil_table)
    content.append(Spacer(1, 6))

    # Soil properties interpretations
    content.append(Paragraph("<b>Soil Properties Interpretation:</b>", ParagraphStyle('SoilSub', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, spaceBefore=4, spaceAfter=2)))
    content.append(Paragraph(f"• <b>Texture ({soil_texture}):</b> {texture_interpretation}", body_style))
    content.append(Paragraph(f"• <b>pH ({soil_ph if soil_ph is not None else 'N/A'}):</b> {ph_interpretation}", body_style))
    content.append(Spacer(1, 12))

    # =====================================================
    # 4. RESULTS TABLE
    # =====================================================

    content.append(Paragraph("4. Calculated Radiation Parameters", heading_style))

    result_table = create_clean_table([
        ["Parameter", "Calculated Value"],
        ["Absorbed Dose Rate", f"{dose} nGy/h"],
        ["Radium Equivalent Activity (Raeq)", f"{raeq} Bq/kg"],
        ["External Hazard Index (Hex)", str(hex_value)],
        ["Internal Hazard Index (Hin)", str(hin_value)],
        ["Annual Effective Dose (AED)", f"{aed} mSv/y"],
        ["Excess Lifetime Cancer Risk (ELCR)", str(elcr)]
    ], col_widths=[252, 252])

    content.append(result_table)
    content.append(Spacer(1, 12))

    # =====================================================
    # 5. ML PREDICTION (COLOR-CODED CARD)
    # =====================================================

    content.append(Paragraph("5. Machine Learning Prediction", heading_style))

    if risk_level == "Low":
        risk_bg = colors.HexColor("#ecfdf5")
        risk_fg = colors.HexColor("#065f46")
        risk_border = colors.HexColor("#a7f3d0")
    elif risk_level == "Moderate":
        risk_bg = colors.HexColor("#fffbeb")
        risk_fg = colors.HexColor("#92400e")
        risk_border = colors.HexColor("#fde68a")
    else:
        risk_bg = colors.HexColor("#fef2f2")
        risk_fg = colors.HexColor("#991b1b")
        risk_border = colors.HexColor("#fca5a5")

    risk_table = Table([
        ["Predicted Risk Level", risk_level.upper()]
    ], colWidths=[252, 252])

    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), risk_bg),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_fg),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('PADDING', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOX', (0, 0), (-1, -1), 1, risk_border),
    ]))

    content.append(risk_table)
    content.append(Spacer(1, 12))

    # =====================================================
    # 6. ISOTOPE DETECTION
    # =====================================================

    content.append(Paragraph("6. Isotope Detection Results", heading_style))

    isotope_table = create_clean_table([
        ["Detail", "Value"],
        ["Detected Isotope", str(detected_isotope)],
        ["Parent Series", str(parent_series)],
        ["Decay Chain", str(decay_chain)],
        ["Energy Peak", str(energy_peak)]
    ], col_widths=[252, 252])

    content.append(isotope_table)
    content.append(Spacer(1, 12))

    # =====================================================
    # 7. INTERPRETATION
    # =====================================================

    interpretation = (
        f"The environmental sample was analyzed using radionuclide activity concentrations of "
        f"Ra-226 ({ra} Bq/kg), Th-232 ({th} Bq/kg), K-40 ({k} Bq/kg), and U-235 ({u235} Bq/kg). "
        f"The calculated gamma dose rate is {dose} nGy/h, giving a Radium Equivalent "
        f"Activity (Raeq) of {raeq} Bq/kg. The External Hazard Index (Hex) is {hex_value} "
        f"and the Internal Hazard Index (Hin) is {hin_value}. The predicted Annual Effective "
        f"Dose (AED) is {aed} mSv/y with an Excess Lifetime Cancer Risk (ELCR) of {elcr}. "
        f"Based on these physical parameters and machine learning prediction, the "
        f"sample represents a {risk_level.upper()} radiation hazard risk."
    )

    content.append(Paragraph("7. Radiation Hazard Interpretation", heading_style))
    content.append(Paragraph(interpretation, body_style))
    content.append(Spacer(1, 4))

    # =====================================================
    # 8. RECOMMENDATIONS
    # =====================================================

    if risk_level == "Low":
        recommendation = (
            "Radionuclide activity concentrations and environmental hazard indices are "
            "within internationally accepted safety limits. No immediate remediation or "
            "containment is required. Routine periodic environmental monitoring is recommended "
            "to track long-term baseline stability."
        )
    elif risk_level == "Moderate":
        recommendation = (
            "The radiological hazard assessment indicates moderate levels of radiation. "
            "It is recommended to conduct a localized follow-up radiological survey and perform "
            "periodic monitoring of water, soil, or building materials in the sample vicinity "
            "to ensure levels do not escalate."
        )
    else:
        recommendation = (
            "Radiological hazard indices significantly exceed safety limits. Immediate remedial "
            "action and localized decontamination investigation are strongly advised. Access "
            "to the site should be managed, and continuous, automated radiation monitoring "
            "should be deployed to mitigate human and environmental exposure risks."
        )

    content.append(Paragraph("8. Recommendations", heading_style))
    content.append(Paragraph(recommendation, body_style))
    content.append(Spacer(1, 4))

    # =====================================================
    # 9. CONCLUSION
    # =====================================================

    if detected_isotope and detected_isotope not in ["Not Detected", "N/A", "Unknown"]:
        detected_isotopes_str = f" Spectroscopic peak matching identified the presence of: {detected_isotope} ({decay_chain})."
    else:
        detected_isotopes_str = " Spectroscopic analysis did not identify any matching isotopes."

    conclusion = (
        f"Based on numerical hazard indexing and machine learning analysis, this environmental "
        f"sample is classified under the {risk_level.upper()} RISK category.{detected_isotopes_str}"
    )

    content.append(Paragraph("9. Conclusion", heading_style))
    content.append(Paragraph(conclusion, body_style))
    content.append(Spacer(1, 12))

    # =====================================================
    # 10. MACHINE LEARNING ANALYSIS
    # =====================================================

    if ml_details:
        content.append(Paragraph("10. Machine Learning Anomaly Detection", heading_style))
        
        # 10.1 ML Summary Table
        ml_summary_rows = [
            ["Metric/Specification", "Value"],
            ["ML Algorithm", ml_details.get("model_name", "N/A")],
            ["Anomaly Screening Status", ml_details.get("anomaly_status", "N/A")],
            ["Anomaly Score", ml_details.get("anomaly_score", "N/A")],
            ["Dose-Rate Independent Validation", ml_details.get("validation_status", "N/A")]
        ]
        ml_summary_table = create_clean_table(ml_summary_rows, col_widths=[180, 324])
        content.append(ml_summary_table)
        content.append(Spacer(1, 10))

        # Show Anomaly Message in a prominent text section
        content.append(Paragraph(f"<b>Anomaly screening analysis message:</b> {ml_details.get('anomaly_message', 'N/A')}", ParagraphStyle('MsgStyle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, leading=13, spaceAfter=10)))
        content.append(Spacer(1, 4))
        
        # 10.2 Feature Importance Table
        feat_rows = [["Radionuclide Feature", "Importance Gini Index"]]
        for feat, val in ml_details.get("feature_importances", []):
            feat_rows.append([feat, f"{val*100:.2f}%"])
        feat_table = create_clean_table(feat_rows, col_widths=[252, 252])
        
        # 10.3 ML Performance Table
        perf_metrics = ml_details.get("performance_metrics", {})
        perf_rows = [
            ["Model Parameter", "Value"],
            ["Training Set Size", perf_metrics.get("sample_count", "N/A")],
            ["Contamination Factor", perf_metrics.get("contamination", "N/A")],
            ["Number of Trees (Estimators)", perf_metrics.get("n_estimators", "N/A")]
        ]
        perf_table = create_clean_table(perf_rows, col_widths=[252, 252])
        
        # 10.4 Screening Criteria Table
        prob_rows = [
            ["Risk Level Classification", "Radiological Screening Criteria"],
            ["Low Risk", "Radium Equivalent (Raeq) < 100 Bq/kg"],
            ["Moderate Risk", "100 <= Radium Equivalent (Raeq) <= 370 Bq/kg"],
            ["High Risk", "Radium Equivalent (Raeq) > 370 Bq/kg"]
        ]
        prob_table = create_clean_table(prob_rows, col_widths=[200, 304])
        
        content.append(Paragraph("<b>Radionuclide Training Importance:</b>", ParagraphStyle('SubH1', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, spaceBefore=4, spaceAfter=4)))
        content.append(feat_table)
        content.append(Spacer(1, 10))
        
        content.append(Paragraph("<b>Model Hyperparameters & Training Specifications:</b>", ParagraphStyle('SubH2', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, spaceBefore=4, spaceAfter=4)))
        content.append(perf_table)
        content.append(Spacer(1, 4))
        content.append(Paragraph("<i>Note: The unsupervised Isolation Forest model is trained exclusively on the background development dataset.</i>", ParagraphStyle('CVNote', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=8, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=10)))
        
        content.append(Paragraph("<b>Radiological Screening Thresholds:</b>", ParagraphStyle('SubH3', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, spaceBefore=4, spaceAfter=4)))
        content.append(prob_table)
        content.append(Spacer(1, 10))
        
        content.append(Paragraph("<b>Anomaly Screening Explanation:</b>", ParagraphStyle('SubH4', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, leading=13, spaceBefore=4, spaceAfter=4)))
        explanation_bullets = ml_details.get("explanation", [])
        for bullet in explanation_bullets:
            content.append(Paragraph(f"• {bullet}", body_style))

    doc.build(content)

    return filename