NORM-Scan — Environmental Radiation Hazard Screening System
Overview

NORM-Scan is a Python-based environmental radiation screening system that combines scientific radiation calculations with Machine Learning to analyze naturally occurring radioactive materials (NORM) in environmental samples.

The system accepts radionuclide and environmental inputs and provides radiation hazard parameters, ML-based anomaly detection, isotope identification, visualization, mapping, and automated PDF reports.

Key Features
Radiation Dose Rate calculation
Radium Equivalent Activity (Raeq)
External Hazard Index (Hex)
Internal Hazard Index (Hin)
Annual Effective Dose (AED)
Excess Lifetime Cancer Risk (ELCR)
Machine Learning-based anomaly detection
Random Forest / ML analysis
Feature analysis
Gamma-ray isotope identification
Interactive dashboard
Interactive geographical mapping
Soil and sample analysis
Automated PDF report generation
Sample database and data management
Machine Learning

The Machine Learning component is used for anomaly detection rather than replacing the scientific radiation calculations.

The model learns patterns in the environmental sample data and identifies samples that are statistically unusual.

The output is:

Normal — sample is within the learned data pattern.
Anomalous — Verify Measurement — sample is unusual and should be checked further.

An anomaly does not automatically mean that a sample is contaminated or unsafe.

Radiation Assessment

The scientific radiation calculations remain separate from the Machine Learning component.

The system calculates:

Dose Rate → Raeq → Hex → Hin → AED → ELCR

These parameters are then used for radiological screening and risk assessment.

Input Parameters

The system can work with parameters including:

Ra-226 activity
Th-232 activity
K-40 activity
U-235 activity
Soil pH
Soil texture
Material/environmental information
Latitude and longitude
Gamma-ray energy information
Technology Stack
Python
Pandas
NumPy
Scikit-learn
Plotly
Dash
Dash Bootstrap Components
Dash Leaflet
ReportLab
OpenStreetMap
JSON / CSV
SQLite
Project Structure
RadiationHazardSystem/
│
├── app.py
├── formula_engine.py
├── isotope_lookup.py
├── pdf_generator.py
├── train_model.py
├── requirements.txt
│
├── assets/
├── dataset/
├── model/
├── reports/
└── db/
Data

The project uses authorized environmental sample data together with clearly identified development data where applicable.

Synthetic data, when used, are explicitly labelled and are not presented as real laboratory measurements.

The original authorized dataset is retained separately from Machine Learning development data.

Limitations

This system is intended as a screening and research prototype.

Machine Learning anomaly detection identifies statistical patterns and does not by itself establish contamination, health risk, or regulatory non-compliance.

Final environmental radiation assessment should be supported by certified laboratory measurements, gamma-ray spectrometry, and applicable regulatory standards.

Purpose

The project demonstrates how environmental radiation science, data analytics, visualization, and Machine Learning can be integrated into a single software platform for faster and more informative radiation screening.
