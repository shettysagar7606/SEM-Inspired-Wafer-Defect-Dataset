# SEM-Inspired Wafer Defect Analysis (Phase-1)

## Overview
In semiconductor manufacturing, early detection of wafer-level defects is critical for improving yield and process reliability.  
However, access to real SEM (Scanning Electron Microscope) inspection data is often limited due to proprietary and cost constraints.

This project focuses on building a **Phase-1 prototype pipeline** for wafer defect analysis using **SEM-inspired synthetic images** and a **baseline deep learning model structure**. The goal is to validate the overall approach before moving to real-world data and deployment.

---

## Problem Statement
Developing AI-based inspection systems in the early stages faces several challenges:
- Limited availability of labeled SEM defect datasets
- Difficulty in rapid experimentation without accessible data
- Need for a scalable and reproducible inspection pipeline

The objective of this work is to demonstrate a **structured approach** for defect analysis using synthetic data generation and a planned ML model workflow.

---

## Approach
The current Phase-1 approach includes:
- Generation of **SEM-inspired grayscale wafer images** using Python-based image processing
- Simulation of common defect patterns such as:
  - Line Edge Roughness (LER)
  - Open defects
  - Bridging defects
  - Cracks and surface anomalies
- Inclusion of a **baseline CNN model structure** to outline the planned training and inference pipeline

> The synthetic images are intended for **prototyping, validation, and augmentation purposes only**, and are not a replacement for real SEM inspection data.

---

## Repository Structure
