# CyberGuard-IndiaAI-Hackathon

# Cybercrime Text Classification

This repository contains code for a hierarchical classification model designed to categorize and subcategorize cybercrime-related textual data. The model classifies input text into predefined categories and subcategories of cybercrime, providing a granular view of different cybercrime types.

## Project Overview

The project implements a multi-label, hierarchical classification model, using NLP techniques to classify cybercrime descriptions into categories such as **Financial Fraud**, **Ransomware**, **Hacking**, and **Cyber Terrorism**. Additionally, subcategories within these main categories (e.g., **Phishing**, **DDoS Attacks**, **Cryptocurrency Fraud**) are included to ensure more detailed classification.

### Key Features
- **Hierarchical Multi-Label Classification**: Categorizes text at both main category and subcategory levels.
- **BERT-based Model**: Uses the BERT architecture to handle text embeddings and improve classification accuracy.
- **Evaluation Metrics**: Provides precision, recall, and F1-score for both category and subcategory classifications.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
3. Usage
   ```bash
   python classification_model.py --input-file path/to/input.csv --output-file path/to/output.csv
