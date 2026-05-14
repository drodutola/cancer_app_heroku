# Cancer Drug Prediction App (Heroku Deployment) 🚀

A deployed web application serving the cancer drug prediction model as a live, accessible tool.

## Live App

Hosted on Heroku — provides a web interface to the underlying ML prediction engine from [`cancerdrug_predictor`](https://github.com/drodutola/cancerdrug_predictor).

## What It Does

Users can input tumor/patient features and receive a drug prediction output in real time. Designed to demonstrate how ML models can be packaged and served as practical clinical decision-support tools.

## Tech Stack

- **Backend:** Python / Flask
- **ML Model:** scikit-learn
- **Deployment:** Heroku
- **Data Processing:** Pandas, NumPy

## Local Setup

```bash
git clone https://github.com/drodutola/cancer_app_heroku
cd cancer_app_heroku
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in your browser.

## Deployment

```bash
heroku create
git push heroku main
```

## Author

**Dr. Peter Odutola, M.D.** — Physician, AI developer, and clinical researcher.  
[GitHub Profile](https://github.com/drodutola)
