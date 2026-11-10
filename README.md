# Hit Song Popularity Prediction (Spotify Audio Features Regression)

Predicts a track's popularity score (0–100) from acoustic signals (danceability, energy, loudness, valence, tempo) and genre, using Linear Regression on the Spotify Tracks dataset. This is the AI101 Mini-Project of Group 13.

**Research question:** Can we predict a song's popularity using its audio features and genre?

## Team & Roles (Group 13)

| Member | Name | Entry No. | GitHub | Deliverable lead | Coding task |
|---|---|---|---|---|---|
| 1 | Kayalvizhi S U | 2026EEB1364 | [@Vizhi-8](https://github.com/Vizhi-8) | PPT | EDA & graphs |
| 2 | Aaditya Chandel | 2026MCB1502 | [@Aadityaisafk](https://github.com/Aadityaisafk) | Execution | Evaluation & integration |
| 3 | Mannat | 2026AIB1056 | [@mannat-s12](https://github.com/mannat-s12) | Video | Linear Regression model |
| 4 | Hemanshu Samadhan Borse | 2026ESB1477 | [@xlent-h](https://github.com/xlent-h) | Documentation | Data cleaning & preprocessing |

## Repository Structure

```
ai101-team--13-spotify-popularity-predictor/
├── data/
│   └── dataset.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── eda_and_graphs.ipynb
│   ├── model_training_linear_regression.ipynb
│   └── evaluation.ipynb
├── outputs/            # saved plots and metrics
├── docs/
│   └── slides.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Quickstart

```bash
git clone https://github.com/Aadityaisafk/ai101-team--13-spotify-popularity-predictor
cd ai101-team--13-spotify-popularity-predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Python version: <x.y>

Run the notebooks in this order: `data_preprocessing` → `eda_and_graphs` → `model_training_linear_regression` → `evaluation`.

## Dataset

- Source: <website, author, download link>
- Size before cleaning: 114,000 rows × 22 columns, 114 genres
- Cleaning changes: <missing values handled, duplicate track_id records, columns removed>

## Expected Outputs

| Metric | Value |
|---|---|
| RMSE | <fill after running> |
| MAE | <fill after running> |
| R² | <fill after running> |

Plots saved in `outputs/`: <list file names>

## Assets

- 5-slide deck (PDF): <link or docs/slides.pdf>
- 5-minute walkthrough video: <link>

## AI Disclosure

Models used: <e.g. Claude>. Used for: <e.g. planning, debugging, README drafting>.
Verification: <how the team ran the code and checked the results>.
