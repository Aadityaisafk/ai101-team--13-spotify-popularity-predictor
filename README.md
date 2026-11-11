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
│   ├── datapreprocessing.ipynb
│   ├── eda and graphs(1).ipynb
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

### 1. Clone the repository
```bash
git clone https://github.com/Aadityaisafk/ai101-team--13-spotify-popularity-predictor
cd ai101-team--13-spotify-popularity-predictor
```

### 2. Create and activate a virtual environment

- **Windows (PowerShell):**
  ```powershell
  # If script execution is restricted on Windows, run this once:
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

  python -m venv venv
  venv\Scripts\activate
  ```

- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the end-to-end pipeline
```bash
python src/main.py
```

Python version: 3.10+

Run the notebooks in this order: `data_preprocessing` → `eda_and_graphs` → `model_training_linear_regression` → `evaluation`.

## Dataset

- Source: [Spotify 114k Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- Size before cleaning: 114,000 rows × 22 columns, 114 genres
- Cleaning changes: Dropped unneeded index columns (`Unnamed: 0`), removed missing values, dropped duplicate track records by `track_id`, and converted `explicit` to binary integer (0/1).
## Expected Outputs

| Metric | Value |
|---|---|
| RMSE | 22.0445 |
| MAE | 18.3904 |
| R² | 0.0226 |

Files generated in `outputs/`:
- `actual_vs_predicted.png`: Scatter plot comparing true test popularity against predictions with ideal diagonal line.
- `residuals_plot.png`: Residual error distribution plot.
- `correlation_matrix.png`: Pearson correlation heatmap of audio features and popularity.
- `metrics_summary.csv`: Summary table containing the evaluation scores.

## Assets

- 5-slide deck (PDF): [docs/slides.pdf](docs/slides.pdf)
- 5-minute walkthrough video: <link>

## AI Disclosure

- Models used: ChatGPT,claude and Gemini.
- Used for: Assisting with matplotlib diagnostic plot formatting, debugging import paths, and drafting initial documentation structure.
- Verification: All mathematical logic, preprocessing steps, feature scaling, and model evaluations were tested and verified locally by the team.

