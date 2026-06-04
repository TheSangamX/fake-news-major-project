# NewsIntel 🛡️
### NLP-Based News Authenticity Verification Platform

NewsIntel is a production-ready, full-stack web application designed to analyze news articles and detect potential misinformation using Natural Language Processing (NLP) and Machine Learning. The platform features secure Google Authentication, a dynamic verification history synced to a cloud database, and a highly polished glassmorphic interface.

---

## 🚀 Key Features

*   **ML-Powered Verification:** Evaluates news article text credibility using a trained **Passive Aggressive Classifier** model.
*   **Tactile Verification Card:** Displays verification results with dynamic status banners, SVG icons, and a custom description response.
*   **Interactive History Logs:** Logs verification results to a **Supabase** PostgreSQL database and renders the last three checks in the sidebar. History entries open detail modals with copy-to-clipboard functionality.
*   **Validation Safeguards:** Requires inputs to be at least **50 characters** long to avoid meaningless predictions. Includes dynamic input highlights and live warning clearances as the user types.
*   **Immersive UI Transitions:** Features realistic 2-second processing loaders, dynamic page background color transitions (green/red depending on status), and custom-generated favicons.
*   **Responsive enhanced Footer:** Premium center-aligned footer containing professional portfolio links, animated CSS-only tooltips, and micro-hover translations.

---

## 🛠️ Tech Stack & Architecture

-   **Backend:** Python 3.x, Flask (development server)
-   **Machine Learning & NLP:** Scikit-Learn, NLTK, Passive Aggressive Classifier, TF-IDF Vectorization
-   **Database & Auth:** Supabase (PostgreSQL database + Supabase Google OAuth)
-   **Frontend:** HTML5, CSS3 (translucent glassmorphism, linear gradients), Bootstrap 5.0.2, JQuery

---

## 🧠 Machine Learning Model & Pipeline

### 1. NLP Preprocessing
Before classification, news articles are processed using the following pipeline:
1.  **Text Cleaning:** Non-alphabetic characters are filtered out using regular expressions.
2.  **Case Normalization:** All text is converted to lowercase to maintain consistency.
3.  **Tokenization:** Sentences are split into individual word tokens.
4.  **Stopwords Removal:** Standard English stopwords (e.g., "the", "is", "at") are removed using the NLTK corpus.
5.  **Stemming:** Words are reduced to their root forms using the **Porter Stemmer** algorithm.

### 2. Feature Extraction
A **TF-IDF Vectorizer** (`TfidfVectorizer`) transforms preprocessed tokens into numerical sparse matrices based on Term Frequency-Inverse Document Frequency weightings.

### 3. Classification
The core classifier is a **Passive Aggressive Classifier** (`PassiveAggressiveClassifier`), which is highly suited for large-scale text classification tasks:
-   **Passive Step:** If the prediction matches the true label within margins, the model coefficients remain unchanged.
-   **Aggressive Step:** If the prediction is incorrect, the classifier aggressively updates its weights to correct the boundary.

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure you have Python 3.8+ installed.

### 1. Clone and Navigate
```bash
git clone https://github.com/TheSangamX/NewsIntel.git
cd NewsIntel
```

### 2. Create and Activate Virtual Environment
```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python app.py
```
Open **`http://localhost:3000/`** in your browser to view the application.

---

## 📂 Project Structure
```text
NewsIntel/
├── app.py                  # Core Flask server & API endpoints
├── Training_Code.py        # Model training script
├── model2.pkl              # Serialized Passive Aggressive Classifier
├── tfidfvect2.pkl          # Serialized TF-IDF Vectorizer
├── requirements.txt        # Python dependency list
├── .gitignore              # Files ignored by git
├── images/                 # Original source assets
│   └── sheild.png          # Custom logo shield
├── static/                 # Static assets folder
│   ├── fake-news.png       # Original favicon
│   └── sheild.png          # App checkmark shield icon
└── templates/              # HTML layout templates
    ├── index.html          # Verification dashboard template
    └── login.html          # Glassmorphic auth page template
```

---

## 🧑‍💻 Developed By
**Sangam Gupta**  
*   **Portfolio:** [sangamgupta.in](https://www.sangamgupta.in/)
*   **GitHub:** [@TheSangamX](https://github.com/TheSangamX)
*   **LinkedIn:** [in/thesangamx](https://linkedin.com/in/thesangamx)

---
*Built with Machine Learning, NLP and Flask.*
