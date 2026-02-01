<div align="center">

# 🎵 DataZen Case Study

### ⚡ Music Intelligence & Analytics Dashboard

*A Risk Mitigation Tool for the Modern Record Label*

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

**Dark Mode** • **Cyberpunk Aesthetics** • **Glassmorphism UI** • **Neon Accents**

</div>

---

## 🎯 The Elevator Pitch

> **"Replace gut feelings with data-driven decisions."**

In the high-stakes world of music production, A&R executives have traditionally relied on intuition to greenlight projects worth millions. **DataZen** changes the game by transforming raw audio analytics into actionable intelligence.

This dashboard serves as a **Risk Mitigation Engine** for record labels operating in the dynamic Indian music market. By leveraging machine learning on thousands of tracks, we quantify what makes a hit—enabling executives to make informed investment decisions, identify emerging market opportunities, and strategically position their catalog for maximum impact.

---

## ✨ Key Features: The 4-Tab Architecture

### 📊 Tab 1: Executive Overview
*Market Surveillance at a Glance*

- **Geospatial Heatmaps** — Interactive Folium-powered maps visualizing regional music consumption patterns across India
- **Temporal Trend Analysis** — Year-over-year performance metrics and seasonal patterns
- **KPI Dashboard** — Real-time market share, streaming velocity, and engagement metrics

---

### 🧬 Tab 2: Hit DNA
*Decoding the Anatomy of a Successful Track*

- **Audio Fingerprinting** — Radar charts dissecting tracks across 8+ audio features (energy, danceability, valence, acousticness, etc.)
- **Feature Correlation Matrix** — Discover hidden relationships between audio attributes and popularity scores
- **🔮 AI Hit Simulator** — A Random Forest Regressor model that predicts hit potential with a stunning **Glowing Gauge** visualization
  - Adjust sliders for tempo, energy, danceability, and more
  - Get instant HIT/FLOP predictions with confidence scores
  - Neon-animated gauge responds in real-time

---

### 🤖 Tab 3: Athena AI
*Your Deterministic Intelligence Assistant*

A **rule-based NLP chatbot** designed for precise, hallucination-free responses:

- *"Top songs of 2015"* → Instant ranked list
- *"Compare Hindi vs Punjabi energy"* → Side-by-side audio feature analysis
- *"Which genre has the highest danceability?"* → Data-backed answers

> ⚠️ **No LLM hallucinations** — Athena operates on structured queries against the dataset, ensuring 100% factual responses.

---

### 📈 Tab 4: Strategy Deck
*The BCG Matrix for Music*

Strategic portfolio analysis identifying:

| Quadrant | Description | Example |
|----------|-------------|---------|
| 🌟 **Stars** | High growth, high market share | Punjabi Pop |
| 💰 **Cash Cows** | Low growth, high market share | Telugu Film Music |
| 💎 **Hidden Gems** | High growth, low market share | Odia Regional |
| ❓ **Question Marks** | Emerging genres requiring investment decisions | Indie Electronic |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit with Custom CSS (Dark/Glassmorphism UI) |
| **Data Visualization** | Plotly Interactive Charts, Folium Geospatial Maps |
| **ML Engine** | Scikit-Learn (Random Forest Regressor) |
| **NLP** | TextBlob for sentiment analysis |
| **Data Processing** | Pandas, NumPy |
| **Network Analysis** | NetworkX |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/Datathon-CaseStudy-2026.git
cd Datathon-CaseStudy-2026

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Datathon-CaseStudy-2026/
│
├── main.py                        # Core Streamlit application
├── data_cleaner.py                # Data preprocessing utilities
├── requirements.txt               # Python dependencies
├── spotify_cleaned_master.csv     # Cleaned dataset (~6.5MB)
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🎨 Design Philosophy

This dashboard embraces a **Cyberpunk-inspired Dark Mode** aesthetic:

- **Glassmorphism** — Frosted glass effects on containers
- **Neon Accents** — Cyan, magenta, and electric blue highlights
- **Dark Canvas** — Reduced eye strain for extended analysis sessions
- **Subtle Animations** — Glowing effects that respond to user interaction

---

## 📜 License

This project was developed for the **2026 Datathon Case Study Competition**.

---

<div align="center">

**Built with 💜 for data-driven music intelligence**

*Transforming the art of A&R into a science*

</div>
