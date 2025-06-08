
# 📚 Educational Assistant Chatbot with EI Prediction

This is a Flask-based web application that integrates an educational assistant chatbot (powered by Google Gemini AI), sentiment analysis using VADER, and Emotional Intelligence (EI) score prediction using trained ML models.

---

## 🚀 Features

- Gemini AI chatbot to assist users with educational queries.
- Sentiment analysis using VADER Sentiment Analyzer.
- Emotional Intelligence (EI) prediction using:
  - Real data model: `ei_stack_model.pkl`
  - Synthetic data model: `ei_synthetic_stack_model.pkl`
- User interface using HTML, CSS, and Jinja templates.
- Graphical output with `matplotlib`.

---

## 🧰 Tech Stack

- **Backend**: Flask
- **Frontend**: HTML, CSS (Jinja Templates)
- **ML Libraries**: Scikit-learn, LightGBM, XGBoost
- **Others**: Google Generative AI, VADER Sentiment, Pandas, Numpy, Matplotlib

---

## 🗂 Project Structure

```
project/
│
├── app.py                       # Main Flask application
├── ei_stack_model.pkl          # Trained EI model on real data
├── ei_synthetic_stack_model.pkl# Trained EI model on synthetic data
├── templates/
│   ├── index.html              # Home/chat UI
│   ├── ei_form.html            # Form to input data for EI prediction
│   └── result.html             # Displays prediction results
├── static/
│   ├── css/
│   │   └── style.css           # Styling
│   └── images/                 # Any images used
├── requirements.txt            # Dependencies
└── README.md                   # Project instructions
```

---

## 💻 How to Run Locally

### 🔧 Prerequisites

- Python 3.8+
- Git (optional but useful)
- [Google API Key for Gemini AI](https://ai.google.dev/)

### 🛠 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/educational-chatbot.git
cd educational-chatbot
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install Requirements**

```bash
pip install -r requirements.txt
```

4. **Set Environment Variables**

```bash
export GOOGLE_API_KEY='your_gemini_api_key_here'
```

Or create a `.env` file:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
```

5. **Run the App**

```bash
python app.py
```

6. **Open in Browser**

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📈 Example Use Cases

- Ask study-related questions to the chatbot.
- Analyze user input sentiment (positive/negative/neutral).
- Predict user's Emotional Intelligence category based on form inputs.

---

## 🧪 Testing

You can test:
- Chatbot: via homepage.
- Sentiment analysis: by entering a sentence.
- EI prediction: via form (try both models).

---

## ✅ To-Do

- Add user authentication
- Improve chatbot contextual memory
- Add database logging of predictions and chats

---

## 📜 License

MIT License
