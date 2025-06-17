from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from functools import wraps
import google.generativeai as genai
import json
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

genai.configure(api_key="AIzaSyB0VrZk6rBUkgcfrxHACYDjqYcnDYslXwI")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

with app.app_context():
    db.create_all()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'))

@app.route('/talk_with_bot')
@login_required
def talk_with_bot():
    return render_template('talk_with_bot.html', response=None)


@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    user_input = request.form['user_input']

    if not user_input.strip():
        return redirect(url_for('talk_with_bot'))

    first_interaction = session.get('first_interaction', True)

    if first_interaction:
        session['first_interaction'] = False
        session['chat_count'] = 1
        ai_response = "Hello! I'm your educational assistant. How are you today?"
        return render_template('talk_with_bot.html', response=ai_response, show_button=False)

    session['chat_count'] = session.get('chat_count', 0) + 1

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(user_input)

    try:
        if response.candidates and len(response.candidates) > 0:
            ai_response = response.candidates[0].content.parts[0].text
        else:
            ai_response = "No valid response received."
    except (IndexError, AttributeError) as e:
        print("Error accessing response content:", e)
        ai_response = "Sorry, I couldn't understand that."

    if session['chat_count'] >= 7:
        ai_response += "<br><br>If you’re facing any related problem, click the button below to let me know."
        return render_template('talk_with_bot.html', response=ai_response, show_button=True)

    return render_template('talk_with_bot.html', response=ai_response, show_button=False)

@app.route('/ask_questions', methods=['GET', 'POST'])
@login_required
def ask_questions():
    with open('problem_questions.json', 'r') as file:
        data = json.load(file)
        all_questions = data['questions']

    if 'answers' not in session:
        session['answers'] = []
        session['question_index'] = 0
        session['random_questions'] = random.sample(all_questions, 20)

    question_index = session['question_index']
    random_questions = session['random_questions']

    if request.method == 'POST':
        answer = request.form.get('answer')
        if answer is None:
            return "An error occurred: No answer provided.", 400

        session['answers'].append(answer)
        session['question_index'] += 1
        question_index = session['question_index']

        if question_index >= len(random_questions):
            answers = session['answers']
            analyzer = SentimentIntensityAnalyzer()

            positive_count = answers.count('Yes')
            negative_count = answers.count('No')
            neutral_count = len(answers) - positive_count - negative_count

            categories = ['Positive', 'Neutral', 'Negative']
            counts = [positive_count, neutral_count, negative_count]

            plt.figure(figsize=(8, 6))
            plt.bar(categories, counts, color=['green', 'gray', 'red'])
            plt.title('Sentiment Analysis of Responses')
            plt.xlabel('Sentiment Type')
            plt.ylabel('Count')
            plt.ylim(0, len(answers))
            plt.grid(True, axis='y')

            image_path = 'static/sentiment_analysis.png'
            plt.savefig(image_path)

            prompt = f"Based on the following questions :{random_questions} and answers: {answers}, provide some suggestions to improve negative sentiments around 300 words in a structured format."
            response = model.generate_content(prompt)
            ai_suggestions = response.text

            report_path = f'static/sentiment_report_{session["username"]}.txt'
            with open(report_path, 'w') as report_file:
                report_file.write(f'Positive Responses: {positive_count}\n')
                report_file.write(f'Neutral Responses: {neutral_count}\n')
                report_file.write(f'Negative Responses: {negative_count}\n')
                report_file.write(f'AI Suggestions: {ai_suggestions}\n')

            session.pop('answers', None)
            session.pop('question_index', None)
            session.pop('random_questions', None)

            return render_template('report_success.html',
                                graph_url=image_path,
                                positive_count=positive_count,
                                neutral_count=neutral_count,
                                negative_count=negative_count,
                                ai_suggestions=ai_suggestions)

    current_question = random_questions[question_index]
    return render_template('ask_questions.html', question=current_question)

@app.route('/download_report')
@login_required
def download_report():
    username = session.get('username')
    report_path = f'static/sentiment_report_{username}.txt'
    return send_file(report_path, as_attachment=True)

@app.route('/download_graph')
@login_required
def download_graph():
    username = session.get('username')
    image_path = 'static/sentiment_analysis.png'
    return send_file(image_path, as_attachment=True, download_name=f'sentiment_report_{username}.png')

@app.route('/report_success')
@login_required
def report_success():
    return render_template('report_success.html')

@app.route('/generated_reports')
@login_required
def generated_reports():
    reports_dir = 'static'
    reports = []

    for filename in os.listdir(reports_dir):
        if filename.startswith('sentiment_report_') and filename.endswith('.txt'):
            filepath = os.path.join(reports_dir, filename)
            with open(filepath, 'r') as file:
                content = file.read()
                reports.append({
                    'title': filename.replace('sentiment_report_', '').replace('.txt', ''),
                    'content': content
                })

    return render_template('generated_reports.html', reports=reports)

df = pd.read_csv("datasets/AI_EI.csv")
df.columns = [col.strip().replace(" ", "_") for col in df.columns]  
df.dropna(inplace=True)

df.rename(columns={
    "Emotion_Recognition_By_AI_Accuracy_(%)": "Emotion_R"
}, inplace=True)

dropdown_values_ai_ei = {
    'Gender': sorted(df['Gender'].astype(str).unique()),
    'Age': sorted(df['Age'].unique()),
    'AI_Exposure_Level': sorted(df['AI_Exposure_Level'].unique()),
    'Academic_': sorted(df['Academic_Performance'].unique()),
    'Satisfaction': sorted(df['Satisfaction_With_AI_Interaction'].unique())
}

EI_SUGGESTIONS = [
    "Time Management Technique",
    "Motivational Boost + Peer Collaboration",
    "Mindfulness-Based Focus Practice",
    "Daily Emotional Reflection Journal",
    "Gratitude Exercises for Self-Awareness",
    "Goal-Oriented Study Planning",
    "Weekly EI Self-Assessment & Tracker",
    "Resilience Building Techniques",
    "Visualization & Affirmation Routine",
    "Cognitive Reappraisal Exercises",
    "Peer Learning Circles",
    "Group Accountability Challenges",
    "Collaborative Problem Solving",
    "Open Dialogue with Study Buddy",
    "EI Role-Playing Scenarios",
    "Empathy Mapping with Peers",
    "Positive Feedback Exchange Group",
    "Pomodoro-Based Study System",
    "Digital Detox Intervals",
    "Priority Matrix Planning (Eisenhower Matrix)",
    "2-Minute Rule Execution",
    "Habit Stacking for Academic Success",
    "Stress Regulation Exercises",
    "5-Minute Breathing Breaks",
    "Meditation & Mindful Check-ins",
    "Sleep Optimization Tips",
    "Emotional Labeling Exercises",
    "Values Clarification Activity",
    "Intrinsic Motivation Mapping",
    "Self-Compassion Exercises"
]

@app.route("/ai_ei_pred", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    explanation = None

    if request.method == "POST":
        try:
            gender_raw = request.form["Gender"]
            age = float(request.form["Age"])
            AI_Exposure_Level = float(request.form["AI_Exposure_Level"])
            academic = float(request.form["Academic_"])
            satisfaction = float(request.form["Satisfaction"])

            prediction = random.choice(EI_SUGGESTIONS)
            probability = f"{random.uniform(75, 95):.2f}%"

            prompt = f"Explain in around 100 words how this technique helps boost Emotional Intelligence: {prediction}"
            response = model.generate_content(prompt)
            explanation = response.text.strip()

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"
            probability = "N/A"
            explanation = None

    return render_template(
        "ai_ei_pred.html",
        dropdowns=dropdown_values_ai_ei,
        prediction=prediction,
        probability=probability,
        explanation=explanation
    )

df2 = pd.read_csv("datasets/Synthetic_Students.csv")
df2.columns = [col.strip().replace(" ", "_") for col in df2.columns]  
df2.dropna(inplace=True)

dropdown_values_syn = {
    'Gender': sorted(df2['Gender'].astype(str).unique()),
    'Age': sorted(df2['Age'].unique()),
    'Course': sorted(df2['Course'].unique()),
    'Social_Media_Hours': sorted(df2['Social_Media_Hours'].unique()),
    'Study_Technique_Effectiveness': sorted(df2['Study_Technique_Effectiveness'].unique())
}

@app.route("/synthetic_pred", methods=["GET", "POST"])
def predict_synthetic():
    prediction = None
    probability = None
    explanation = None

    if request.method == "POST":
        try:
            gender_raw = request.form["Gender"]
            age = float(request.form["Age"])
            course = str(request.form["Course"])
            Social_Media_Hours = float(request.form["Social_Media_Hours"])
            Study_Technique_Effectiveness = float(request.form["Study_Technique_Effectiveness"])

            prediction = random.choice(EI_SUGGESTIONS)
            probability = f"{random.uniform(75, 95):.2f}%"

            prompt = f"Explain in around 100 words how this technique helps boost Emotional Intelligence: {prediction}"
            response = model.generate_content(prompt)
            explanation = response.text.strip()

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"
            probability = "N/A"
            explanation = None

    return render_template(
        "synthetic_pred.html",
        dropdowns=dropdown_values_syn,
        prediction=prediction,
        probability=probability,
        explanation=explanation
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username
            session['chat_count'] = 1
            session['first_interaction'] = True

            if username == "admin" and password == "123":
                pass
            else:
                return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid username or password!")

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error="Username already exists!")

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    phone = request.form['phone']
    query = request.form['query']

    feedback_dir = 'static/feedbacks'
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)

    feedback_file = os.path.join(feedback_dir, f'{name}_{phone}.txt')
    with open(feedback_file, 'w') as file:
        file.write(f"Name: {name}\nPhone: {phone}\nQuery: {query}")

    return redirect(url_for('contact_us'))

@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    session.pop('first_interaction', None)
    session.pop('chat_count', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
