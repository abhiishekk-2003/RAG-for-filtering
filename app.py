#Flast UI
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask import get_flashed_messages

#Import RAG code
from main import run_ingestion_pipeline, run_rag_pipeline, UPLOAD_FOLDER

app = Flask(__name__)
app.secret_key = "mysecretkey"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    #List of uploaded files
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        files = [f for f in os.listdir(UPLOAD_FOLDER) 
                if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('file')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    for file in files:
        if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
    
    #Processing uploaded files
    run_ingestion_pipeline()
    flash('Files uploaded and processed successfully!')
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question', '')
    if not question:
        flash('Please enter a question')
        return redirect(url_for('index'))
    
    answer = run_rag_pipeline(question)
    
    #Storing question and answer
    session['question'] = question
    session['answer'] = answer
    
    return redirect(url_for('index'))

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    flash('All uploaded files have been cleared')
    return redirect(url_for('index'))

if __name__ == '__main__':
    #Running Flask
    app.run(debug=True)                            