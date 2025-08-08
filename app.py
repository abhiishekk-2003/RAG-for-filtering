# Flask UI
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from flask import get_flashed_messages

# Import RAG code
from main import run_ingestion_pipeline, run_rag_pipeline, UPLOAD_FOLDER

app = Flask(__name__)
app.secret_key = "mysecretkey"

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # List of uploaded files
    files = []
    if os.path.exists(UPLOAD_FOLDER):
        files = [f for f in os.listdir(UPLOAD_FOLDER) 
                if f.lower().endswith(('.pdf', '.docx', '.txt', '.json'))]  # Added .json support
    
    # Get any stored question/answer from session
    question = session.get('question', '')
    answer = session.get('answer', '')
    
    return render_template('index.html', files=files, question=question, answer=answer)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    files = request.files.getlist('file')
    
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    uploaded_files = []
    for file in files:
        if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt', '.json')):  # Added .json support
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            uploaded_files.append(filename)
        else:
            flash(f'Unsupported file type: {file.filename}. Supported types: PDF, DOCX, TXT, JSON')
    
    if uploaded_files:
        try:
            # Processing uploaded files
            run_ingestion_pipeline()
            flash(f'Files uploaded and processed successfully: {", ".join(uploaded_files)}')
        except Exception as e:
            flash(f'Error processing files: {str(e)}')
            print(f"Ingestion error: {e}")
    
    return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question', '').strip()
    if not question:
        flash('Please enter a question')
        return redirect(url_for('index'))
    
    try:
        answer = run_rag_pipeline(question)
        
        # Store question and answer in session
        session['question'] = question
        session['answer'] = answer
        
        flash('Question answered successfully!')
        
    except Exception as e:
        flash(f'Error processing question: {str(e)}')
        print(f"RAG pipeline error: {e}")
        session['question'] = question
        session['answer'] = 'Sorry, I encountered an error while processing your question.'
    
    return redirect(url_for('index'))

@app.route('/clear_uploads', methods=['POST'])
def clear_uploads():
    """Clear all uploaded files from the upload folder"""
    if os.path.exists(UPLOAD_FOLDER):
        deleted_files = []
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    deleted_files.append(file)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                flash(f'Error deleting {file}: {str(e)}')
        
        if deleted_files:
            flash(f'Deleted files: {", ".join(deleted_files)}')
        else:
            flash('No files to delete')
    else:
        flash('Upload folder does not exist')
    
    # Clear session data
    session.pop('question', None)
    session.pop('answer', None)
    
    return redirect(url_for('index'))

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear the current question and answer from session"""
    session.pop('question', None)
    session.pop('answer', None)
    flash('Question and answer cleared')
    return redirect(url_for('index'))

@app.route('/delete_file/<filename>', methods=['POST'])
def delete_file(filename):
    """Delete a specific file"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if os.path.exists(file_path):
            os.unlink(file_path)
            flash(f'File {filename} deleted successfully')
        else:
            flash(f'File {filename} not found')
    except Exception as e:
        flash(f'Error deleting {filename}: {str(e)}')
        print(f"Error deleting file {filename}: {e}")
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Running Flask
    app.run(debug=True, host='0.0.0.0', port=5000)