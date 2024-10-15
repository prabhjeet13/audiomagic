from flask import Flask, render_template,request,jsonify
from STTmodel import audio_to_text,clean_transcription
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@app.route('/')
def homepage():
    return render_template('index.html')



def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/translate',methods=['POST'])
def tranlateSpeech():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No audio file part in the request"}), 400

        file = request.files['file']
        target_lang = request.form.get('target_lang')
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        
            transcription = audio_to_text(file_path,target_lang)
            cleaned_transcription = clean_transcription(transcription)
            os.remove(file_path)

            return jsonify({
                "original_transcription" : transcription,
                "cleaned_transcription" : cleaned_transcription
            }),200
        else:
            return jsonify({"error" : "unsupported file type"}),400    
    
    except Exception as e:
        return jsonify({"error": f"An unexpected error: {str(e)}"}),500

if __name__ == "__main__":
    app.run(port = 5000,debug=True)