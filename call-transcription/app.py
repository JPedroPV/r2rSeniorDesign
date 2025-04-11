from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import whisperx, gc, os, ollama
chatting = False

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

TOKEN = "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"

model = whisperx.load_model("base", "cuda", compute_type="float16") 
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400

    if file and (file.filename.endswith('.mp3') or file.filename.endswith('.wav') or file.filename.endswith('.m4a')):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            audio = whisperx.load_audio(filepath)
            result = model.transcribe(audio, batch_size=16)

            model_align, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
            result = whisperx.align(result["segments"], model_align, metadata, audio, "cuda", return_char_alignments=False)

            model_diarize = whisperx.DiarizationPipeline(use_auth_token=TOKEN, device="cuda")
            diarize_segments = model_diarize(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            transcript_content = ""
            for line in result["segments"]:
                try:
                    transcript_content += f"[{line['start']} -> {line['end']}] [{line['speaker']}]\t{line['text']}\n"
                except:
                    line['speaker'] = "UNKNOWN"
                    transcript_content += f"[{line['start']} -> {line['end']}] [{line['speaker']}]\t{line['text']}\n"
            
            if chatting:
                # Prompt Ollama to replace all speaker labels at once
                prompt = f"""
                Below is a transcript from a customer service call between an R2R Representative and a Caller.
                The transcript currently has generic speaker labels (like SPEAKER_01, SPEAKER_02).
                
                Please analyze the conversation flow and content, then return the EXACT SAME transcript but replace 
                each generic speaker label with either "R2R Representative" or "Caller" based on the context of what's being said.
                
                Maintain the exact same format including timestamps, tabs, and line breaks.
                Only change the speaker labels inside the square brackets. Do not reply with any extra explanation.
                
                Transcript:
                {transcript_content}
                """

                print(prompt)
                
                # Query Ollama
                response = ollama.chat(model="llama3.2:1b", messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options={
                        "num_gpu": 1,
                        "num_thread": 16
                    }
                )
                
                # Get the processed transcript with proper speaker roles
                processed_transcript = response['message']['content'].strip()

            if True:
                transcript_content = transcript_content.replace("SPEAKER_01", "R2R Representative")
                transcript_content = transcript_content.replace("SPEAKER_02", "Caller")
            txt_filepath = filepath + '.txt'
            with open(txt_filepath, 'w') as txt_file:
                txt_file.write(transcript_content)

            return jsonify({'success': True, 'message': 'Transcription complete!'})
        except Exception as e:
            print(e)
            return jsonify({'success': False, 'message': str(e)}), 500

    return jsonify({'success': False, 'message': 'Invalid file format'}), 400



if __name__ == '__main__':
    app.run(debug=True)