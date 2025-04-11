import argparse
import os
import sys
import whisperx
import gc
import torch
import ollama

# Enable TF32 for better performance on CUDA devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration
TOKEN = "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"
CHATTING = False  # Toggle for Ollama processing

def process_audio_file(filepath, output_dir=None):
    print(f"Processing file: {filepath}")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(filepath) + '.txt')
    else:
        output_path = filepath + '.txt'
    
    try:
        print("Loading WhisperX model...")
        model = whisperx.load_model("base", "cuda", compute_type="float16")
        
        print("Transcribing audio...")
        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=16)
        
        print(f"Aligning transcript with detected language: {result['language']}")
        model_align, metadata = whisperx.load_align_model(language_code=result["language"], device="cuda")
        result = whisperx.align(result["segments"], model_align, metadata, audio, "cuda", return_char_alignments=False)
        
        print("Performing speaker diarization...")
        model_diarize = whisperx.DiarizationPipeline(use_auth_token=TOKEN, device="cuda")
        diarize_segments = model_diarize(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        print("Formatting transcript...")
        transcript_content = ""
        for line in result["segments"]:
            try:
                transcript_content += f"[{line['start']} -> {line['end']}] [{line['speaker']}]\t{line['text']}\n"
            except:
                line['speaker'] = "UNKNOWN"
                transcript_content += f"[{line['start']} -> {line['end']}] [{line['speaker']}]\t{line['text']}\n"
        
        if CHATTING:
            print("Processing with Ollama...")
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
            
            transcript_content = response['message']['content'].strip()
        else:
            transcript_content = transcript_content.replace("SPEAKER_01", "R2R Representative")
            transcript_content = transcript_content.replace("SPEAKER_02", "Caller")
        
        with open(output_path, 'w') as txt_file:
            txt_file.write(transcript_content)
        
        print(f"Transcription complete! Saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Audio transcription using WhisperX')
    parser.add_argument('-f', '--file', required=True, help='Path to audio file (MP3, WAV, or M4A)')
    parser.add_argument('-o', '--output', help='Output directory for transcription files')
    parser.add_argument('--chat', action='store_true', help='Enable Ollama processing for speaker identification')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist")
        sys.exit(1)
        
    if not args.file.endswith(('.mp3', '.wav', '.m4a')):
        print("Error: File must be MP3, WAV, or M4A format")
        sys.exit(1)

    global CHATTING
    CHATTING = args.chat
    
    success = process_audio_file(args.file, args.output)
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()