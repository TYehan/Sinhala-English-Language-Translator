from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
import nltk
import os

# Suppress NLTK download output in production, but let's keep it for now.
# nltk.download('punkt', quiet=True)

class TranslationService:
    """
    A service class to handle loading translation models and performing translation.
    """
    def __init__(self, dataset_path="sinhala_english_sentences_dataset.csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self._load_models()
        self._load_dataset(dataset_path)

    def _load_dataset(self, path):
        """Loads the translation dataset for lookup."""
        self.lookup = {}
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Assuming columns are named 'english' and 'sinhala'
                self.lookup["English"] = pd.Series(df.sinhala.values, index=df.english).to_dict()
                self.lookup["Sinhala"] = pd.Series(df.english.values, index=df.sinhala).to_dict()
                print("Successfully loaded dataset for lookup.")
            except Exception as e:
                print(f"Error loading dataset: {e}. Lookup will be disabled.")
        else:
            print(f"Warning: Dataset not found at {path}. Lookup will be disabled.")

    def _load_models(self):
        """Loads the translation models and tokenizers."""
        print("Loading translation models...")
        
        try:
            # Check for trained model first
            if os.path.exists("trained_model") and os.path.exists("trained_model/config.json"):
                print("Loading custom fine-tuned model...")
                
                # Load the fine-tuned model (works for both directions)
                self.tokenizers["en-si"] = MarianTokenizer.from_pretrained("trained_model")
                self.models["en-si"] = MarianMTModel.from_pretrained("trained_model")
                
                # Use the same model for both directions
                self.tokenizers["si-en"] = self.tokenizers["en-si"]
                self.models["si-en"] = self.models["en-si"]
                
                print("✅ Custom fine-tuned model loaded successfully!")
                
            else:
                print("No trained model found. Loading default pre-trained models...")
                
                # Load default models
                en_si_model = "Helsinki-NLP/opus-mt-en-mul"
                si_en_model = "Helsinki-NLP/opus-mt-mul-en"
                
                self.tokenizers["en-si"] = MarianTokenizer.from_pretrained(en_si_model)
                self.models["en-si"] = MarianMTModel.from_pretrained(en_si_model)
                
                self.tokenizers["si-en"] = MarianTokenizer.from_pretrained(si_en_model)
                self.models["si-en"] = MarianMTModel.from_pretrained(si_en_model)
                
                print("✅ Default pre-trained models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Falling back to lookup-only mode...")
            self.models = {}
            self.tokenizers = {}
            return

        # Move models to device and set to evaluation mode
        for model in self.models.values():
            model.to(self.device)
            model.eval()
            
        print(f"Models moved to device: {self.device}")

    def perform_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates a given text from a source language to a target language.
        """
        print(f"Translating '{text}' from {source_lang} to {target_lang}")
        
        # Check lookup first for exact matches
        if source_lang in self.lookup:
            # Case-insensitive lookup
            for key, value in self.lookup[source_lang].items():
                if str(key).lower().strip() == text.lower().strip():
                    print(f"Found exact match in lookup: {value}")
                    return value

        # If no models are loaded, fall back to lookup only
        if not self.models:
            return f"Translation not found. Models not available."

        # Determine model key and handle language codes
        if source_lang.lower() == "english" and target_lang.lower() == "sinhala":
            model_key = "en-si"
            # For fine-tuned model, we still use the language prefix
            input_text = f">>sin<< {text}"
        elif source_lang.lower() == "sinhala" and target_lang.lower() == "english":
            model_key = "si-en"  # This will use the same model as en-si
            # For Sinhala to English, no prefix needed
            input_text = text
        else:
            return f"Translation from {source_lang} to {target_lang} is not supported."

        if model_key not in self.models:
            return f"Model for {source_lang} to {target_lang} is not available."

        try:
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]

            # Tokenize and translate
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated_ids = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            
            translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            
            print(f"AI Translation result: {translated_text}")
            return translated_text
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation failed: {str(e)}"

app = Flask(__name__)
translation_service = TranslationService(dataset_path="sinhala_english_sentences_dataset.csv")

@app.route('/', methods=['GET', 'POST'])
def home_page():
    translation_result = None
    source_language = request.form.get('source_lang', 'English')
    target_language = request.form.get('target_lang', 'Sinhala')
    source_text = request.form.get('source_text', '')

    if request.method == 'POST':
        if source_text:
            translated_text = translation_service.perform_translation(source_text, source_language, target_language)
            translation_result = {'translation': translated_text}

    return render_template('index.html', 
                         result=translation_result, 
                         source_lang=source_language, 
                         target_lang=target_language,
                         source_text=source_text)

if __name__ == '__main__':
    import sys
    print(f"🐍 Running with Python: {sys.executable}")
    print(f"🌐 Starting TranslateHub server...")
    print(f"📍 Access the application at: http://localhost:5000")
    print(f"⚡ Press Ctrl+C to stop the server")
    print("-" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
