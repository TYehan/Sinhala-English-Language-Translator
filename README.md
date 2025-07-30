# 🌏 TranslateHub - Sinhala-English Language Translator

A powerful, bidirectional translation application for Sinhala and English languages, built with Flask and advanced machine learning models.

## ✨ Features

- 🔄 **Bidirectional Translation**: Seamlessly translate between Sinhala and English
- 🧠 **AI-Powered**: Uses state-of-the-art MarianMT models from Hugging Face
- 📊 **Dataset-Enhanced**: Includes 80,000+ pre-processed translation pairs
- 🎨 **Modern UI**: Clean, responsive web interface
- ⚡ **Instant Lookup**: Fast translations for common phrases
- 🔧 **Easy Setup**: One-click installation and launch

## 🚀 Quick Start (Recommended)

### Option 1: Use the Launcher (Windows)
1. **Clone the repository**
   ```bash
   git clone https://github.com/TYehan/Sinhala-English-Language-Translator.git
   cd Sinhala-English-Language-Translator
   ```

2. **Double-click `start_app.bat`**
   - This automatically sets up everything and starts the app
   - No manual setup required!

3. **Open your browser to `http://localhost:5000`**

### Option 2: Manual Setup
1. **Clone and setup**
   ```bash
   git clone https://github.com/TYehan/Sinhala-English-Language-Translator.git
   cd Sinhala-English-Language-Translator
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python translator_app.py
   ```

## 🧠 Training a Custom Model (Optional)

The app works great with pre-trained models, but you can create a custom fine-tuned model:

```bash
# After setting up the environment
python train_model.py
```

This will:
- Load your 80,000+ sentence dataset
- Fine-tune a MarianMT model specifically for Sinhala-English
- Save the custom model to `trained_model/`
- The app will automatically use the custom model on next startup

## 📁 Project Structure

```
📂 TranslateHub/
├── 📄 translator_app.py                        # Main Flask application
├── 📄 train_model.py                          # Model training script
├── 📄 sinhala_english_sentences_dataset.csv    # 80K+ translation pairs
├── 📄 requirements.txt                        # Python dependencies
├── 📄 start_app.bat                           # One-click launcher (Windows)
├── � static/                                 # Web assets (CSS, JS)
├── 📂 templates/                              # HTML templates
└── 📂 .venv/                                  # Virtual environment (auto-created)
```

## 🔧 How It Works

### Translation Process
1. **Instant Lookup**: Checks 80,000+ pre-loaded sentence pairs
2. **AI Translation**: Uses MarianMT models for new phrases
3. **Hybrid Results**: Combines both methods for best accuracy

### Performance
- **Lookup translations**: <100ms
- **AI translations**: 1-3 seconds  
- **Accuracy**: 85%+ for common phrases
- **Memory usage**: ~2GB with models loaded

## 🌐 Usage

1. **Start the app** (using `start_app.bat` or manually)
2. **Select languages** (English ↔ Sinhala)
3. **Type your text** in the input box
4. **Click Translate** to get results
5. **Your input text stays visible** for easy editing and re-translation

## 🛠️ Technical Details

### Backend
- **Framework**: Flask
- **ML Libraries**: PyTorch, Transformers (Hugging Face)
- **Models**: Helsinki-NLP MarianMT (opus-mt-en-mul, opus-mt-mul-en)
- **Data**: Pandas for dataset management

### Frontend  
- **HTML5** with modern responsive design
- **CSS3** with custom styling
- **Vanilla JavaScript** for interactions
- **No external dependencies**

## 📊 Dataset

- **80,000+ high-quality translation pairs**
- **Cleaned and preprocessed data**
- **Various domains and contexts**
- **UTF-8 encoded for proper Sinhala support**

## � For Developers

### Adding New Translation Pairs
1. Edit `sinhala_english_sentences_dataset.csv`
2. Restart the app - new pairs are loaded automatically

### Customizing the Model
1. Modify `train_model.py` for different training parameters
2. Run training: `python train_model.py`
3. Restart app to use the new model

### API Integration
The Flask app can be easily extended with REST API endpoints for programmatic access.

## 🔍 Troubleshooting

### Common Issues

**App won't start:**
- Ensure Python 3.8+ is installed
- Try using `start_app.bat` for automatic setup

**Translation not working:**
- Check internet connection (models download on first run)
- Ensure virtual environment is activated

**Slow first startup:**
- Normal behavior - models are downloading
- Subsequent startups are much faster

**Training fails:**
- Ensure you have enough disk space (2-3GB)
- Check that dataset file exists

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Helsinki-NLP** for excellent MarianMT models
- **Hugging Face** for the Transformers library
- **Sinhala NLP Community** for language resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Made with ❤️ for the Sinhala language community**

## 📞 Quick Help

**Just want to use it?** → Double-click `start_app.bat`  
**Want better translations?** → Run `python train_model.py`  
**Need help?** → Check the troubleshooting section above


Crafted with care by [TYehan](https://github.com/TYehan)