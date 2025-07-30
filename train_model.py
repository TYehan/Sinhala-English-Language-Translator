#!/usr/bin/env python3
"""
Model Training Script for TranslateHub
Creates a fine-tuned MarianMT model for Sinhala-English translation
"""

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import os
from datetime import datetime
from typing import Union
from tqdm import tqdm

def main():
    print("=" * 60)
    print("TRANSLATEHUB - MODEL TRAINING")
    print("=" * 60)
    
    # Check system capabilities
    print(f"PyTorch version: {torch.__version__}")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset_path = "sinhala_english_sentences_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset file exists in the project directory.")
        return False
    
    df = pd.read_csv(dataset_path)
    df = df.dropna()  # Remove any empty rows
    print(f"✅ Loaded {len(df)} translation pairs")
    print(f"Sample: '{df.iloc[0]['english']}' → '{df.iloc[0]['sinhala']}'")
    print()
    
    # Load pre-trained model
    print("🤖 Loading pre-trained MarianMT model...")
    model_name = "Helsinki-NLP/opus-mt-en-mul"
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model = model.to(device)  # type: ignore
        model.train()  # Set to training mode
        print(f"✅ Successfully loaded: {model_name}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Prepare training data
    print()
    print("📋 Preparing training data...")
    
    # Use subset for training (adjust based on your system capacity)
    train_size = min(1000, len(df))  # Start with 1000 samples
    train_df = df.sample(n=train_size, random_state=42).reset_index(drop=True)
    print(f"Using {train_size} samples for training")
    
    # Prepare training batches
    batch_size = 4  # Small batch size for stability
    num_epochs = 3
    learning_rate = 5e-5
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = (len(train_df) // batch_size) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    print(f"Training setup:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Total steps: {total_steps}")
    
    # Training loop
    print()
    print("🚀 Starting model training...")
    print("=" * 50)
    
    model.train()
    total_loss = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        # Create progress bar
        pbar = tqdm(range(0, len(train_df), batch_size), desc=f"Training Epoch {epoch + 1}")
        
        for i in pbar:
            batch_df = train_df.iloc[i:i + batch_size]
            
            # Prepare batch inputs and targets
            inputs = []
            targets = []
            
            for _, row in batch_df.iterrows():
                # English to Sinhala
                input_text = f">>sin<< {row['english']}"
                target_text = row['sinhala']
                
                inputs.append(input_text)
                targets.append(target_text)
            
            # Tokenize
            input_encodings = tokenizer(
                inputs, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            target_encodings = tokenizer(
                targets, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = input_encodings.input_ids.to(device)
            attention_mask = input_encodings.attention_mask.to(device)
            labels = target_encodings.input_ids.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / (len(train_df) // batch_size)
        print(f"Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / total_steps
    print()
    print("=" * 50)
    print(f"🎯 Training completed! Average Loss: {avg_total_loss:.4f}")
    
    # Test a few translations
    print()
    print("🧪 Testing trained model...")
    model.eval()
    
    test_sentences = ["hello", "good morning", "thank you"]
    for sentence in test_sentences:
        try:
            input_text = f">>sin<< {sentence}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=4)
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"  '{sentence}' → '{translation}'")
        except Exception as e:
            print(f"  '{sentence}' → Error: {e}")
    
    # Save the model
    print()
    print("💾 Saving trained model...")
    output_dir = "trained_model"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Create model info file
        with open(f"{output_dir}/model_info.txt", "w", encoding="utf-8") as f:
            f.write(f"TranslateHub Fine-Tuned Model\n")
            f.write(f"============================\n")
            f.write(f"Base model: {model_name}\n")
            f.write(f"Training samples: {train_size} (from {len(df)} total)\n")
            f.write(f"Epochs: {num_epochs}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Final average loss: {avg_total_loss:.4f}\n") 
            f.write(f"Device: {device}\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: sinhala_english_sentences_dataset.csv\n")
        
        print(f"✅ Model saved to '{output_dir}' directory")
        print(f"✅ Model info saved to '{output_dir}/model_info.txt'")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    print()
    print("=" * 60)
    print("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Restart the translator app: python translator_app.py")
    print("2. The app will automatically detect and use the fine-tuned model")
    print("3. Access the app at: http://localhost:5000")
    print("4. Your model should now provide better translations!")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed. Please check the errors above.")
        exit(1)
    else:
        print("✅ Training completed successfully!")
