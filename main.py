import os
import re
import pickle
import random
import tkinter as tk
from tkinter import scrolledtext, ttk
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

#  PREPROCESSING
class ArabicPreprocessor:
    def __init__(self):
        self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
        self.arabic_punctuations = re.compile(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E\u060C\u061B\u061F\u066D\u06D4]')
        self.non_arabic_letters = re.compile(r'[^\u0621-\u063A\u0641-\u064A\s]')
        self.whitespace = re.compile(r'\s+')    #matches one or more white space

    def preprocess_text(self, text):
        # Remove diacritics (tashkeel)
        text = self.arabic_diacritics.sub('', text)
        
        # Remove punctuations
        text = self.arabic_punctuations.sub(' ', text)
        
        # Remove non-Arabic letters
        text = self.non_arabic_letters.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace.sub(' ', text)   #replace more than one line space or new lines with one white space
        return text.strip()

    def tokenize(self, text):
        
        return text.split()

    def process_file(self, file_path):
       
        all_tokens = []
        sentence_count = 0
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return all_tokens
            
        print(f"Processing file: {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                print(f"Successfully read file with {len(text)} characters")
        except Exception as e:
            print(f"Error reading file: {e}")
            # Try again with different encodings if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='cp1256') as f:  # Arabic Windows encoding
                    text = f.read()
                    print(f"Successfully read file with cp1256 encoding, {len(text)} characters")
            except Exception as e2:
                print(f"Error reading file with alternative encoding: {e2}")
                return all_tokens
        
        # Split into sentences 
        sentences = re.split(r'[.!?؟!\n]+', text)
        print(f"Split text into {len(sentences)} sentences")

        # Process each sentence
        for sentence in sentences:
            if sentence.strip():  # Skip empty sentences
                processed_text = self.preprocess_text(sentence)
                tokens = self.tokenize(processed_text)
                
                if len(tokens) > 3:  # Only include sentences with at least 3 tokens
                    all_tokens.extend(tokens)
                    sentence_count += 1
        
        print(f"Completed processing with {sentence_count} valid sentences.")
        print(f"Total tokens collected: {len(all_tokens)}")
        return all_tokens


# Part 2: N-GRAM MODEL TRAINING
class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.models = {}  
        self.vocab = set()
        
    def train(self, tokens, verbose=True):
        if verbose:
            print(f"Training n-gram models up to n={self.n}...")
        
        self.vocab = set(tokens)
        vocab_size = len(self.vocab)
        
        if verbose:
            print(f"Vocabulary size: {vocab_size} unique tokens")
        
        # Train models for each n-gram order
        for order in range(1, self.n + 1):
            if verbose:
                print(f"Training {order}-gram model...")
            
            # For each order, create a nested defaultdict
            model = defaultdict(Counter)
            
            # Generate n-grams and count
            for i in range(len(tokens) - order + 1):
                context = tuple(tokens[i:i+order-1]) if order > 1 else ()
                next_word = tokens[i+order-1]
                model[context][next_word] += 1
            
            # Convert counts to probabilities
            for context, next_word_counts in model.items():
                total_count = sum(next_word_counts.values())
                for word in next_word_counts:
                    next_word_counts[word] /= total_count
            
            self.models[order] = model
            
        if verbose:
            print("Training completed.")
            
    def save_model(self, file_path):
        #Save the trained model to a file
        with open(file_path, 'wb') as f:
            pickle.dump((self.models, self.vocab, self.n), f)
        print(f"Model saved to {file_path}")
            
    def load_model(self, file_path):
        #Load a trained model from a file
        with open(file_path, 'rb') as f:
            self.models, self.vocab, self.n = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return self
    
    def predict_next_words(self, context, num_suggestions=5):

        suggestions = []
        remaining = num_suggestions
        
        # Prepare context for each model
        tokens = context.strip().split()
        
        # Start with the highest order model and backoff if needed
        for order in range(self.n, 0, -1):
            if remaining <= 0:
                break
                
            # Get the appropriate context for this n-gram model
            if order == 1:
                context_tuple = ()  # Unigram has no context
            else:
                # Take the last n-1 tokens for context
                start_idx = max(0, len(tokens) - (order - 1))
                context_tuple = tuple(tokens[start_idx:])
                
                if len(context_tuple) < order - 1:
                    continue
            
            # Get predictions from this model
            if context_tuple in self.models[order]:
                # Sort by probability in descending order
                next_words = self.models[order][context_tuple].most_common(remaining)
                
                # Add unique suggestions
                for word, prob in next_words:
                    if word not in [s[0] for s in suggestions]:
                        suggestions.append((word, prob))
                        remaining -= 1
                        
                    if remaining <= 0:
                        break
        
        return suggestions


# Part 3: EVALUATION
class ModelEvaluator:
    def __init__(self, model):
        self.model = model
        
    def evaluate_perplexity(self, test_tokens):
        #Calculate perplexity of the model on test data

        # Using the highest-order n-gram model
        n = self.model.n
        log_likelihood = 0
        
        # Count the number of n-grams we can evaluate
        count = 0
        
        for i in range(len(test_tokens) - n + 1):
            context = tuple(test_tokens[i:i+n-1])
            next_word = test_tokens[i+n-1]
            
            # Get probability from model with backoff
            prob = 0
            for order in range(n, 0, -1):
                if order == 1:
                    current_context = ()
                else:
                    current_context = context[-(order-1):]
                
                if current_context in self.model.models[order] and next_word in self.model.models[order][current_context]:
                    prob = self.model.models[order][current_context][next_word]
                    break
            
            # Apply smoothing to avoid log(0)
            prob = max(prob, 1e-10)
            log_likelihood += np.log2(prob)
            count += 1
        
        # Calculate perplexity
        if count > 0:
            avg_log_likelihood = log_likelihood / count
            perplexity = 2 ** (-avg_log_likelihood)
            return perplexity
        else:
            return float('inf')
            
    def evaluate_accuracy(self, test_data, k=5):
        hits = 0
        total = 0
        
        for i in range(len(test_data) - self.model.n):
            # Get context
            context = ' '.join(test_data[i:i+self.model.n-1])
            actual_next_word = test_data[i+self.model.n-1]
            
            # Get predictions
            predictions = self.model.predict_next_words(context, k)
            predicted_words = [word for word, _ in predictions]
            
            # Check if the actual next word is in predictions
            if actual_next_word in predicted_words:
                hits += 1
            
            total += 1
        
        return hits / total if total > 0 else 0
        
    def evaluate_and_visualize(self, test_tokens, k_values=[1, 3, 5, 10]):
        #Evaluate the model and create visualizations
        # Calculate perplexity
        perplexity = self.evaluate_perplexity(test_tokens)
        print(f"Model perplexity: {perplexity:.2f}")
        
        # Calculate accuracy for different k values
        accuracies = []
        for k in k_values:
            acc = self.evaluate_accuracy(test_tokens, k)
            accuracies.append(acc)
            print(f"Top-{k} accuracy: {acc:.4f}")
        
        # Create accuracy visualization
        plt.figure(figsize=(10, 6))
        plt.bar(k_values, accuracies, color='skyblue')
        plt.xlabel('k (number of suggestions)')
        plt.ylabel('Accuracy')
        plt.title('Top-k Accuracy of Arabic Autocomplete Model')
        plt.xticks(k_values)
        plt.ylim(0, max(accuracies) * 1.2)
        
        # Add value labels on bars
        for i, acc in enumerate(accuracies):
            plt.text(k_values[i], acc + 0.01, f'{acc:.4f}', 
                    ha='center', va='bottom', fontsize=10)
        
        # Save the plot
        plt.savefig('accuracy_evaluation.png')
        plt.close()
        
        return {
            'perplexity': perplexity,
            'accuracies': dict(zip(k_values, accuracies))
        }


# Part 4: GUI INTERFACE
class ArabicAutocompleteGUI:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Arabic Autocomplete System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Configure the style
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 12))
        self.style.configure('TLabel', font=('Arial', 14), background="#f0f0f0")
        
        # Setup frames
        self.setup_frames()
        
        # Setup widgets
        self.setup_widgets()
        
    def setup_frames(self):
        # Main frames
        self.input_frame = ttk.Frame(self.root, padding=10)
        self.input_frame.pack(fill=tk.BOTH, expand=True)
        
        self.suggestion_frame = ttk.Frame(self.root, padding=10)
        self.suggestion_frame.pack(fill=tk.X)
        
        self.stats_frame = ttk.Frame(self.root, padding=10)
        self.stats_frame.pack(fill=tk.X)
        
    def setup_widgets(self):
        # Title
        title_label = ttk.Label(self.input_frame, text="Arabic Autocomplete System", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=10)
        
        # Instructions
        instr_label = ttk.Label(self.input_frame, 
                               text="Type in Arabic and get autocomplete suggestions",
                               font=('Arial', 12))
        instr_label.pack(pady=5)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(self.input_frame, height=10, 
                                                  font=('Arial', 14), wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=10)
        self.text_input.bind("<KeyRelease>", self.on_text_changed)
        
        # Suggestions area
        suggestion_label = ttk.Label(self.suggestion_frame, text="Suggestions:", 
                                    font=('Arial', 14, 'bold'))
        suggestion_label.pack(anchor="w")
        
        # Suggestion buttons container
        self.suggestion_buttons_frame = ttk.Frame(self.suggestion_frame)
        self.suggestion_buttons_frame.pack(fill=tk.X, pady=5)
        
        # Initialize suggestion buttons
        self.suggestion_buttons = []
        for i in range(5):
            btn = ttk.Button(self.suggestion_buttons_frame, text="", width=20,
                            command=lambda idx=i: self.use_suggestion(idx))
            btn.pack(side=tk.LEFT, padx=5)
            self.suggestion_buttons.append(btn)
        
        # Stats label
        self.stats_label = ttk.Label(self.stats_frame, text="Model Statistics: N=3, Vocabulary size: 0")
        self.stats_label.pack(pady=5)
        
    def update_suggestions(self, suggestions):
        """Update the suggestion buttons with new suggestions"""
        for i, btn in enumerate(self.suggestion_buttons):
            if i < len(suggestions):
                word, prob = suggestions[i]
                btn.config(text=f"{word} ({prob:.2f})", state=tk.NORMAL)
            else:
                btn.config(text="", state=tk.DISABLED)
    
    def on_text_changed(self, event=None):
        """Handle text change event to update suggestions"""
        text = self.text_input.get("1.0", tk.END).strip()
        
        if text:
            suggestions = self.model.predict_next_words(text)
            self.update_suggestions(suggestions)
        else:
            # Clear suggestions if no text
            for btn in self.suggestion_buttons:
                btn.config(text="", state=tk.DISABLED)
    
    def use_suggestion(self, index):
        """Insert the selected suggestion into the text"""
        if index < len(self.suggestion_buttons):
            btn_text = self.suggestion_buttons[index]['text']
            if btn_text:
                # Extract the word from button text (remove probability)
                word = btn_text.split(" (")[0]
                
                # Insert the word with a space
                self.text_input.insert(tk.END, f" {word}")
                
                # Update suggestions for the new text
                self.on_text_changed()
    
    def update_stats(self, vocab_size):
        """Update the statistics display"""
        self.stats_label.config(text=f"Model Statistics: N={self.model.n}, Vocabulary size: {vocab_size}")
    
    def run(self):
        """Run the GUI application"""
        # Update stats before starting
        self.update_stats(len(self.model.vocab))
        self.root.mainloop()


# Main Application
def main():
    # Path to specific file - UPDATE THIS TO YOUR ACTUAL PATH
    data_file = r"project\All Data\All Data"   # Path to the dataset file
    model_path = r"project\arabic_ngram_model_full.pkl"  # Path to save/load the model
    
    # Create instances
    preprocessor = ArabicPreprocessor()
    model = NGramModel(n=3)  # Using trigrams
    
    # Check if model already exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_model(model_path)
    else:
        print(f"Processing data from file {data_file} and training a new model...")
        
        # Process the specific file
        all_tokens = preprocessor.process_file(data_file)
        
        if not all_tokens:
            print("No tokens were extracted from the file. Please check the file path and content.")
            return
            
        print(f"Total tokens after preprocessing: {len(all_tokens)}")
        
        # Split data for training and evaluation (90% train, 10% test)
        train_size = int(0.8 * len(all_tokens))
        train_tokens = all_tokens[:train_size]
        test_tokens = all_tokens[train_size:]
        
        print(f"Training with {len(train_tokens)} tokens, testing with {len(test_tokens)} tokens...")
        
        # Train model
        model.train(train_tokens)
        
        # Save the model
        model.save_model(model_path)
        
        # Evaluate model
        print("Evaluating model...")
        evaluator = ModelEvaluator(model)
        results = evaluator.evaluate_and_visualize(test_tokens)
        print("Evaluation Results:", results)
    
    # Start GUI
    print("Starting GUI...")
    app = ArabicAutocompleteGUI(model)
    app.run()


if __name__ == "__main__":
    main()