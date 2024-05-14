import random
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import time

class ChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Classifier")
        self.root.geometry("600x400")
        self.root.configure(bg="#F0F0F0")

        self.frame = tk.Frame(self.root, bg="#F0F0F0", bd=0)
        self.frame.place(relx=0.5, rely=0.5, relwidth=0.8, relheight=0.7, anchor="center")

        self.heading_label = tk.Label(self.frame, text="Emotion Classifier", font=("Helvetica", 24), bg="#F0F0F0")
        self.heading_label.pack(pady=10)

        self.chat_log = tk.Text(self.frame, height=15, width=50, font=("Helvetica", 12), bd=0, bg="#FFFFFF", wrap="word")
        self.chat_log.pack(pady=10)

        self.text_entry = tk.Text(self.frame, height=4, width=50, font=("Helvetica", 12), bd=0, bg="#FFFFFF", wrap="word")
        self.text_entry.pack(pady=5)

        self.classify_button = tk.Button(self.frame, text="Send", font=("Helvetica", 12), bg="#007ACC", fg="white", command=self.classify_text)
        self.classify_button.pack(pady=5)

        self.clear_button = tk.Button(self.frame, text="Clear Chat", font=("Helvetica", 12), bg="#E63946", fg="white", command=self.clear_chat)
        self.clear_button.pack(side="left", padx=10)

        self.exit_button = tk.Button(self.frame, text="Exit", font=("Helvetica", 12), bg="#E63946", fg="white", command=self.exit_app)
        self.exit_button.pack(side="right", padx=10)

        self.tfidf_vectorizer = TfidfVectorizer()
        self.model = make_pipeline(self.tfidf_vectorizer, MultinomialNB())

        try:
            self.data = pd.read_csv('emotion_sentimen_dataset.csv')
        except FileNotFoundError:
            messagebox.showerror("Error", "Dataset file 'emotion_sentimen_dataset.csv' not found.")
            exit(1)
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", "Dataset file is empty.")
            exit(1)

        if 'text' not in self.data.columns or 'Emotion' not in self.data.columns:
            messagebox.showerror("Error", "Dataset does not have the required columns 'text' and 'Emotion'.")
            exit(1)

        self.X = self.data['text']
        self.y = self.data['Emotion']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def classify_text(self):
        text = self.text_entry.get("1.0", "end").strip()
        if len(text.split()) > 1:  # Check if the input contains more than one word
            emotion = self.classify_emotion(text)
            quote = self.get_random_quote(emotion)
            self.display_message(text, "user")
            self.display_message(emotion, "bot")
            self.display_message(quote, "bot")
            self.text_entry.delete("1.0", "end")
        else:
            messagebox.showwarning("Warning", "Please enter a valid sentence.")
    def classify_emotion(self, text):
        predicted_emotion = self.model.predict([text])[0]
        return predicted_emotion

    def get_random_quote(self, emotion):
        emotion_quotes = {
            'joy': [
                "The purpose of our lives is to be happy. - Dalai Lama",
                "Count your age by friends, not years. Count your life by smiles, not tears. - John Lennon",
                "Happiness is not something ready-made. It comes from your own actions. - Dalai Lama"
            ],
            'sadness': [
                "The way sadness works is one of the strange riddles of the world. - Lemony Snicket",
                "Tears are words that need to be written. - Paulo Coelho",
                "Every man has his secret sorrows which the world knows not; and often times we call a man cold when he is only sad. - Henry Wadsworth Longfellow"
            ],
            'anger': [
                "Anger is an acid that can do more harm to the vessel in which it is stored than to anything on which it is poured. - Mark Twain",
                "For every minute you are angry you lose sixty seconds of happiness. - Ralph Waldo Emerson",
                "Speak when you are angry and you will make the best speech you will ever regret. - Ambrose Bierce"
            ],
            'fear': [
                "The only thing we have to fear is fear itself. - Franklin D. Roosevelt",
                "Do not be afraid of your fears. They're not there to scare you. They're there to let you know that something is worth it. - C. JoyBell C.",
                "The oldest and strongest emotion of mankind is fear, and the oldest and strongest kind of fear is fear of the unknown. - H.P. Lovecraft"
            ],
            'neutral': [
                "Life is neither good or bad, it just depends on your point of view. - Paulo Coelho",
                "Stay calm and carry on. - Winston Churchill",
                "Sometimes the most productive thing you can do is relax. - Mark Black"
            ],
            'hate': [
                "Hate, it has caused a lot of problems in the world, but has not solved one yet. - Maya Angelou",
                "Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that. - Martin Luther King Jr.",
                "The opposite of love is not hate, it's indifference. - Elie Wiesel"
            ],
            'love': [
                "Love is like the wind, you can't see it but you can feel it. - Nicholas Sparks",
                "The greatest happiness you can have is knowing that you do not necessarily require happiness. - William Saroyan",
                
                "Being deeply loved by someone gives you strength, while loving someone deeply gives you courage. - Lao Tzu"
            ],
            'enthusiasm': [
                "Nothing great was ever achieved without enthusiasm. - Ralph Waldo Emerson",
                "Your enthusiasm will be infectious, stimulating and attractive to others. They will love you for it. They will go for you and with you. - Norman Vincent Peale",
                "Enthusiasm is the yeast that makes your hopes shine to the stars. Enthusiasm is the sparkle in your eyes, the swing in your gait. - Henry Ford"
            ],
            'boredom': [
                "Only boring people get bored. - Ruth Burke",
                "Boredom is the feeling that everything is a waste of time; serenity, that nothing is. - Thomas Szasz",
                "Boredom is simply the absence of an interesting perspective. - Naval Ravikant"
            ]
        }

        if emotion in emotion_quotes:
            quotes = emotion_quotes[emotion]
            return random.choice(quotes)
        else:
            return "Unable to find quotes for this emotion."

    def display_message(self, message, sender):
        if sender == "user":
            self.chat_log.insert(tk.END, f"You: {message}\n\n")
        else:
            self.chat_log.insert(tk.END, f"Bot: {message}\n\n")
        self.chat_log.see(tk.END)

    def typing_animation(self, sender):
        if sender == "bot":
            self.display_message("Bot is typing...", "bot")
            self.root.update()
            time.sleep(1)  # Simulating typing delay

    def clear_chat(self):
        self.chat_log.delete("1.0", tk.END)

    def exit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    chat_bot = ChatBot(root)
    root.mainloop()
