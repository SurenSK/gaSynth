import tkinter as tk
from tkinter import ttk
import json
import sys

print("Script started")

class TextRatingApp:
    def __init__(self, master):
        print("Initializing TextRatingApp")
        self.master = master
        master.title("Text Rating Application")

        try:
            self.samples = self.load_samples()
            print(f"Loaded {len(self.samples)} samples")
        except Exception as e:
            print(f"Error loading samples: {e}")
            sys.exit(1)

        self.current_index = 0

        # Name input
        name_frame = ttk.Frame(master)
        name_frame.pack(pady=5)
        ttk.Label(name_frame, text="Your ID:").pack(side=tk.LEFT, padx=(0, 5))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        self.name_entry.pack(side=tk.LEFT)

        # Text display
        self.text_display = tk.Text(master, height=10, width=50, wrap=tk.WORD)
        self.text_display.pack(pady=10)
        self.text_display.config(state=tk.DISABLED)

        # Deceptiveness scale
        ttk.Label(master, text="Deceptiveness:").pack()
        self.deceptiveness_var = tk.IntVar(value=3)
        self.deceptiveness_scale = ttk.Scale(master, from_=1, to=6, orient=tk.HORIZONTAL, 
                                             length=200, variable=self.deceptiveness_var, 
                                             command=lambda value: self.update_scale(self.deceptiveness_var, self.deceptiveness_label, value))
        self.deceptiveness_scale.pack()
        self.deceptiveness_label = ttk.Label(master, text="3")
        self.deceptiveness_label.pack()

        # Completeness scale
        ttk.Label(master, text="Completeness:").pack()
        self.completeness_var = tk.IntVar(value=3)
        self.completeness_scale = ttk.Scale(master, from_=1, to=6, orient=tk.HORIZONTAL, 
                                            length=200, variable=self.completeness_var, 
                                            command=lambda value: self.update_scale(self.completeness_var, self.completeness_label, value))
        self.completeness_scale.pack()
        self.completeness_label = ttk.Label(master, text="3")
        self.completeness_label.pack()

        # Navigation buttons
        self.button_frame = ttk.Frame(master)
        self.button_frame.pack(pady=10)
        self.prev_button = ttk.Button(self.button_frame, text="Previous", command=self.previous_sample)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.next_sample)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.display_current_sample()
        self.update_button_states()
        print("GUI setup complete")

    def update_scale(self, var, label, value):
        int_value = round(float(value))
        var.set(int_value)
        label.config(text=str(int_value))

    def load_samples(self):
        print("Loading samples from samples.jsonl")
        try:
            with open('samples.jsonl', 'r') as file:
                return [json.loads(line)['text'] for line in file]
        except FileNotFoundError:
            print("samples.jsonl file not found")
            raise
        except json.JSONDecodeError:
            print("Error decoding JSON in samples.jsonl")
            raise

    def display_current_sample(self):
        print(f"Displaying sample {self.current_index + 1}")
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, self.samples[self.current_index])
        self.text_display.config(state=tk.DISABLED)

    def save_ratings(self):
        deceptiveness = self.deceptiveness_var.get()
        completeness = self.completeness_var.get()
        name = self.name_var.get() or "Anonymous"
        print(f"Saving ratings: Name={name}, Deceptiveness={deceptiveness}, Completeness={completeness}")
        try:
            with open('data.txt', 'a') as file:
                file.write(f"[{name}, Sample{self.current_index + 1}, {deceptiveness}, {completeness}]\n")
        except IOError as e:
            print(f"Error saving ratings: {e}")

    def next_sample(self):
        if self.current_index < len(self.samples) - 1:
            print("Next button clicked")
            self.save_ratings()
            self.current_index += 1
            self.display_current_sample()
            self.reset_scales()
            self.update_button_states()

    def previous_sample(self):
        if self.current_index > 0:
            print("Previous button clicked")
            self.current_index -= 1
            self.display_current_sample()
            self.reset_scales()
            self.update_button_states()

    def reset_scales(self):
        self.deceptiveness_var.set(3)
        self.completeness_var.set(3)
        self.deceptiveness_label.config(text="3")
        self.completeness_label.config(text="3")

    def update_button_states(self):
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.samples) - 1 else tk.DISABLED)

if __name__ == "__main__":
    print("Starting main application")
    try:
        root = tk.Tk()
        app = TextRatingApp(root)
        print("Entering main event loop")
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")