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
            print(f"Loaded {len(self.samples)} sample sets")
        except Exception as e:
            print(f"Error loading samples: {e}")
            sys.exit(1)

        self.current_set_index = 0

        # Name input
        name_frame = ttk.Frame(master)
        name_frame.pack(pady=5)
        ttk.Label(name_frame, text="Your ID:").pack(side=tk.LEFT, padx=(0, 5))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        self.name_entry.pack(side=tk.LEFT)

        # Main content frame
        content_frame = ttk.Frame(master)
        content_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Text displays and radio buttons
        self.text_displays = []
        self.relevance_vars = []
        for i in range(5):
            row_frame = ttk.Frame(content_frame)
            row_frame.pack(fill=tk.X, pady=5)

            text_display = tk.Text(row_frame, height=3, width=50, wrap=tk.WORD)
            text_display.pack(side=tk.LEFT, padx=(0, 10))
            text_display.config(state=tk.DISABLED)
            self.text_displays.append(text_display)

            var = tk.StringVar(value="0")
            self.relevance_vars.append(var)
            ttk.Radiobutton(row_frame, text="Not Relevant", variable=var, value="0").pack(side=tk.LEFT)
            ttk.Radiobutton(row_frame, text="Relevant", variable=var, value="1").pack(side=tk.LEFT)

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
        print("Loading samples from generated_questions.json")
        try:
            with open('generated_questions.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print("generated_questions.json file not found")
            raise
        except json.JSONDecodeError:
            print("Error decoding JSON in generated_questions.json")
            raise

    def display_current_sample(self):
        print(f"Displaying sample set {self.current_set_index + 1}")
        current_set = self.samples[self.current_set_index]
        for i in range(5):
            question_key = f"question{i+1}"
            text_display = self.text_displays[i]
            text_display.config(state=tk.NORMAL)
            text_display.delete(1.0, tk.END)
            text_display.insert(tk.END, f"{i+1}. {current_set[question_key]}")
            text_display.config(state=tk.DISABLED)

    def save_ratings(self):
        deceptiveness = self.deceptiveness_var.get()
        completeness = self.completeness_var.get()
        relevance = ''.join([var.get() for var in self.relevance_vars])
        name = self.name_var.get() or "Anonymous"
        print(f"Saving ratings: Name={name}, Set={self.current_set_index + 1}, Deceptiveness={deceptiveness}, Completeness={completeness}, Relevance={relevance}")
        try:
            with open('data.txt', 'a') as file:
                file.write(f"[{name}, Set{self.current_set_index + 1}, {deceptiveness}, {completeness}, {relevance}]\n")
        except IOError as e:
            print(f"Error saving ratings: {e}")

    def next_sample(self):
        print("Next button clicked")
        self.save_ratings()
        if self.current_set_index < len(self.samples) - 1:
            self.current_set_index += 1
            self.display_current_sample()
            self.reset_scales()
            self.update_button_states()

    def previous_sample(self):
        print("Previous button clicked")
        if self.current_set_index > 0:
            self.current_set_index -= 1
            self.display_current_sample()
            self.reset_scales()
            self.update_button_states()

    def reset_scales(self):
        self.deceptiveness_var.set(3)
        self.completeness_var.set(3)
        self.deceptiveness_label.config(text="3")
        self.completeness_label.config(text="3")
        for var in self.relevance_vars:
            var.set("0")

    def update_button_states(self):
        self.prev_button.config(state=tk.NORMAL if self.current_set_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_set_index < len(self.samples) - 1 else tk.DISABLED)

if __name__ == "__main__":
    print("Starting main application")
    try:
        root = tk.Tk()
        app = TextRatingApp(root)
        print("Entering main event loop")
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")