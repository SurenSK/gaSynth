from flask import Flask, render_template, request, jsonify, redirect, url_for
import json

app = Flask(__name__)

# Load samples from JSON file
def load_samples():
    samples = []
    try:
        with open('ga_output.jsonl', 'r') as file:  # Make sure the file extension is correct
            for line in file:
                samples.append(json.loads(line))
    except FileNotFoundError:
        print("generated_questions.jsonl file not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error decoding JSONL: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    return samples


samples = load_samples()
current_set_index = 0

@app.route('/')
def index():
    if samples:
        current_set = samples[current_set_index]
        return render_template('index.html', sample_set=current_set, set_index=current_set_index + 1, total_sets=len(samples))
    else:
        return "Error loading samples", 500

@app.route('/rate', methods=['POST'])
def rate():
    global current_set_index
    # Save current ratings
    # Extract ratings from request.form, similar to how they're handled in the Tkinter app

    # Navigation logic (Next/Previous)
    if 'next' in request.form:
        current_set_index = min(current_set_index + 1, len(samples) - 1)
    elif 'previous' in request.form:
        current_set_index = max(current_set_index - 1, 0)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
