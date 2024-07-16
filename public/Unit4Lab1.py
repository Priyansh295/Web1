from flask import Flask, request, render_template_string

app = Flask(__name__)

class BajajExpertSystem:
    def __init__(self):
        self.rules = [
            {"conditions": ["not_starting", "battery_low"], "conclusion": "Charge the battery."},
            {"conditions": ["not_starting", "battery_ok"], "conclusion": "Check the starter motor."},
            {"conditions": ["starting", "stalls_frequently"], "conclusion": "Check the fuel supply."},
            {"conditions": ["starting", "poor_acceleration"], "conclusion": "Check the air filter."},
            {"conditions": ["starting", "unusual_noises"], "conclusion": "Check the engine."},
        ]

    def diagnose(self, symptoms):
        for rule in self.rules:
            if all(condition in symptoms for condition in rule["conditions"]):
                return rule["conclusion"]
        return "No diagnosis found. Please consult a professional mechanic."

@app.route('/', methods=['GET', 'POST'])
def home():
    system = BajajExpertSystem()
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        diagnosis = system.diagnose(symptoms)
        return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        text-align: center;
                        padding: 50px;
                    }
                    h1 {
                        color: #333;
                    }
                    p {
                        color: #666;
                    }
                    .button {
                        background-color: #4CAF50;
                        color: white;
                        padding: 14px 20px;
                        margin: 8px 0;
                        border: none;
                        cursor: pointer;
                        transition: all 0.3s ease 0s;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                    form {
                        background-color: #ffffff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                    }
                    label {
                        margin-right: 10px;
                    }
                </style>
            </head>
            <body>
                <h1>Diagnosis Result</h1>
                <p>{{ diagnosis }}</p>
                <a href="/" class="button">Back</a>
            </body>
            </html>
        """, diagnosis=diagnosis)
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 50px;
                }
                h1 {
                    color: #333;
                }
                form {
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px 0 rgba(0,0,0,0.1);
                }
                .button {
                    background-color: #008CBA;
                    color: white;
                    padding: 15px 20px;
                    margin: 10px 0;
                    border: none;
                    cursor: pointer;
                    transition: opacity 0.3s ease-in-out;
                }
                .button:hover {
                    opacity: 0.7;
                }
                label {
                    margin-right: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to the Bajaj Expert System</h1>
            <form method="post">
                <label><input type="checkbox" name="symptoms" value="not_starting"> Not Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_low"> Battery Low</label><br>
                <label><input type="checkbox" name="symptoms" value="battery_ok"> Battery OK</label><br>
                <label><input type="checkbox" name="symptoms" value="starting"> Starting</label><br>
                <label><input type="checkbox" name="symptoms" value="stalls_frequently"> Stalls Frequently</label><br>
                <label><input type="checkbox" name="symptoms" value="poor_acceleration"> Poor Acceleration</label><br>
                <label><input type="checkbox" name="symptoms" value="unusual_noises"> Unusual Noises</label><br>
                <input type="submit" value="Diagnose" class="button">
            </form>
        </body>
        </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)
