from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import subprocess
import tempfile

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://mldenizen.com"}})


def predict(x, w, b):
    return 1 if np.dot(w, x) + b >= 0 else 0

def plot_decision_boundary(X, y, w, b, title):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([predict(pt, w, b) for pt in grid])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

@app.route("/visualize", methods=["POST"])
def visualize():
    try:
        data = request.json
        code = data.get("code", "")
        if not code:
            return jsonify({"error": "No code provided"}), 400

        # Save code to a temporary file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        # Run the code and capture output
        result = subprocess.run(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )

        os.remove(tmp_path)  # Clean up

        if result.returncode != 0:
            return jsonify({"error": result.stderr.decode("utf-8")}), 400

        output = result.stdout.decode("utf-8")
        return jsonify({"output": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)