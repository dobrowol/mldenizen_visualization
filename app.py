from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os

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
        print("📥 Received data:", data) 
        X = np.array(data["X"])
        y = np.array(data["y"])
        eta = float(data["eta"])
        epochs = int(data["epochs"])

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        w = np.zeros(X.shape[1])
        b = 0

        plt.sca(axs[0])
        plot_decision_boundary(X, y, w, b, "Epoch 0")

        for epoch in range(epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_pred = predict(x_i, w, b)
                error = y[i] - y_pred
                w += eta * error * x_i
                b += eta * error

            if epoch == 1:
                plt.sca(axs[1])
                plot_decision_boundary(X, y, w, b, "Epoch 2")

        plt.sca(axs[2])
        plot_decision_boundary(X, y, w, b, f"Epoch {epochs}")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({"image": f"data:image/png;base64,{img_b64}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)