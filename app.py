from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from model import load_model
from knowledge import CONDITIONS

app = Flask(__name__)

CLASSES = ["benign", "melanoma", "nevus"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    return probs


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return "No file selected"

        image = Image.open(file)

        probs = predict(image)

        # Sort probabilities
        sorted_probs = sorted(
            [(CLASSES[i], probs[i].item()) for i in range(len(CLASSES))],
            key=lambda x: x[1],
            reverse=True
        )

        top_label, top_conf = sorted_probs[0]
        second_label, second_conf = sorted_probs[1]

        # Decision logic
        if top_conf < 0.5:
            final_label = "Uncertain"
            status = "Low confidence prediction"
        elif (top_conf - second_conf) < 0.1:
            final_label = "Uncertain"
            status = "Ambiguous between classes"
        else:
            final_label = top_label
            status = "Prediction is reasonably confident"

        result = {
            "final": final_label,
            "top": top_label,
            "confidence": round(top_conf * 100, 2),
            "status": status,
            "probs": {
                CLASSES[i]: round(probs[i].item() * 100, 2)
                for i in range(len(CLASSES))
            },
            "causes": CONDITIONS.get(top_label, {}).get("causes", []),
            "precautions": CONDITIONS.get(top_label, {}).get("precautions", [])
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)