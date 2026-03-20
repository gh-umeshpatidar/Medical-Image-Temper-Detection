# can we use more Best Model
import torch
from ml_models.image_model import image_model, device
from ml_models.metadata_model import predict_metadata
from app.services.preprocess import preprocess_image
from app.services.dicom_metadata import extract_metadata
from app.services.metadata_rules import check_metadata_rules

# For image
def run_inference_image(file_path):

    image = preprocess_image(file_path).to(device)

    with torch.no_grad():
        outputs = image_model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "tampered": bool(predicted.item()),
        "confidence": round(confidence.item(), 4),
        "class": "Tampered" if predicted.item() == 1 else "Authentic"
    }


def run_inference_dicom(file_path):

    image = preprocess_image(file_path).to(device)
    image_outputs = image_model(image)
    image_probabilities = torch.softmax(image_outputs, dim=1)
    # confidence, predicted = torch.max(image_probabilities, 1)

    metadata_list = extract_metadata(file_path)
    # metadata_issue  = check_metadata_rules(metadata_list)

    metadata_ouput = predict_metadata(metadata_list)
    metadata_probabilities = float(metadata_ouput)

    score = 0

    # Visual tampering weight
    score += image_probabilities * 0.7

    # Metadata ML weight
    score += metadata_probabilities * 0.3

    # Metadata rule violations
    # if len(metadata_issues) > 0:
    #     score += 0.1

    final = score > 0.4

    return {
        "tampered": final,
        "risk_score": round(score, 3),
        "image_tamper_probability": round(image_probabilities, 3),
        "metadata_tamper_probability": round(metadata_probabilities, 3),
        # "metadata_issues": metadata_issue
    }