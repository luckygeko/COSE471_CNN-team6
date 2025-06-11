import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import shap
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        self.target_layer.register_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

def create_region_masks(image_size=(224, 224)):
    masks = {}
    eye_mask = np.zeros(image_size, dtype=np.uint8)
    cv2.rectangle(eye_mask, (60, 70), (100, 100), 1, -1)
    cv2.rectangle(eye_mask, (124, 70), (164, 100), 1, -1)
    masks['eye'] = eye_mask
    forehead_mask = np.zeros(image_size, dtype=np.uint8)
    cv2.rectangle(forehead_mask, (70, 30), (154, 65), 1, -1)
    masks['forehead'] = forehead_mask
    mouth_mask = np.zeros(image_size, dtype=np.uint8)
    cv2.rectangle(mouth_mask, (85, 140), (140, 170), 1, -1)
    masks['mouth'] = mouth_mask
    return masks

def analyze_attention(cam_mask, region_masks):
    return {
        region: float(cam_mask[mask.astype(bool)].mean())
        for region, mask in region_masks.items()
    }

def explain_all_shap_features(region_scores, threshold=0.00001):
    explanations = []
    for region, score in region_scores.items():
        if score > threshold:
            explanations.append(f"The {region} area contributes to looking older.")
        elif score < -threshold:
            explanations.append(f"The {region} area contributes to looking younger.")
    if not explanations:
        explanations.append("No significant feature influence detected.")
    return explanations

def top_k_match(shap_scores, cam_scores, k=1):
    shap_top = sorted(shap_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:k]
    cam_top = sorted(cam_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return {region for region, _ in shap_top} & {region for region, _ in cam_top}

def summarize_appearance_reason(region_scores, cam_scores, shap_threshold=0.00001):
    shap_explanations = explain_all_shap_features(region_scores, threshold=shap_threshold)
    cam_focus = max(cam_scores, key=cam_scores.get)
    cam_value = cam_scores[cam_focus]

    print("\nSHAP-based Feature Contributions:")
    for line in shap_explanations:
        print("-", line)

    print(f"\nGrad-CAM shows highest attention on the {cam_focus} region (score: {cam_value:.3f}).")

    matched = top_k_match(region_scores, cam_scores, k=2)
    if matched:
        print(f"SHAP and Grad-CAM both highlight: {', '.join(matched)}")
    else:
        print("SHAP and Grad-CAM focus on different regions.")

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0), np.array(img.resize((224, 224)))

def main():
    # Load and preprocess the image
    image_path = "/Users/geko/COSE471_data-science/Test_IMG.jpg"
    input_tensor, original_img = preprocess(image_path)

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("model2.pth", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Grad-CAM
    cam = GradCAM(model, model.layer3[-1])
    cam_map = cam.generate(input_tensor)
    region_masks = create_region_masks()
    cam_scores = analyze_attention(cam_map, region_masks)

    # SHAP
    e = shap.GradientExplainer((model, model.layer4), input_tensor.clone())
    shap_values = e.shap_values(input_tensor)[0].mean(axis=0)
    shap_map_resized = cv2.resize(shap_values, (224, 224), interpolation=cv2.INTER_LINEAR)
    shap_scores = analyze_attention(shap_map_resized, region_masks)

    with torch.no_grad():
        pred = model(input_tensor.to(device)).item()
    
    # actual age and threshold for comparison
    true_age = 23
    threshold = 3
    
    print(f"Predicted Age: {pred:.2f} years")
    print(f"Actual Age: {true_age} years")

    if pred < true_age - threshold:
        print("The person looks younger than their age.")
    elif pred > true_age + threshold:
        print("The person looks older than their age.")
    else:
        print("The person looks appropriate for their age.")

    # explain the results
    summarize_appearance_reason(shap_scores, cam_scores)

if __name__ == "__main__":
    main()
