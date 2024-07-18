import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont

class TableDetector:
    def __init__(self, model_name="TahaDouaji/detr-doc-table-detection"):
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.processor = DetrImageProcessor.from_pretrained(model_name)

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detections.append({
                    "score": round(score.item(), 3),
                    "label": self.model.config.id2label[label.item()],
                    "box": box
                })

            return detections
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    def draw_boxes(self, tables, image_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for table in tables:
            box = table["box"]
            draw.rectangle(box, outline="red", width=2)
        return image
        