from PIL import Image, ImageDraw
import torch 
import numpy as np

def draw(bboxes: torch.Tensor, image_path: str):
    image_path = 'example.jpg'
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        bbox = bbox.int()
        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    image.save(f'output_{image_path}.jpg')


if __name__ == '__main__':
    bboxes = None
    img = 'example.jpg'

    draw(bboxes, img)
