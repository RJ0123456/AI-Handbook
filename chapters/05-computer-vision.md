# Chapter 5: Computer Vision

## What Is Computer Vision?

Computer Vision (CV) is the field of AI that enables machines to interpret and understand visual information from the world — images, video, and other visual inputs.

Humans process visual information effortlessly, but encoding this ability in machines requires solving hard problems: recognizing objects despite changes in scale, lighting, angle, and occlusion; understanding spatial relationships; and reasoning about motion and scene context.

## Core Computer Vision Tasks

| Task | Description | Example |
|------|-------------|---------|
| Image Classification | Assign a label to an entire image | "This is a cat" |
| Object Detection | Locate and classify objects in an image | Bounding boxes around cars and pedestrians |
| Image Segmentation | Classify each pixel | Separate road, sky, buildings in a scene |
| Pose Estimation | Detect body keypoints | Skeleton overlay on a person |
| Optical Character Recognition (OCR) | Extract text from images | Digitizing printed documents |
| Image Generation | Create new images | Text-to-image (Stable Diffusion, DALL-E) |
| Image Captioning | Describe an image in text | "A dog playing in the park" |
| Video Understanding | Analyze temporal sequences | Action recognition, tracking |
| Depth Estimation | Predict distance from a single image | Monocular depth for robotics |
| Face Recognition | Identify individuals | Unlock a phone, security cameras |

## Image Basics

- **Pixel:** The smallest unit of a digital image. Represented as intensity values.
- **Grayscale image:** Single channel; each pixel is a value from 0 (black) to 255 (white).
- **Color image (RGB):** Three channels — Red, Green, Blue — each 0–255.
- **Resolution:** Width × Height in pixels.

## Convolutional Neural Networks for Vision

CNNs are the foundation of modern computer vision.

### How Convolution Works

A filter (kernel) slides across the input image, computing a dot product at each position to produce a **feature map**. Filters learn to detect patterns: edges → textures → shapes → objects.

```
Input Image (H × W × C) → Conv Layer → Feature Maps → Pooling → ... → Output
```

### Landmark CNN Architectures

| Model | Year | Key Innovation |
|-------|------|---------------|
| LeNet-5 | 1998 | First successful CNN (handwriting) |
| AlexNet | 2012 | Deep CNN on GPU; ImageNet breakthrough |
| VGGNet | 2014 | Very deep networks with 3×3 filters |
| GoogLeNet (Inception) | 2014 | Inception modules; parallel paths |
| ResNet | 2015 | Residual connections; 152 layers |
| DenseNet | 2016 | Dense connections between all layers |
| EfficientNet | 2019 | Compound scaling; state-of-the-art efficiency |
| Vision Transformer (ViT) | 2020 | Applying Transformers to image patches |

## Object Detection

Detect **where** objects are (bounding boxes) and **what** they are (class labels).

### Key Approaches

| Approach | Models | Description |
|----------|--------|-------------|
| Two-stage | R-CNN, Fast R-CNN, Faster R-CNN | Region proposal + classification |
| One-stage | YOLO, SSD, RetinaNet | Direct prediction; faster inference |
| Transformer-based | DETR, DINO | End-to-end detection with attention |

**YOLO (You Only Look Once)** is the most popular family for real-time detection due to its speed.

## Image Segmentation

Classify every pixel in an image.

| Type | Description | Use Case |
|------|-------------|---------|
| Semantic Segmentation | Each pixel gets a class | Road vs. sky vs. building |
| Instance Segmentation | Distinguish individual instances | Person 1 vs. Person 2 |
| Panoptic Segmentation | Combines semantic + instance | Full scene understanding |

**Key models:** U-Net (medical imaging), Mask R-CNN, SegFormer, SAM (Segment Anything Model)

## Generative Computer Vision

### GANs for Images

- **StyleGAN:** High-quality, controllable face generation
- **Pix2Pix:** Image-to-image translation (sketch → photo)
- **CycleGAN:** Unpaired image translation (horse → zebra)

### Diffusion Models

Current state-of-the-art for image generation:
- **Stable Diffusion:** Open-weight text-to-image model
- **DALL-E 3:** OpenAI's text-to-image system
- **Midjourney:** Commercial text-to-image system

**How diffusion works:** Gradually add noise to images during training, then learn to reverse the process during inference.

## Data Augmentation for Vision

Artificially expand the training dataset by transforming existing images:

| Augmentation | Description |
|-------------|-------------|
| Horizontal Flip | Mirror the image left-right |
| Random Crop | Crop a random region |
| Color Jitter | Randomly change brightness, contrast, saturation |
| Rotation | Rotate by a random angle |
| Cutout / Random Erasing | Mask out a random patch |
| Mixup / CutMix | Blend two images together |

## Key Libraries and Tools

| Library | Purpose |
|---------|---------|
| OpenCV | Classical CV: image processing, filtering, feature matching |
| Pillow (PIL) | Basic image loading and manipulation in Python |
| torchvision | PyTorch datasets, transforms, and pre-trained CV models |
| timm | Large collection of pre-trained vision models |
| Ultralytics YOLO | Easy-to-use YOLO training and inference |
| Detectron2 | Facebook's object detection library |
| Albumentations | Fast, flexible image augmentation library |

## Evaluation Metrics

| Metric | Task |
|--------|------|
| Top-1 / Top-5 Accuracy | Image Classification |
| mAP (mean Average Precision) | Object Detection |
| IoU (Intersection over Union) | Detection & Segmentation |
| mIoU (mean IoU) | Semantic Segmentation |
| FID (Fréchet Inception Distance) | Generative Image Quality |

## Summary

Computer vision has been transformed by deep learning. CNNs remain a powerful tool, but Vision Transformers and diffusion models are redefining the state of the art. From classification and detection to generation, CV techniques underpin applications in healthcare, autonomous vehicles, security, and entertainment.

---

**Previous:** [Chapter 4 – Natural Language Processing](04-nlp.md)  
**Next:** [Chapter 6 – AI Tools and Frameworks](06-tools-and-frameworks.md)
