import os
import cv2
import sys
import PIL
import torch
import pickle
import appdirs
import numpy as np
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
from pathlib import Path
from loguru import logger
from collections import Counter
from types import SimpleNamespace
from torchvision import transforms
from aircanvas.HandTrackingModule import sketch
from dino.ImageRetrieval_class import ImageRetrieval
from PIL import Image, ImageOps, ImageDraw, ImageFont
from transformers import BlipForConditionalGeneration, AutoProcessor

# Device 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP 모델 및 Processor 로드
model = BlipForConditionalGeneration.from_pretrained("ybelkada/blip-image-captioning-base-football-finetuned").to(device)
processor = AutoProcessor.from_pretrained("ybelkada/blip-image-captioning-base-football-finetuned")

# 체크포인트 로드
checkpoint_path = "/content/drive/MyDrive/Last_Dance/Sketch2Image_Retrieval/checkpoints"
checkpoint = torch.load(os.path.join(checkpoint_path, "blip_fine_tuning.pth"))  
model.load_state_dict(checkpoint['model_state_dict'])

# BLIP 모델을 사용하여 캡션 생성
image_path = "/content/drive/MyDrive/img/flb.png"  
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

out = model.generate(**inputs, max_length=128)

generated_caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption: ", generated_caption)

# Diffusion 모델 실행
caption_for_diffusion_prompt = generated_caption  
os.system(f'python img2img-turbo/src/inference_paired.py --model_name "sketch_to_image_stochastic" '
          f'--input_image "{image_path}" --gamma 0.4 '
          f'--prompt "{caption_for_diffusion_prompt}" '
          f'--output_dir "diffusion_output"')


# DINOv2 모델을 사용하여 이미지 검색
args = SimpleNamespace(**{
    'query': "diffusion_output/flb.png",  
    'database': "/content/drive/MyDrive/Last_Dance/Sketch2Image_Retrieval/gallery_data/gallery", 
    'output_root': "/content/drive/MyDrive/Last_Dance/Sketch2Image_Retrieval/final_output",
    'model_size': "base",  
    'num': 5,  
    'size': 224,
    'verbose': True,
    'disable_cache': False,
    'model_path': None
})

retriever = ImageRetrieval(args)
retriever.run(args)
