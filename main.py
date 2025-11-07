from colorama import Fore, Style, init 
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import json
import time
import os

#Initialize coloroma
init(autoreset=True)
#Load the model
model_name="nlpconnect/vit-gpt2-image-captioning"
print(f"{Fore.LIGHTCYAN_EX}Loading Model⏳⏰  '{model_name}'.....{Style.RESET_ALL}")

#Setup the model
model=VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor=ViTFeatureExtractor.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

#Where is the model going to run
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"{Fore.GREEN}☑️Model Loaded successfully{Style.RESET_ALL}")

#Store data in cache-avoid repeated processing
CACHE_FILE="cache.json"
def save_caption(path, caption):
    cache={}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE,"r") as f:
                cache=json.load(f)
        except json.JSONDecodeError:
            cache={}
    cache[path]=caption
    with open(CACHE_FILE,"w") as f:
        json.dump(cache, f, indent=2)

#Function to get the caption from cache
def fcfc(path):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE):
                cache=json.load(f)
            return cache.get(path)
        except json.JSONDecodeError:
            return None
    return None

#Function to generate caption
def generate_captain(image,metries=3):
    for attempt in range(metries):
        try:
            print(f"{Fore.RED}Generating caption.... (Atttempt {attempt+1}/{metries}{Style.RESET_ALL})")
            pixel_values=feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
            output_ids=model.generate(pixel_values, max_length=, num_beams=4)
            caption=tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            print(f"{Fore.YELLOW} Error Generating aption : {e}{Style.RESET_ALL}")
            time.sleep(1)#-Wait before it tries again
    raise Exception("❌Failed to generate caption after multiple retries❌")


