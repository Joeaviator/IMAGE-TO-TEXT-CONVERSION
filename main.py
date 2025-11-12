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
def gcfc(path):
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache=json.load(f)
            return cache.get(path)
        except json.JSONDecodeError:
            return None
    return None

#Function to generate caption
def generate_caption(image,metries=3):
    for attempt in range(metries):
        try:
            print(f"{Fore.RED}Generating caption.... (Atttempt {attempt+1}/{metries}{Style.RESET_ALL})")
            pixel_values=feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
            output_ids=model.generate(pixel_values, max_length=64, num_beams=4)
            caption=tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            print(f"{Fore.YELLOW} Error Generating aption : {e}{Style.RESET_ALL}")
            time.sleep(1)#-Wait before it tries again
    raise Exception("❌Failed to generate caption after multiple retries❌")
#Truncate-cut the word into specfic words
def truncate_text(text, word_limit):
    #remove unwanted spaces
    words=text.strip().split()
    return " ".join(words[:word_limit])
#Function to save caption
def save_to_file(image_path, caption):
    output="caption.txt"
    with open(output, "a", encoding="utf-8") as f:
        f.write(f"\n Image:{image_path}\nCaption:{caption}")
    print(f"{Fore.GREEN}Caption saved")
def print_menu():
    print(f"""{Style.BRIGHT}
          {Fore.GREEN}=============================== Image Caption==============================
          Select Output type:
          1. Caption(5 words)
          2. Description(30 words)
          3. Summary(50 words)
          4.Exit)
          =============================================
          """)
def main():
    image_path = input(f"{Fore.BLUE}Enter the Path of the Image (e.g., test.jpg): {Style.RESET_ALL}")
    if not os.path.exists(image_path):
        print(f"{Fore.RED}❌File Not Found: {image_path}❌")
        return
    try:
        image=Image.open(image_path)
    except Exception as e:
        print(f"{Fore.RED}Failed to open image: {e}")
        return
    #Check cache
    cached_caption=gcfc(image_path)
    if cached_caption:
        print(f"{Fore.LIGHTBLACK_EX}Caption: {cached_caption}")
        caption=cached_caption
    else:
        caption=generate_caption(image)
        save_caption(image_path, caption)
    print(f"{Fore.Yellow} Basic Caption: {Style.BRIGHT}{caption}\n")
    while True:
        print_menu()
        choice=input(f"{Fore.GREEN} EEnter you choice(1-4):{Style.RESET_ALL}")
        if choice=="1":
            result=truncate_text(caption,5)
            print(f"{Fore.GREEN}Caption(5 words): {result}")
            save_caption=(image_path, result)
        elif choice=="2":
            result=truncate_text(caption,30)
            print(f"{Fore.GREEN}Description(30 words): {result}")
            save_caption=(image_path, result)
        elif choice=="3":
            result=truncate_text(caption,50)
            print(f"{Fore.GREEN}Summary(50 words): {result}")
            save_caption=(image_path, result)
        elif choice=="4":
            print(f"{Fore.CYAN}Good Bye! {Style.RESET_ALL}")
            break
        else:
            print(f"{Fore.RED} Invalid option.... Please try again{Style.RESET_ALL}")
if __name__=="__main__":
    main()
          
        


