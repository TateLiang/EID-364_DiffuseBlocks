import tkinter as tk
from tkinter import ttk
import customtkinter as ctk 

import cv2
import time
import threading

from PIL import ImageTk, Image
from authtoken import auth_token

import torch
import gc
from torch.cuda.amp import autocast
from diffusers import StableDiffusionImg2ImgPipeline

torch.cuda.empty_cache()
gc.collect()

# UI
app = tk.Tk()
#app.geometry("1054x582")
app.state('zoomed')
app.title("Architectural Model Generation") 
app.grid_columnconfigure(0, weight = 1)
app.grid_columnconfigure(1, weight = 1)
app.grid_rowconfigure(0, weight = 1)
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)
progress = ttk.Progressbar(app, orient=tk.HORIZONTAL,length=200,mode="determinate",takefocus=True,maximum=100)
progress.place(x=542, y=-25)

cb1 = tk.IntVar() 
checkbox1 = ttk.Checkbutton(app, text = "Repeat", variable = cb1, onvalue = 1, offvalue = 0, width = 10)
checkbox1.place(x=542, y=70)

imageFrame = ctk.CTkFrame(app, width=678, height=678, bg_color="red")
imageFrame.grid(row=0, column=0)
lmain = ctk.CTkLabel(master=imageFrame, text="", height=300)
lmain.grid(row=0, column=0)

lphotos = ttk.Frame(app, width=700, height=700)
lphotos.grid(row=0, column=1)
lphotos.grid_propagate(False)
l1 = ctk.CTkLabel(master=lphotos, text="", height=350, width=350,bg_color="white")
l1.grid(row=0, column=0, sticky='nsew')
l1.grid_propagate(False)
l2 = ctk.CTkLabel(master=lphotos, text="", height=350, width=350,bg_color="white")
l2.grid(row=1, column=0, sticky='nsew')
l2.grid_propagate(False)
l3 = ctk.CTkLabel(master=lphotos, text="", height=350, width=350,bg_color="white")
l3.grid(row=0, column=1, sticky='nsew')
l3.grid_propagate(False)
l4 = ctk.CTkLabel(master=lphotos, text="", height=350, width=350,bg_color="white")
l4.grid(row=1, column=1, sticky='nsew')
l4.grid_propagate(False)

current_grid_index=0

#webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cam.isOpened():
    print("opened")
    
def show_frame():
    _, frame = cam.read()
    #frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 

    progress.step()            
    app.update()

#Stable Diffusion Pipeline
model_id = "stabilityai/stable-diffusion-2-base"
device = "cuda"

#IMAGE-TO-IMAGE
pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe2.to(device)
gc.collect()

def generateI2I():
    global current_grid_index

    #Capture Image
    #Take 5 Images & use the last because for some reason the first couple are sometimes black
    images = []
    for i in range(0,5):
        result, image = cam.read()
        if not result:
            print("Image Capture Error")
            return
        
        images.append(image)
        time.sleep(0.1)
    
    cv2.imwrite("Images\\CapturedImage.png", images[-1])
    color_converted = cv2.cvtColor(images[-1], cv2.COLOR_BGR2RGB)
    captured_image = Image.fromarray(color_converted)

    #Generate Image
    with autocast():
        with torch.no_grad():
            result_image = pipe2(prompt.get(), image=captured_image, guidance_scale=1.5).images[0]
            
    result_image.save('Images\\GeneratedImage.png')

    #resizing & converting image
    basewidth = 350
    wpercent = (basewidth/float(result_image.size[0]))
    hsize = int((float(result_image.size[1])*float(wpercent)))
    result_image = result_image.resize((basewidth,hsize), Image.Resampling.LANCZOS)
    result_image_tk1 = ImageTk.PhotoImage(result_image)

    if current_grid_index==0:
        l1.configure(image=result_image_tk1)
        current_grid_index = 1
    elif current_grid_index==1:
        l2.configure(image=result_image_tk1)
        current_grid_index = 2
    elif current_grid_index==2:
        l3.configure(image=result_image_tk1)
        current_grid_index = 3
    else:
        l4.configure(image=result_image_tk1)
        current_grid_index = 0

    finish_generation()
    torch.cuda.empty_cache()

def finish_generation():
    #restore button
    if cb1.get():
        generateI2I()
    else:
        trigger2.place(x=542, y=10) 
        progress.place(x=542, y=-25)
    

#async to disable button & allow loading bar
def generateI2I_async():
    trigger2.place(x=542, y=-40) 
    progress.place(x=542, y=15)
    threading.Thread(target=generateI2I).start()



#buttons
trigger2 = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="black", command=generateI2I_async) 
trigger2.configure(text="Generate") 
trigger2.place(x=542, y=10) 

show_frame()
app.mainloop()

'''
#TEXT-TO-IMAGE
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 
gc.collect()

def generate():

    with autocast(): 
        with torch.no_grad():
            image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img) 
    torch.cuda.empty_cache()

trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="black", command=generate) 
trigger.configure(text="Generate Txt2Img") 
trigger.place(x=522, y=10) 
'''

'''
#WEBCAM DEBUGGING
result, image = cam.read()
if result:
    cv2.imshow("img", image)
    cv2.imwrite("Images\\CapturedImage.png", image)
    cv2.waitKey(0)
    cv2.destroyWindow("img")
else:
    print("imgcaptureerror")
'''