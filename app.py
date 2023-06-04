import cv2
import numpy as np
from telegram import Bot, Update
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackContext
from telegram.ext import Application, ApplicationBuilder
from io import BytesIO
import os 
import subprocess
import shutil

def run_inference(image_path):
    
    if os.path.isdir("results"):
        shutil.rmtree("results")
    
    command = f"python3 GFPGAN/inference_gfpgan.py -i {image_path} -o results -v 1.3 -s 2 --bg_upsampler realesrgan"
    os.system(command)
    #output = subprocess.check_output(command, shell=False, text=False)    
    print("Inference finished")


async def start(update: Update, context: CallbackContext) -> None:
      await update.message.reply_text('Hi! Send me an image and I will increase the resolution.')
      

async def handle_message(update: Update, context: CallbackContext) -> None:
    file = await context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(await file.download_as_bytearray())
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), 1)

    cv2.imwrite("sample.jpeg", img)

    await update.message.reply_text('This may take few seconds...')
    run_inference("sample.jpeg")
    
    with open(r'results\restored_imgs\sample.jpeg', 'rb') as photo_file:
        await context.bot.send_photo(chat_id=update.message.chat_id, photo=photo_file)

async def handle_text(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Please send an image for resolution enhancement.')


nakish_bot = Bot(token='6106659941:AAFj7NpuVdIMLqVEyVHlwqAUBkpW6L_iQ0Q')
application = Application.builder().bot(bot=nakish_bot).build()
#application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.PHOTO, handle_message))
application.add_handler(MessageHandler(filters.TEXT, start))
application.run_polling(1.0)
