from PIL import Image

im = Image.open('Background.png')
im = im.resize((800,600))
im.putalpha(100)
im.save('Background.png')