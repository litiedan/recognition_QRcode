from pyzbar.pyzbar import decode
from PIL import Image
image = "/home/mr/cam_file/recognition_QRcode/rikirobot101.png"
img = Image.open(image)
barcodes = decode(img)
for barcode in barcodes:
    url = barcode.data.decode("utf-8")
    print(url)
