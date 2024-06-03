import google.generativeai as genai
from dotenv import load_dotenv
import pypdfium2 as pdfium
import tempfile
import os

load_dotenv("keys.env")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

prompt = """
You are a frugal and imaginative home cook. I will give you a flyer for a local grocery store that has a sale on ingredients.
You are planning to make a meal for your 4-person family. You want to make a meal that is both delicious and healthy.
Make a weekly recipe plan and shopping list based on the flyer.
"""

# convert PDF to image and upload them
flyer = []
with tempfile.TemporaryDirectory() as temp_dir:
    pdf = pdfium.PdfDocument("flyer.pdf")
    for page_num in range(len(pdf)):
        filename = os.path.join(temp_dir, f"flyer_p{page_num}.png")
        image = pdf.get_page(page_num).render().to_pil()
        image.save(os.path.join(temp_dir, filename))
        flyer.append(genai.upload_file(filename))
    response = model.generate_content([prompt] + flyer)
    print(response.text)
