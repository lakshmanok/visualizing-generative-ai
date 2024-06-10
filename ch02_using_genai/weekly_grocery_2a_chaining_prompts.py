# First part of chain. Goes from flyer -> list of items on sale

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import pypdfium2 as pdfium
import base64
import os, tempfile, io

MAX_PAGES_OF_FLYER = 2  # to reduce token usage

load_dotenv("keys.env")

model = ChatOpenAI(model="gpt-4o")
# model = GoogleGenerativeAI(model="gemini-1.5-flash")


class SaleItem(BaseModel):
    name: str = Field(description="item name")
    price: str = Field(description="price e.g. $3.97/lb")
    sale_type: str = Field(descrption="sale type e.g. None, member price, digital coupon, etc.")


sale_item_parser = JsonOutputParser(pydantic_object=SaleItem)
find_sale_items_prompt = """
Parse the weekly grocery ad brochure and find all the items on sale.
\n
""" + sale_item_parser.get_format_instructions()
print(find_sale_items_prompt)

# convert PDF to image and save the base64-contents
flyer_pages = []
with tempfile.TemporaryDirectory() as temp_dir:
    pdf = pdfium.PdfDocument("flyer.pdf")
    num_pages = min(MAX_PAGES_OF_FLYER, len(pdf))
    for page_num in range(num_pages):
        image = pdf.get_page(page_num).render().to_pil()
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="png")
        image_data = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        flyer_pages.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            }
        )

find_sale_items_message = HumanMessage(
    content=[
        {"type": "text", "text": find_sale_items_prompt},
    ] + flyer_pages
)
prompt = ChatPromptTemplate.from_messages([find_sale_items_message])
chain = prompt | model | sale_item_parser
response = chain.invoke(input={})
print("**Response**")
print(response)

"""
[{'name': 'USDA Choice Boneless Beef Chuck Roast', 'price': '$4.97/lb', 'sale_type': 'Member Price'}, {'name': 'Waterfront Bistro Jumbo Raw Shrimp', 'price': '$3.97/lb', 'sale_type': 'Member Price'}, {'name': 'Hass Avocado, Mango or Green Bell Pepper', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Foster Farms Boneless Skinless Chicken Breasts or Thighs', 'price': '$1.97/lb', 'sale_type': 'Member Price'}, {'name': 'Fresh Cut Fruit Bowl', 'price': 'Buy 2 Get 1 Free', 'sale_type': 'Member Price'}, {'name': 'Coca-Cola, 7UP or Pepsi Products', 'price': '3 for $12', 'sale_type': 'Member Price'}, {'name': 'Doritos Tortilla Chips', 'price': '$2.49 each', 'sale_type': 'Member Price when you buy 3'}, {'name': 'Tillamook Ice Cream', 'price': 'Buy 1 Get 1 Free', 'sale_type': 'Member Price'}, {'name': 'Starbucks Coffee', 'price': '$6.99', 'sale_type': 'Member Price when you buy 2'}, {'name': 'Coors Light, Corona, Firestone Walker, Modelo Spiked Agua Fresca', 'price': '$13.77', 'sale_type': 'Member Price'}, {'name': 'Fresh Atlantic Salmon Portion', 'price': '$3.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Fresh Baked Artisan French Bread', 'price': '$1.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Lucerne Milk', 'price': '$3.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Lucerne Shredded, Chunk, String or Fancy Cheese', 'price': '$5.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Nabisco Family Size Oreo, Snack Crackers, Chips Ahoy!', 'price': '$2.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Barilla Pasta', 'price': '$1.27', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Jennie-O 93% Lean Ground Turkey', 'price': '$3.99/lb', 'sale_type': 'Member Price'}, {'name': 'USDA Choice Beef for Taco Meat, Stew or Cube Steak', 'price': '$7.99/lb', 'sale_type': 'Member Price'}, {'name': 'Branding Iron Ranch Boneless Beef Ribeye Steak', 'price': '$12.99/lb', 'sale_type': 'Member Price'}, {'name': 'Open Nature 85% Lean Ground Beef', 'price': '$9.99/lb', 'sale_type': 'Member Price'}, {'name': 'Signature SELECT Marinated Tri-Tip Roast', 'price': '$7.99/lb', 'sale_type': 'Member Price'}, {'name': 'Oscar Mayer Selects Uncured Beef Franks', 'price': '$2.57', 'sale_type': 'Member Price'}, {'name': 'Hebrew National Beef Franks', 'price': '$3.99', 'sale_type': 'Member Price'}, {'name': 'Silva Sausage', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': "Miller's 'The Standard' or 'The Colossal' Beef Franks", 'price': '$5.49', 'sale_type': 'Member Price'}, {'name': 'Fresh Taylor Bay Oysters', 'price': '$12 for 12', 'sale_type': 'Member Price'}, {'name': 'Shrimp', 'price': '$5', 'sale_type': 'Member Price'}, {'name': 'Colossal Raw Prawns', 'price': '$9.99/lb', 'sale_type': 'Member Price'}, {'name': 'Gallo Salame', 'price': '$6.99/lb', 'sale_type': 'Member Price'}, {'name': "Sukhi's Gourmet Entrees", 'price': '$7.99', 'sale_type': 'Member Price'}, {'name': 'FAGE Greek Yogurt', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Lucerne Yogurt', 'price': '$2.49', 'sale_type': 'Member Price'}, {'name': 'Happy Egg Co. Free Range Eggs', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Clover Organic Milk', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Oatly Oat Milk', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Green Giant Frozen Vegetables', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Jimmy Dean Breakfast Bowls', 'price': '$2 for 7', 'sale_type': 'Member Price'}, {'name': "Red's Burritos", 'price': '$2 for 5', 'sale_type': 'Member Price'}, {'name': 'Nestle Drumstick Variety Pack or Haagen Dazs Ice Cream Bars', 'price': '$6.99', 'sale_type': 'Member Price'}, {'name': 'Signature SELECT 100% Apple Juice', 'price': '$2.55', 'sale_type': 'Member Price'}, {'name': "Kellogg's Family Size Cereal", 'price': '$2.49', 'sale_type': 'Member Price'}, {'name': 'Keebler Family Size Crackers', 'price': '$2.55', 'sale_type': 'Member Price'}, {'name': 'McCormick Grill Mates Seasoning or Grinders', 'price': '$0.99', 'sale_type': 'Member Price'}, {'name': "Sweet Baby Ray's BBQ or Dipping Sauce", 'price': '$3 for 5', 'sale_type': 'Member Price'}, {'name': 'BodyArmor', 'price': '$5 for 5', 'sale_type': 'Member Price'}, {'name': 'Ziploc Slider Bags', 'price': '$6.99', 'sale_type': 'Member Price'}, {'name': 'Guinness, Heineken, Dos Equis, Modelo Chelada, Stella Artois', 'price': '$16.99', 'sale_type': 'Member Price'}, {'name': 'Blue Moon', 'price': '$18.99', 'sale_type': 'Member Price'}, {'name': 'Truly, Smirnoff, Cayman Jack', 'price': '$19.99', 'sale_type': 'Member Price'}, {'name': 'Coors Light, Miller Lite, Bud Light', 'price': '$21.99', 'sale_type': 'Member Price'}, {'name': 'Jim Beam or Skky', 'price': '$11.99', 'sale_type': 'Member Price'}, {'name': "Bulleit or Maker's Mark", 'price': '$24.99', 'sale_type': 'Member Price'}, {'name': 'Hess, Edna Valley or Apothic', 'price': '$8.99', 'sale_type': 'Member Price'}, {'name': 'Bread & Butter, Santa Barbara Cellars or Gerard Bertrand', 'price': '$12.99', 'sale_type': 'Member Price'}]
"""