# Second part of chain. Goes from list of items on sale --> recipes

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import typing

# Output of 2a
sale_items = """
[{'name': 'USDA Choice Boneless Beef Chuck Roast', 'price': '$4.97/lb', 'sale_type': 'Member Price'}, {'name': 'Waterfront Bistro Jumbo Raw Shrimp', 'price': '$3.97/lb', 'sale_type': 'Member Price'}, {'name': 'Hass Avocado, Mango or Green Bell Pepper', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Foster Farms Boneless Skinless Chicken Breasts or Thighs', 'price': '$1.97/lb', 'sale_type': 'Member Price'}, {'name': 'Fresh Cut Fruit Bowl', 'price': 'Buy 2 Get 1 Free', 'sale_type': 'Member Price'}, {'name': 'Coca-Cola, 7UP or Pepsi Products', 'price': '3 for $12', 'sale_type': 'Member Price'}, {'name': 'Doritos Tortilla Chips', 'price': '$2.49 each', 'sale_type': 'Member Price when you buy 3'}, {'name': 'Tillamook Ice Cream', 'price': 'Buy 1 Get 1 Free', 'sale_type': 'Member Price'}, {'name': 'Starbucks Coffee', 'price': '$6.99', 'sale_type': 'Member Price when you buy 2'}, {'name': 'Coors Light, Corona, Firestone Walker, Modelo Spiked Agua Fresca', 'price': '$13.77', 'sale_type': 'Member Price'}, {'name': 'Fresh Atlantic Salmon Portion', 'price': '$3.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Fresh Baked Artisan French Bread', 'price': '$1.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Lucerne Milk', 'price': '$3.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Lucerne Shredded, Chunk, String or Fancy Cheese', 'price': '$5.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Nabisco Family Size Oreo, Snack Crackers, Chips Ahoy!', 'price': '$2.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Barilla Pasta', 'price': '$1.27', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Jennie-O 93% Lean Ground Turkey', 'price': '$3.99/lb', 'sale_type': 'Member Price'}, {'name': 'USDA Choice Beef for Taco Meat, Stew or Cube Steak', 'price': '$7.99/lb', 'sale_type': 'Member Price'}, {'name': 'Branding Iron Ranch Boneless Beef Ribeye Steak', 'price': '$12.99/lb', 'sale_type': 'Member Price'}, {'name': 'Open Nature 85% Lean Ground Beef', 'price': '$9.99/lb', 'sale_type': 'Member Price'}, {'name': 'Signature SELECT Marinated Tri-Tip Roast', 'price': '$7.99/lb', 'sale_type': 'Member Price'}, {'name': 'Oscar Mayer Selects Uncured Beef Franks', 'price': '$2.57', 'sale_type': 'Member Price'}, {'name': 'Hebrew National Beef Franks', 'price': '$3.99', 'sale_type': 'Member Price'}, {'name': 'Silva Sausage', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': "Miller's 'The Standard' or 'The Colossal' Beef Franks", 'price': '$5.49', 'sale_type': 'Member Price'}, {'name': 'Fresh Taylor Bay Oysters', 'price': '$12 for 12', 'sale_type': 'Member Price'}, {'name': 'Shrimp', 'price': '$5', 'sale_type': 'Member Price'}, {'name': 'Colossal Raw Prawns', 'price': '$9.99/lb', 'sale_type': 'Member Price'}, {'name': 'Gallo Salame', 'price': '$6.99/lb', 'sale_type': 'Member Price'}, {'name': "Sukhi's Gourmet Entrees", 'price': '$7.99', 'sale_type': 'Member Price'}, {'name': 'FAGE Greek Yogurt', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Lucerne Yogurt', 'price': '$2.49', 'sale_type': 'Member Price'}, {'name': 'Happy Egg Co. Free Range Eggs', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Clover Organic Milk', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Oatly Oat Milk', 'price': '$4.99', 'sale_type': 'Member Price'}, {'name': 'Green Giant Frozen Vegetables', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Jimmy Dean Breakfast Bowls', 'price': '$2 for 7', 'sale_type': 'Member Price'}, {'name': "Red's Burritos", 'price': '$2 for 5', 'sale_type': 'Member Price'}, {'name': 'Nestle Drumstick Variety Pack or Haagen Dazs Ice Cream Bars', 'price': '$6.99', 'sale_type': 'Member Price'}, {'name': 'Signature SELECT 100% Apple Juice', 'price': '$2.55', 'sale_type': 'Member Price'}, {'name': "Kellogg's Family Size Cereal", 'price': '$2.49', 'sale_type': 'Member Price'}, {'name': 'Keebler Family Size Crackers', 'price': '$2.55', 'sale_type': 'Member Price'}, {'name': 'McCormick Grill Mates Seasoning or Grinders', 'price': '$0.99', 'sale_type': 'Member Price'}, {'name': "Sweet Baby Ray's BBQ or Dipping Sauce", 'price': '$3 for 5', 'sale_type': 'Member Price'}, {'name': 'BodyArmor', 'price': '$5 for 5', 'sale_type': 'Member Price'}, {'name': 'Ziploc Slider Bags', 'price': '$6.99', 'sale_type': 'Member Price'}, {'name': 'Guinness, Heineken, Dos Equis, Modelo Chelada, Stella Artois', 'price': '$16.99', 'sale_type': 'Member Price'}, {'name': 'Blue Moon', 'price': '$18.99', 'sale_type': 'Member Price'}, {'name': 'Truly, Smirnoff, Cayman Jack', 'price': '$19.99', 'sale_type': 'Member Price'}, {'name': 'Coors Light, Miller Lite, Bud Light', 'price': '$21.99', 'sale_type': 'Member Price'}, {'name': 'Jim Beam or Skky', 'price': '$11.99', 'sale_type': 'Member Price'}, {'name': "Bulleit or Maker's Mark", 'price': '$24.99', 'sale_type': 'Member Price'}, {'name': 'Hess, Edna Valley or Apothic', 'price': '$8.99', 'sale_type': 'Member Price'}, {'name': 'Bread & Butter, Santa Barbara Cellars or Gerard Bertrand', 'price': '$12.99', 'sale_type': 'Member Price'}]
"""

load_dotenv("keys.env")

model = ChatOpenAI(model="gpt-4o")
# model = GoogleGenerativeAI(model="gemini-1.5-flash")


class RecipeItem(BaseModel):
    name: str = Field(description="item name")
    price: str = Field(description="price e.g. $3.97/lb if known")
    sale_type: str = Field(descrption="sale type e.g. None, member price, digital coupon, etc.")


class Recipe(BaseModel):
    title: str = Field(description="Recipe title")
    ingredients: typing.List[RecipeItem] = Field(description="list of ingredients")
    instructions: typing.List[str] = Field(descrption="steps to take")


recipe_parser = JsonOutputParser(pydantic_object=Recipe)
create_recipes_prompt = f"""
You are a frugal and imaginative home cook. 
I will give you a JSON list of ingredients that are on sale at the local grocery store.
Plan out 5 delicious and healthy dinner recipes for a family of four, prioritizing the use of items that are on sale.

**Sale Items**
""" + f"{sale_items}\n\n" + recipe_parser.get_format_instructions()
print(create_recipes_prompt)

create_recipes_message = HumanMessage(
    content=[
        {"type": "text", "text": create_recipes_prompt},
    ]
)
prompt = ChatPromptTemplate.from_messages([create_recipes_message])
chain = prompt | model | recipe_parser
response = chain.invoke(input={})
print("**Response**")
print(response)

"""
[{'title': 'Grilled Chicken and Avocado Salad', 'ingredients': [{'name': 'Foster Farms Boneless Skinless Chicken Breasts or Thighs', 'price': '$1.97/lb', 'sale_type': 'Member Price'}, {'name': 'Hass Avocado', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Green Bell Pepper', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Fresh Baked Artisan French Bread', 'price': '$1.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Lucerne Shredded Cheese', 'price': '$5.97', 'sale_type': 'Exclusive Digital Coupon'}], 'instructions': ['Grill the chicken breasts until fully cooked and let them rest.', 'Dice the avocados and bell peppers.', 'In a large bowl, combine the diced avocados, bell peppers, and shredded cheese.', 'Slice the grilled chicken and add it to the salad.', 'Serve with slices of fresh baked artisan French bread.']}, {'title': 'Shrimp and Mango Stir-fry', 'ingredients': [{'name': 'Waterfront Bistro Jumbo Raw Shrimp', 'price': '$3.97/lb', 'sale_type': 'Member Price'}, {'name': 'Mango', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Green Bell Pepper', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Barilla Pasta', 'price': '$1.27', 'sale_type': 'Exclusive Digital Coupon'}], 'instructions': ['Cook the Barilla pasta according to the package instructions and set aside.', 'In a large pan, sauté the shrimp until they turn pink.', 'Add diced mangoes and bell peppers to the pan and stir-fry for a few minutes.', 'Combine the stir-fried shrimp and vegetables with the cooked pasta.', 'Serve hot.']}, {'title': 'Beef Chuck Roast with Veggies', 'ingredients': [{'name': 'USDA Choice Boneless Beef Chuck Roast', 'price': '$4.97/lb', 'sale_type': 'Member Price'}, {'name': 'Green Giant Frozen Vegetables', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Lucerne Shredded Cheese', 'price': '$5.97', 'sale_type': 'Exclusive Digital Coupon'}], 'instructions': ['Preheat the oven to 350°F.', 'Season the beef chuck roast and place it in a roasting pan.', 'Roast the beef in the oven for about 2-3 hours or until tender.', 'Steam the Green Giant frozen vegetables.', 'Serve the roast beef with steamed vegetables and a sprinkle of shredded cheese on top.']}, {'title': 'Salmon and Avocado Tacos', 'ingredients': [{'name': 'Fresh Atlantic Salmon Portion', 'price': '$3.97', 'sale_type': 'Exclusive Digital Coupon'}, {'name': 'Hass Avocado', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Green Bell Pepper', 'price': '10 for $10', 'sale_type': 'Member Price'}, {'name': 'Fresh Baked Artisan French Bread', 'price': '$1.97', 'sale_type': 'Exclusive Digital Coupon'}], 'instructions': ['Cook the salmon portions in a pan or on the grill until done.', 'Dice the avocados and bell peppers.', 'Flake the cooked salmon into bite-sized pieces.', 'Fill taco shells with salmon, avocado, and bell pepper.', 'Serve with a side of fresh baked artisan French bread.']}, {'title': 'Turkey and Veggie Pasta', 'ingredients': [{'name': 'Jennie-O 93% Lean Ground Turkey', 'price': '$3.99/lb', 'sale_type': 'Member Price'}, {'name': 'Green Giant Frozen Vegetables', 'price': '$4 for 5', 'sale_type': 'Member Price'}, {'name': 'Barilla Pasta', 'price': '$1.27', 'sale_type': 'Exclusive Digital Coupon'}], 'instructions': ['Cook the Barilla pasta according to the package instructions and set aside.', 'In a large pan, cook the ground turkey until browned.', 'Add the Green Giant frozen vegetables to the pan and cook until heated through.', 'Combine the cooked pasta with the turkey and vegetables.', 'Serve hot.']}]
"""
