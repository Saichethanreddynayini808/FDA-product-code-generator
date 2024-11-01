from flask import Flask, request, jsonify, render_template
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(
    api_key="your api key here"  # replace with your actual key
)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_fda_code', methods=['POST'])
def get_fda_code():
    data = request.get_json()
    images_base64 = data['images_base64']  # Array of base64-encoded images
    include_subclass = data.get('include_subclass', True)  # Include subclass by default
    show_descriptions = data.get('show_descriptions', False)  # Checkbox for descriptions
    show_explanations = data.get('show_explanations', False)  # Checkbox for explanations

    # Process images and get detailed FDA codes and explanations
    fda_code = get_FDA_code_from_images_base64(images_base64, include_subclass, show_descriptions, show_explanations)

    return jsonify(fda_code)

def get_FDA_code_from_images_base64(images_base64, include_subclass=True, show_descriptions=False, show_explanations=False):
    # Combine all images into one request and analyze them as a group
    image_data_list = [f"data:image/jpeg;base64,{image_base64}" for image_base64 in images_base64]

    # Load FDA data from CSVs
    industry_csv_file_path = "C:/Users/cheth/OneDrive/Desktop/capstone/industry_api_result_data.csv"
    subclass_csv_file_path = "subclassforfood.csv"
    pic_csv_file_path = "picmodified.csv"

    # Read CSV files into dataframes
    industry_csv = pd.read_csv(industry_csv_file_path)
    subclass_csv = pd.read_csv(subclass_csv_file_path)
    pic_csv = pd.read_csv(pic_csv_file_path)

    # Convert CSVs to JSON for use in prompts
    industry_data = industry_csv.to_json(orient='records')
    subclass_data = subclass_csv.to_json(orient='records')
    pic_data = pic_csv.to_json(orient='records')

    # Step 1: Get the Industry Code
    response_industry = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Can you provide the industry code of this product based on these images? Use the following FDA industry data: {industry_data}. Output only the industry code."
                    },
                    *[{"type": "image_url", "image_url": {"url": img_data}} for img_data in image_data_list]
                ],
            }
        ],
        max_tokens=100
    )
    industry_code = response_industry.choices[0].message.content.strip()

    # Get Industry Description and Explanation (if needed)
    industry_description = ""
    industry_explanation = ""
    if show_descriptions:
        response_industry_description = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Provide the INDDESC of the '{industry_code}' from the industry data: {industry_data}. Just the inddesc, nothing else."
                        }
                    ],
                }
            ],
            max_tokens=200
        )
        industry_description = response_industry_description.choices[0].message.content.strip()

    if show_explanations:
        response_industry_explanation = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Can you explain what the industry code '{industry_code}' means? Provide a detailed explanation of what this industry code represents, using context from the FDA industry data."
                        }
                    ],
                }
            ],
            max_tokens=300
        )
        industry_explanation = response_industry_explanation.choices[0].message.content.strip()

    # Step 2: Get the Class Code and Explanation
    class_csv_file_path = f"class_data_industry_{industry_code}.csv"
    class_csv = pd.read_csv(class_csv_file_path)
    class_data = class_csv.to_json(orient='records')

    response_class = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Can you provide the class code of this product based on these images? Use the following FDA class data: {class_data}. Output only the class code."
                    },
                    *[{"type": "image_url", "image_url": {"url": img_data}} for img_data in image_data_list]
                ],
            }
        ],
        max_tokens=100
    )
    class_code = response_class.choices[0].message.content.strip()

    class_description = ""
    class_explanation = ""
    if show_descriptions:
        response_class_description = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Provide the CLASSDESC for the '{class_code}' from the class dataset: {class_data}. Just give the classdesc, nothing else."
                        }
                    ],
                }
            ],
            max_tokens=200
        )
        class_description = response_class_description.choices[0].message.content.strip()

    if show_explanations:
        response_class_explanation = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Can you explain what the class code '{class_code}' means? Provide a detailed explanation of what this class code represents."
                        }
                    ],
                }
            ],
            max_tokens=300
        )
        class_explanation = response_class_explanation.choices[0].message.content.strip()

    # Step 3: Get Subclass Code and Explanation (if included)
    subclass_code = "No subclass provided"
    subclass_description = ""
    subclass_explanation = ""
    if include_subclass:
        response_subclass = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Can you provide the subclass code of this product based on these images? Use the following FDA subclass data: {subclass_data}. Output only the subclass code."
                        },
                        *[{"type": "image_url", "image_url": {"url": img_data}} for img_data in image_data_list]
                    ],
                }
            ],
            max_tokens=100
        )
        subclass_code = response_subclass.choices[0].message.content.strip()

        if show_descriptions:
            response_subclass_description = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Provide the SUBCLASSDESC for the '{subclass_code}' from the dataset {subclass_data}. Just give the SUBCLASSDESC, nothing else."
                            }
                        ],
                    }
                ],
                max_tokens=200
            )
            subclass_description = response_subclass_description.choices[0].message.content.strip()

        if show_explanations:
            response_subclass_explanation = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Can you explain what the subclass code '{subclass_code}' means? Provide a detailed explanation of what this subclass code represents."
                            }
                        ],
                    }
                ],
                max_tokens=300
            )
            subclass_explanation = response_subclass_explanation.choices[0].message.content.strip()

    # Step 4: Get PIC Code and Explanation
    response_PIC = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Can you provide the PIC code of this product based on these images? Use the following FDA PIC data: {pic_data}. Output only the PIC code."
                    },
                    *[{"type": "image_url", "image_url": {"url": img_data}} for img_data in image_data_list]
                ],
            }
        ],
        max_tokens=100
    )
    pic_code = response_PIC.choices[0].message.content.strip()

    pic_description = ""
    pic_explanation = ""
    if show_descriptions:
        response_PIC_description = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Provide the PICDESC for the '{pic_code}' from the pic dataset: {pic_data}. Just give the PICDESC, nothing else."
                        }
                    ],
                }
            ],
            max_tokens=200
        )
        pic_description = response_PIC_description.choices[0].message.content.strip()

    if show_explanations:
        response_PIC_explanation = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Can you explain what the PIC code '{pic_code}' means? Provide a detailed explanation of what this PIC code represents."
                        }
                    ],
                }
            ],
            max_tokens=300
        )
        pic_explanation = response_PIC_explanation.choices[0].message.content.strip()

    # Step 5: Get the Product Code and Explanation
    product_csv_file_path = f"product_data_industry_{industry_code}.csv"
    product_csv = pd.read_csv(product_csv_file_path)
    product_data = product_csv.to_json(orient='records')

    response_product = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Can you provide the product code of this product based on these images? Use the following FDA product data: {product_data}. Output only the product code, just the  code nothing else."
                    },
                    *[{"type": "image_url", "image_url": {"url": img_data}} for img_data in image_data_list]
                ],
            }
        ],
        max_tokens=100
    )
    product_code = response_product.choices[0].message.content.strip()

    product_description = ""
    product_explanation = ""
    if show_descriptions:
        response_product_description = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Provide the PRODDESC for the '{product_code}' from the product dataset:give me all the product which matches {product_data}. Just give the PRODDESC, nothing else."
                        }
                    ],
                }
            ],
            max_tokens=200
        )
        product_description = response_product_description.choices[0].message.content.strip()

    if show_explanations:
        response_product_explanation = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Can you explain what the product code '{product_code}' means? Provide a detailed explanation of what this product code represents."
                        }
                    ],
                }
            ],
            max_tokens=300
        )
        product_explanation = response_product_explanation.choices[0].message.content.strip()

    # Combine all collected data into the final object
    fda_code_data = {
        'industry': industry_code,
        'industry_description': industry_description,
        'industry_explanation': industry_explanation,
        'class': class_code,
        'class_description': class_description,
        'class_explanation': class_explanation,
        'subclass': subclass_code,
        'subclass_description': subclass_description,
        'subclass_explanation': subclass_explanation,
        'PIC': pic_code,
        'PIC_description': pic_description,
        'PIC_explanation': pic_explanation,
        'product': product_code,
        'product_description': product_description,
        'product_explanation': product_explanation,
        'fda_code': f"{industry_code} {class_code} {subclass_code} {pic_code} {product_code}"
    }

    return fda_code_data

if __name__ == '__main__':
    app.run(debug=True)
