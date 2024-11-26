
FDA Product Code Identifier
A web application that allows users to upload product label images to identify FDA product codes. The application uses OpenAI's API to analyze the images and retrieve detailed FDA classifications, including industry, class, subclass, PIC, and product information. This tool also provides options to display detailed descriptions and explanations for each code category.
________________________________________
Project Overview
This web app allows users to upload product images, which are analyzed to produce an FDA product code along with relevant descriptions and explanations. The app utilizes Flask as the backend framework, OpenAI for data processing, and HTML/CSS with JavaScript for the frontend.
Features
•	Product Code Identification: Extracts and displays FDA product codes from uploaded images.
•	Detailed Descriptions and Explanations: Optionally display explanations and descriptions for each code component, such as industry, class, subclass, PIC, and product.
•	Dynamic UI: Updates output fields and displays descriptions/explanations only when requested.
•	User-Friendly Interface: A clean, responsive layout with a visually engaging background and straightforward controls.


Technologies Used
•	Backend: Flask,python
•	API: OpenAI API for image analysis and FDA code identification
•	Frontend: HTML, CSS (Bootstrap), JavaScript
•	Other: Pandas for CSV data handling in Python

Installation
1.	Download the zip file:
or
bash
Copy code
‘git clone https://github.com/Saichethanreddynayini808/FDA-product-code-generator.git’
      2.Install dependencies:
bash
Copy code
pip install -r requirements.txt
     3.API Key Configuration:
Copy the openai key and paste it in app.py 
     Copy code
     client = OpenAI(
    api_key="your api key here"  # replace with your actual key
)

Run the Flask application
2.	Open the application(app.py):
o	Go to http://127.0.0.1:5000 in your web browser to access the app.
Usage
1.	Upload Images:
o	Click the "Upload" button to select product label images. Multiple images can be uploaded.
2.	Choose Options:
o	Subclass: Enable or disable subclass detection.
o	Show Code Descriptions: Check this option to display detailed descriptions of each code category.
o	Show Code Explanations: Check this option to display explanations of each code category.
3.	View Results:
o	Once processed, the FDA product code will be displayed along with optional descriptions and explanations if selected.
4.	Output Display:
o	The main FDA Product Code section will show only the codes, while individual sections for each code (industry, class, etc.) will show both codes and descriptions or explanations if enabled.
File Structure
plaintext

app.py/
│
├── static/
│   ├── styles.css             # Custom CSS for the app
│   └── products.jpg           # Background image
│
├── templates/
│   └── index.html             # Main HTML template for the frontend
│
├── app.py                     # Flask backend application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


