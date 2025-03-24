from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
from ultralytics import YOLO  # Correct import for YOLOv8
from PIL import Image
import requests
import os

# Load the trained YOLOv8 model
yolo_model = YOLO('best.pt')  # Ensure the path to the model is correct

app = Flask(__name__, template_folder='templates')

# Path to save uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tell Flask to serve files from the 'uploads' directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to predict the class of an uploaded image
def predict_image(image_path):
    img = Image.open(image_path)
    results = yolo_model(img)  # Directly use YOLOv8 for prediction

    # Check if any results were found
    if len(results) == 0:
        return None
    
    # Extract the first detected object's class
    pred = results[0].boxes  # Get bounding boxes for the predictions
    if pred is not None and len(pred) > 0:
        # Assuming we want the first object detected (this can be adjusted)
        class_name = results[0].names[int(pred[0].cls)]  # Get the class name from the 'names' list
        return class_name
    else:
        return None

def search_youtube(query):
    api_key = 'AIzaSyCPJ44pc4wQGLGBHtCNKwe4sX8mQpojHVU'  # Replace with your actual YouTube API key
    url = f'https://www.googleapis.com/youtube/v3/search?q={query}+tutorial&part=snippet&maxResults=3&key={api_key}'
    response = requests.get(url)
    
    if response.status_code != 200:
        return []  # If the API call fails, return an empty list

    videos = response.json().get('items', [])
    video_links = []

    for video in videos:
        if video['id']['kind'] == 'youtube#video':
            video_id = video['id']['videoId']
            video_links.append(f'https://www.youtube.com/watch?v={video_id}')
    
    return video_links

def search_recipe_videos(query):
    api_key = 'AIzaSyCPJ44pc4wQGLGBHtCNKwe4sX8mQpojHVU'  # Replace with your actual YouTube API key
    url = f'https://www.googleapis.com/youtube/v3/search?q={query}+recipe&part=snippet&maxResults=3&key={api_key}'
    response = requests.get(url)
    
    if response.status_code != 200:
        return []  # If the API call fails, return an empty list

    videos = response.json().get('items', [])
    video_links = []

    for video in videos:
        if video['id']['kind'] == 'youtube#video':
            video_id = video['id']['videoId']
            video_links.append(f'https://www.youtube.com/watch?v={video_id}')
    
    return video_links

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def app_page():
    return render_template('app.html')  # Flask will render the app.html from the templates folder

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('page-contact.html')




@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Predict the class of the uploaded image
    class_name = predict_image(file_path)

    if class_name is None:
        return jsonify({"error": "No objects detected in the image"}), 400
    
    # Get YouTube tutorial videos related to the class
    youtube_links = search_youtube(class_name)

    # Get YouTube recipe videos related to the class
    recipe_video_links = search_recipe_videos(class_name)

    return jsonify({
        "class_name": class_name,
        "image_url": f"/uploads/{filename}",  # This is the URL for the image
        "youtube_tutorial_video": youtube_links[0] if youtube_links else "No tutorial video found",
        "recipe_video_links": recipe_video_links
    })

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
