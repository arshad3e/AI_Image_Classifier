import tensorflow as tf
from PIL import Image, ImageDraw
import requests
import json
import io  # For handling image data
import os
import base64

# Clarifai API Key (Replace with your actual API key)
CLARIFAI_API_KEY = "YOUR_CLARIFAI_API_KEY" # IMPORTANT: Replace with your Clarifai API key
CLARIFAI_MODEL_ID = "general-image-recognition"  # General image recognition model


def classify_image_clarifai(image_path, api_key=CLARIFAI_API_KEY, model_id=CLARIFAI_MODEL_ID):
    """
    Classifies objects in an image using the Clarifai API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): Your Clarifai API key.
        model_id (str): The Clarifai model ID to use. Defaults to "general-image-recognition".

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'label': The predicted label (e.g., "dog", "car").
              - 'confidence': The confidence score (0.0 to 1.0).
              - 'bounding_box': A dictionary containing 'top', 'left', 'bottom', 'right'
                               representing the normalized coordinates of the bounding box.
              Returns an empty list if there's an error.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")  #encode to base64

    url = f"https://api.clarifai.com/v2/models/{model_id}/outputs"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": base64_image
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        results = response.json()

        predictions = []
        for concept in results['outputs'][0]['data']['concepts']:
            prediction = {
                'label': concept['name'],
                'confidence': concept['value']
            }
            predictions.append(prediction)
        return predictions

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return []
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def classify_image_clarifai_with_bounding_boxes(image_path, api_key=CLARIFAI_API_KEY, model_id="general-image-detection"): # Use detection model
    """
    Classifies objects and gets bounding boxes in an image using the Clarifai API.

    Args:
        image_path (str): Path to the image file.
        api_key (str): Your Clarifai API key.
        model_id (str): The Clarifai model ID for object detection (e.g., "general-image-detection").

    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - 'label': The predicted label (e.g., "dog", "car").
              - 'confidence': The confidence score (0.0 to 1.0).
              - 'bounding_box': A dictionary containing 'top', 'left', 'bottom', 'right'
                               representing the normalized coordinates of the bounding box.
              Returns an empty list if there's an error.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")  #encode to base64

    url = f"https://api.clarifai.com/v2/models/{model_id}/outputs"
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": base64_image
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        results = response.json()

        predictions = []
        for region in results['outputs'][0]['data']['regions']:
            concept = region['data']['concepts'][0] # Get the top concept
            prediction = {
                'label': concept['name'],
                'confidence': concept['value'],
                'bounding_box': region['region_info']['bounding_box']
            }
            predictions.append(prediction)
        return predictions

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return []
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def draw_bounding_boxes(image_path, predictions, output_path="labeled_image.jpg"):
    """
    Draws bounding boxes on an image based on Clarifai predictions.

    Args:
        image_path (str): Path to the image file.
        predictions (list): A list of prediction dictionaries from classify_image_clarifai_with_bounding_boxes.
        output_path (str): Path to save the labeled image.
    """
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        for prediction in predictions:
            box = prediction['bounding_box']
            label = prediction['label']
            confidence = prediction['confidence']

            # Corrected key names to match the Clarifai API response
            left = box['left_col'] * width
            top = box['top_row'] * height
            right = box['right_col'] * width
            bottom = box['bottom_row'] * height

            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
            draw.text((left, top - 10), f"{label}: {confidence:.2f}", fill="red")  # Add label and confidence

        image.save(output_path)
        print(f"Labeled image saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")



# Example usage
if __name__ == "__main__":
    # 1. Download a test image (or use your own)
    image_url = "https://samples.clarifai.com/metro-north.jpg"  # Example image with objects
    image_path = "test_image.jpg"

    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(image_path, 'wb') as out_file:
            for block in response.iter_content(1024):
                out_file.write(block)
        print(f"Downloaded test image to {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        exit()


    # 2. Classify the image using Clarifai (General Recognition Model)
    print("\nClassifying with general recognition model:")
    predictions = classify_image_clarifai(image_path)
    if predictions:
        for p in predictions:
            print(f"  {p['label']}: {p['confidence']:.4f}")
    else:
        print("  No predictions received.")

    # 3. Classify the image with bounding boxes (Object Detection Model)
    print("\nClassifying with object detection model and drawing bounding boxes:")
    predictions_with_boxes = classify_image_clarifai_with_bounding_boxes(image_path)
    if predictions_with_boxes:
        for p in predictions_with_boxes:
            print(f"  {p['label']}: {p['confidence']:.4f}, Bounding Box: {p['bounding_box']}")
        draw_bounding_boxes(image_path, predictions_with_boxes, "labeled_image_with_boxes.jpg")
    else:
        print("  No predictions with bounding boxes received.")
