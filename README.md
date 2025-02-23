# Image Classification with Object Detection using Clarifai

This project demonstrates image classification and object detection using the Clarifai API and TensorFlow (for image handling with Pillow/PIL).

## Overview

This script takes an image as input, uses the Clarifai API to identify objects within the image and their bounding boxes, and then draws these bounding boxes on the image, labeling each detected object with its name and confidence score.

**Here's a visual representation of the process:**

### Before Object Detection:

![test_image](https://github.com/user-attachments/assets/21aa0d06-24a8-47f0-9c4e-a5638e5a226d)


### After Object Detection:

![labeled_image_with_boxes](https://github.com/user-attachments/assets/8af8d7d8-a17d-45b8-8fb3-97a2d95d1012)


## Prerequisites

Before running the script, ensure you have the following:

*   **Python 3.6+:**  The script is written in Python 3 and may not be compatible with older versions.
*   **Required Libraries:** Install the necessary libraries using pip:

    ```bash
    pip install tensorflow Pillow requests
    ```

*   **Clarifai Account and API Key:**
    *   Sign up for a free Clarifai account at [https://www.clarifai.com/](https://www.clarifai.com/)
    *   Create an application within your Clarifai account.
    *   Obtain your API key from the application settings.
*   **Replace Placeholder API Key:** In the `image_classifier.py` file, replace `"YOUR_CLARIFAI_API_KEY"` with your actual Clarifai API key.

## How to Run

1.  **Download the Script:** Save the `image_classifier.py` file to your local machine.

2.  **Execute the Script:** Open a terminal or command prompt and navigate to the directory where you saved the script.  Run the script using:

    ```bash
    python image_classifier.py
    ```

3.  **View the Results:**

    *   The script will download a sample image (`test_image.jpg`).
    *   It will print the classification results (labels and confidence scores) to the console.
    *   A new image named `labeled_image_with_boxes.jpg` will be created in the same directory as the script.  This image will have bounding boxes drawn around the detected objects, along with their labels and confidence scores.

## Customization

*   **Input Image:**  To process a different image, change the `image_url` variable in the `if __name__ == "__main__":` block to the URL of your desired image.  Alternatively, you can modify the script to accept an image path as a command-line argument.

*   **Clarifai Model:** The script uses the `general-image-detection` Clarifai model for object detection. You can explore other Clarifai models for more specific tasks. Change the `CLARIFAI_MODEL_ID` variable accordingly.

*   **Confidence Threshold:** You can adjust the confidence threshold in the script to filter out low-confidence predictions.  Modify the `predictions` list comprehension to include a confidence check:

    ```python
    predictions = [p for p in predictions if p['confidence'] > 0.7]  # Keep predictions with > 70% confidence
    ```

*   **Output Path:**  Modify the `output_path` argument in the `draw_bounding_boxes` function to change the name or location of the output image.

## Troubleshooting

*   **KeyError: 'left_col' (or similar):** This error usually indicates that the Clarifai API response format has changed. Double-check the API documentation or examine the `results` dictionary in the `classify_image_clarifai_with_bounding_boxes` function to identify the correct key names for bounding box coordinates. (This has already been addressed in the code provided).

*   **API Request Errors:** If you encounter errors related to API requests (e.g., HTTP errors), ensure that your API key is correct and that you have sufficient usage credits in your Clarifai account.

*   **Image Not Found:** Verify that the image file specified by `image_url` exists and is accessible.

## License

MIT License

## Author

Arshad
