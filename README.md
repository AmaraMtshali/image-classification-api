# Image Classification API

This is a Flask-based REST API that uses a pre-trained ResNet50 model to classify images provided via a URL. The API accepts a POST request with a JSON payload containing an image URL, processes the image, and returns the predicted class and confidence score.

## API Endpoint

**POST /predict**

Classifies an image from a provided URL.

### Request

- **Method**: POST
- **URL**: `/predict`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "image_url": "<public_image_url>"
  }