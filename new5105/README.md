# Steganography Web Interface

A lightweight web interface for hiding secret messages in images using the SteganoGAN library.

## Features

- Upload images to encode secret messages
- View and download the encoded images
- Upload encoded images to decode hidden messages
- Simple, intuitive user interface

## Requirements

- Python 3.12 or higher
- Flask
- SteganoGAN

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   or
   ```
   pip install flask steganogan
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Use the interface to encode and decode messages in images

## How it Works

The application uses the SteganoGAN library, which is based on deep neural networks, to hide text messages within images in a way that is imperceptible to the human eye, yet can be decoded by the same model.
