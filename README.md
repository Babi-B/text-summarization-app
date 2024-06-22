# Text Summarization App
## Overview
This is a web application for text summarization using both extractive and abstractive methods. The app allows users to input a block of text and choose between an extractive summary (using the Sumy library) and an abstractive summary (using the T5 model from the Transformers library).

## Features
-   Extractive Summarization: Extracts key sentences from the text to form a summary.
-   Abstractive Summarization: Generates a summary using deep learning models that paraphrase the text.

## Installation
To run this app locally, follow these steps:

### Prerequisites
Python 3.8 or later
pip (Python package installer)

### Clone the Repository

git clone https://github.com/yourusername/text-summarization-app.git
cd text-summarization-app

### Set Up a Virtual Environment (Optional)

It is recommended to use a virtual environment to manage dependencies:
    `python -m venv venv`

### Set Up the Virtual Environment
-   **Windows**
    `venv\Scripts\activate`
-   **MacOS/Linux**
    `source venv/bin/activate`

### Install dependencies
    `pip install -r requirements.txt`

### Usage
    `streamlit run app.py`