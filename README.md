
# Food Lens - Recipe Recommendation System

Food Lens is an AI-powered recipe recommendation system that suggests similar recipes based on an uploaded image. By analyzing the image, it retrieves the recipe details and provides nutritional information, preparation instructions, and the link to the original recipe.

## Features

- **Image Upload**: Upload an image of a dish to get similar recipe suggestions.
- **Recipe Details**: View detailed information about each recommended recipe, including ingredients, preparation steps, nutritional values, and more.
- **Nutrition Breakdown**: Get detailed nutritional breakdown for each recipe (calories, carbs, fat, protein, etc.).
- **Recipe Instructions**: View the full recipe with step-by-step cooking instructions.
  
## Prerequisites

- Python 3.x
- FastAPI (for backend)
- JavaScript (for frontend)
- Bootstrap 5 (for styling)

## Setup Instructions

### Backend

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/food-lens.git
   cd food-lens
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload
   ```

   The FastAPI backend should now be running at `http://127.0.0.1:8000`.

### Frontend

1. **Navigate to the `frontend` folder** (if applicable):
   ```bash
   cd frontend
   ```

2. **Open the `index.html` file in a browser**:
   - This is a simple static page that allows you to upload an image and get similar recipe recommendations.

### How It Works

- **Image Upload**: 
  - The user uploads an image of a dish, and the system sends the image to the backend.
  - The backend processes the image and uses machine learning models to find similar recipes based on image features.
  - The backend returns a list of similar recipes with their details, which are then displayed in the frontend.
  
- **Recipe Details**: 
  - Each recipe includes the title, ingredients, directions, nutritional values (calories, carbs, fat, protein), and a link to the full recipe.

## Folder Structure

```
/food-lens
    /app
        main.py        # FastAPI backend
        models.py      # Machine learning model for image analysis
    /frontend
        index.html     # Static HTML page for image upload
        /css
        /js
    requirements.txt   # List of dependencies for the backend
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
