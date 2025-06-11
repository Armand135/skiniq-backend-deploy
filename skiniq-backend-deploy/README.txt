SkinIQ Backend Deployment Instructions (Render)

1️⃣ Upload code to GitHub repository.

2️⃣ Create a new Web Service on https://dashboard.render.com/

3️⃣ Set the build command:
    pip install -r requirements.txt

4️⃣ Set the start command:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

5️⃣ Set Python version to 3.9

6️⃣ Done. Render will automatically pull model from Hugging Face on first launch.

⚠️ Don't forget to update MODEL_URL in main.py with your real Hugging Face model URL.