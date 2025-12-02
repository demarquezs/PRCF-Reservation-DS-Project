FROM python:3.12-slim

WORKDIR /app

#copy dependencies and install 
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy application files and model
COPY api/main.py .
COPY ../models /app/models

#expose fastAPI port
EXPOSE 8000 

#run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]