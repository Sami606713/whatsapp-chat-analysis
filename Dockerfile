FROM python:3.12

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

# Add this line to download the punkt resource
RUN python -m nltk.downloader punkt

CMD ["streamlit", "run", "app.py"]