.PHONY: install install-train data test api app up

install:          ## dependencias del producto
	pip install -r requirements.txt

install-train:    ## dependencias de entrenamiento (incluye las del producto)
	pip install -r requirements-train.txt

data:             ## baja la data oficial de Kaggle a data/ (requiere ~/.kaggle/kaggle.json)
	kaggle competitions download -c hackathon-ia-pucp-2026 -p data
	cd data && unzip -o hackathon-ia-pucp-2026.zip

test:
	pytest -q

api:
	uvicorn app.api.main:app --reload --port 8000

app:
	streamlit run app/ui/streamlit_app.py

up:
	docker-compose up --build
