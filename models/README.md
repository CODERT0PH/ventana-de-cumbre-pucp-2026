# models/

Aquí van los **pesos de tu modelo final**: los mismos que generaron tu submission de Kaggle y que carga tu producto.

- Guarda solo los pesos, sin el estado del optimizador. Por ejemplo, en PyTorch: `torch.save(model.state_dict(), "models/modelo.pt")`.
- Si aun así pasan de **100 MB** (el límite de GitHub), súbelos a Google Drive o Hugging Face, con acceso para cualquiera con el link, y pon el link en el README principal y aquí.

Link a los pesos (solo si no están en esta carpeta): [pegar link]

## Archivos del modelo final

- `modelo_final.pt`: pesos (state_dict) del submission final.
- `artefactos.npz`: normalización, climatología por hora y canales estáticos derivados del DEM. Los genera `Preprocessor.save()` en el notebook; la app los carga con `Downscaler`.

Pendiente de confirmar con la organización: si `artefactos.npz` (derivado de la data oficial) puede ir en el repo público.
