import gzip
import json
import os
import pickle
import zipfile

from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import (  
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import GridSearchCV  
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import OneHotEncoder  


def cargar_datasets_comprimidos(file_directory):
	list_dfs = []
	files_zip = glob(os.path.join(file_directory, "*"))
	
	for file in files_zip:
		with zipfile.ZipFile(file, "r") as zip_file:
			for content in zip_file.namelist():
				with zip_file.open(content) as file_csv:
					df = pd.read_csv(file_csv, sep=",", index_col=0)
					list_dfs.append(df)
	
	return list_dfs


def limpiar_directorio(directory):
	if os.path.exists(directory):
		for file in glob(os.path.join(directory, "*")):
			os.remove(file)
			
		os.rmdir(directory)
	
	os.makedirs(directory, exist_ok=True)


def serializar_modelo_comprimido(file_path, model):
	parent_directory = os.path.dirname(file_path)
	limpiar_directorio(parent_directory)
	
	with gzip.open(file_path, "wb") as file_gz:
		pickle.dump(model, file_gz)


def limpiar_datos(dataframe):
	data = dataframe.copy()
	
	data = data.rename(columns={"default payment next month": "default"})
	
	data = data.loc[data["MARRIAGE"] != 0]
	data = data.loc[data["EDUCATION"] != 0]
	
	data["EDUCATION"] = data["EDUCATION"].apply(lambda valor: 4 if valor >= 4 else valor)
	
	return data.dropna()


def dividir_caracteristicas_objetivo(dataframe):
	characteristics = dataframe.drop(columns=["default"])
	objective = dataframe["default"]
	return characteristics, objective


def construir_pipeline_optimizacion():
	variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
	
	codificador = OneHotEncoder(handle_unknown="ignore")
	
	transformador = ColumnTransformer(
		transformers=[("categoricas", codificador, variables_categoricas)],
		remainder="passthrough",
	)
	
	bosque_aleatorio = RandomForestClassifier(random_state=42)
	
	pipeline = Pipeline(
		steps=[
			("preprocesamiento", transformador),
			("clasificador", bosque_aleatorio),
		]
	)
	
	parametros = {
		"clasificador__n_estimators": [100, 200, 500],
		"clasificador__max_depth": [None, 5, 10],
		"clasificador__min_samples_split": [2, 5],
		"clasificador__min_samples_leaf": [1, 2],
	}
	
	optimizador = GridSearchCV(
		estimator=pipeline,
		param_grid=parametros,
		cv=10,
		scoring="balanced_accuracy",
		n_jobs=-1,
		refit=True,
		verbose=2,
	)
	
	return optimizador


def calcular_metricas_rendimiento(nombre_conjunto, valores_reales, valores_predichos):
	metrics = {
		"type": "metrics",
		"dataset": nombre_conjunto,
		"precision": precision_score(valores_reales, valores_predichos, zero_division=0),
		"balanced_accuracy": balanced_accuracy_score(valores_reales, valores_predichos),
		"recall": recall_score(valores_reales, valores_predichos, zero_division=0),
		"f1_score": f1_score(valores_reales, valores_predichos, zero_division=0),
	}
	return metrics


def generar_matriz_confusion(nombre_conjunto, valores_reales, valores_predichos):
	matriz = confusion_matrix(valores_reales, valores_predichos)
	
	result = {
		"type": "cm_matrix",
		"dataset": nombre_conjunto,
		"true_0": {
			"predicted_0": int(matriz[0][0]),
			"predicted_1": int(matriz[0][1])
		},
		"true_1": {
			"predicted_0": int(matriz[1][0]),
			"predicted_1": int(matriz[1][1])
		},
	}
	
	return result


def ejecutar_pipeline_completo():
	datasets_crudos = cargar_datasets_comprimidos("files/input")
	datasets_limpios = [limpiar_datos(dataset) for dataset in datasets_crudos]
	
	datos_prueba, datos_entrenamiento = datasets_limpios
	
	X_entrenamiento, y_entrenamiento = dividir_caracteristicas_objetivo(datos_entrenamiento)
	X_prueba, y_prueba = dividir_caracteristicas_objetivo(datos_prueba)
	
	modelo_optimizado = construir_pipeline_optimizacion()
	modelo_optimizado.fit(X_entrenamiento, y_entrenamiento)
	
	ruta_modelo = os.path.join("files", "models", "model.pkl.gz")
	serializar_modelo_comprimido(ruta_modelo, modelo_optimizado)
	
	predicciones_prueba = modelo_optimizado.predict(X_prueba)
	predicciones_entrenamiento = modelo_optimizado.predict(X_entrenamiento)
	
	metricas_entrenamiento = calcular_metricas_rendimiento(
		"train", y_entrenamiento, predicciones_entrenamiento
	)
	metricas_prueba = calcular_metricas_rendimiento(
		"test", y_prueba, predicciones_prueba
	)
	
	confusion_entrenamiento = generar_matriz_confusion(
		"train", y_entrenamiento, predicciones_entrenamiento
	)
	confusion_prueba = generar_matriz_confusion(
		"test", y_prueba, predicciones_prueba
	)
	
	Path("files/output").mkdir(parents=True, exist_ok=True)
	
	ruta_metricas = "files/output/metrics.json"
	with open(ruta_metricas, "w", encoding="utf-8") as archivo_metricas:
		archivo_metricas.write(json.dumps(metricas_entrenamiento) + "\n")
		archivo_metricas.write(json.dumps(metricas_prueba) + "\n")
		archivo_metricas.write(json.dumps(confusion_entrenamiento) + "\n")
		archivo_metricas.write(json.dumps(confusion_prueba) + "\n")

if __name__ == "__main__":
    ejecutar_pipeline_completo()