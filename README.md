Tarea N°1

Para esta tarea: 

1.- Se escogieron los modelos de Regresión Logística Multi Clase, Random Forest y Gradient Boosting .

2.- La base de datos seleccionada fue  “California Housing”, proporcionado por la librería “sklearn” de Python, la cual contiene un conjunto de datos derivado de un censo en California. Se discretizo la Variable objetivo "Target", con el objetivo de dividir los valores en cuartiles que nos permitietran crear 3 clases con las que trabajar. El cuartil N°1, representa la clase "Precio Bajo", el cuartil N°3 representa la clase "Precio Alto", y el resto "Precio Medio"

Al momento de correr los modelos podemos interpretar los resultados de la siguiente manera: 

Regresion Logistica Multiclase

- 911 (Viviendas Baratas): Casos correctamente clasificados como bajo. 
- 617 (Error Viviendas Medias-Baratas): Casos de viviendas que eran baratas pero clasificó como medias. 
- 9 (Errores Viviendas Caras-Baratas): Casos de viviendas que eran baratas pero clasificó como caras. 
- 347 (Errores Viviendas Baratas-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como baratas. 
- 2510 (Viviendas Media): Casos de viviendas de precio medio que el modelo predijo correctamente como precio medio. 
- 243 (Errores Viviendas Caras-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como caras. 
- 47 (Errores Viviendas Baratas-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como baratas. 
- 602 (Errores Viviendas Media-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como medio.
- 906 (Viviendas Caras): Casos de viviendas de precio alto que el modelo predijo correctamente como caras. 

Random Forest
- 1235(Viviendas Baratas): Casos correctamente clasificados como bajo. 
- 299(Error Viviendas Medias-Baratas): Casos de viviendas que eran baratas pero clasificó como medias. 
- 3 (Errores Viviendas Caras-Baratas): Casos de viviendas que eran baratas pero clasificó como caras. 
- 220(Errores Viviendas Baratas-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como baratas. 
- 2693(Viviendas Media): Casos de viviendas de precio medio que el modelo predijo correctamente como precio medio. 
- 187(Errores Viviendas Caras-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como caras. 
- 9(Errores Viviendas Baratas-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como baratas. 
- 371(Errores Viviendas Media-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como medio.
- 1175(Viviendas Caras): Casos de viviendas de precio alto que el modelo predijo correctamente como caras.

Gradient Boosting

- 1196(Viviendas Baratas): Casos correctamente clasificados como bajo. 
- 336(Error Viviendas Medias-Baratas): Casos de viviendas que eran baratas pero clasificó como medias. 
- 5 (Errores Viviendas Caras-Baratas): Casos de viviendas que eran baratas pero clasificó como caras. 
- 221(Errores Viviendas Baratas-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como baratas. 
- 2690(Viviendas Media): Casos de viviendas de precio medio que el modelo predijo correctamente como precio medio. 
- 189(Errores Viviendas Caras-Media): Casos de viviendas de precio medio que el modelo predijo erróneamente como caras. 
- 11(Errores Viviendas Baratas-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como baratas. 
- 370(Errores Viviendas Media-Caras): Casos de viviendas de precio alto que el modelo predijo erróneamente como medio.
- 1174(Viviendas Caras): Casos de viviendas de precio alto que el modelo predijo correctamente como caras. 

