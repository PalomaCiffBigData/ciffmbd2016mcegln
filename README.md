## ciffmbd2016mcegln
###Proyecto CIFF-MBD-Finanzas (AF#5)

![Image of Yaktocat](https://s31.postimg.org/ukn96biwb/Imagen1.png width="48px" height="50px")


## AutoPredict

Bunch of libraries that will make the world a better place. Edge technology to work with data and build models.

### Install

pip install ciffmbd2016mcegln

### Usage

**IdentificacionTipos (df, output_var)**: Recibe un dataframe y nos indica de que tipo es cada columna
```python
dfVars = IdentificacionTipos(df, output_var) # out_var es el nombre de la variable target
```

**LimpiezaDatos (df, dfo, output_var, list_inputs, list_if)**:   Rellenar NaNs y valores fuera de rango
```python
dfo = LimpiezaDatos (df, dfo, output_var, list_inputs, list_if)
# df es el dataframe de training, para obtener la media y la desviacion
# dfo para el dataframe de test
# output_var variables de salida
# list_inputs lista de todas las variables
# list_if lista de todas las variables tipo float
```

**CreacionRatios (df, dfo, output_var, list_inputs, list_if, iMaxRatios)**: Partiendo de un conjunto de variables, realizar una explosión de las mismas combinándolas dos a dos mendiante funciones aritméticas
```python
df, dfo, list_inputs, list_if = CreacionRatios(df, dfo, output_var, list_inputs, list_if, 200)

```
**Normalizacion (df, dfo, list_if)**: Normaliza las variables
```python
df, dfo, list_if = Normalizacion(df, dfo, list_if)
```
**PCAexpand (df, dfo, iNumComponentes)**: Realiza una transformación PCA
```python
df, dfo = PCAexpand(df, dfo, output_var, id_var, 20)
```
**GeneticFeatureSelection (df, dfo, output_var, list_inputs, iNumEstimators)**: Selecciona las mejores variables usando algorito genético.
```python
gini, listFeatures = GeneticFeatureSelection(df, dfo, output_var, list_inputs, 20)
```
