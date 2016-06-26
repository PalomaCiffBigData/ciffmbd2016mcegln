# -*- coding: utf-8 -*-
import string
import ciffmbd2016mcegln
import pandas as pd
import numpy as np
import statsmodels.api as stats
import re
import requests
import random
import math

import scipy.stats
from scipy import stats as scistats

from deap import creator, base, tools, algorithms
import urllib2
import os
import os.path
import time

from sklearn.metrics import roc_auc_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

def IdentificacionTipos (df):
    list_ib  = set() #input binary
    list_icn = set() #input categorical nominal
    list_ico = set() #input categorical ordinal
    list_if  = set() #input numerical continuos (input float)
    list_inputs = set()

    dfVars = pd.DataFrame(columns=('var_name', 'tipo', 'categoria', 'minimo', 'maximo', 'num_valores'))

    for i in range(1, len(df.columns)):
        var_name = df.columns[i]
        if (var_name != output_var):        
            list_inputs.add(var_name)         
            oUnique = df[var_name].unique()               
            iNumValores = len(oUnique)
            minimo = min(df[var_name])
            maximo = max(df[var_name])

            try:        
                sTipo = type(df[var_name][0])
                sCategoria = 'desconocido'

                if (iNumValores == 2 and ((oUnique == [0,1]).all() or (oUnique == [1,0]).all())):
                    list_ib.add(var_name)
                    sCategoria = 'ib'
                elif (iNumValores == 1 and ((oUnique == [1]).all() or (oUnique == [0]).all())):
                    list_ib.add(var_name)        
                    sCategoria = 'ib'        
                elif (sTipo == np.int64 and minimo < 5 and iNumValores < 30 and ((maximo - minimo)*0.8 < iNumValores) ):
                    list_icn.add(var_name)
                    sCategoria = 'ico'
                elif (sTipo == np.float64 or sTipo == np.int64):
                    list_if.add(var_name)
                    sCategoria = 'if'
                else:
                    sCategoria = 'desconocida'     
            except:
                sTipo = 'error'
                sCategoria = 'error'

            dfVars.loc[i] = [var_name, sTipo, sCategoria, minimo, maximo, iNumValores]

    return list_ib, list_icn, list_ico, list_if, list_inputs, dfVars


#LIMPIEZA DE DATOS: rellenar NaNs y valores fuera de rango
def LimpiezaDatos (df, dfo, list_inputs, list_if):    
    for var_name in df.columns:
        if (var_name in list_inputs):
            minimo = min(df[var_name])
            maximo = max(df[var_name])
            stddev = df[var_name].std()

            if (var_name in list_if):
                valormalo = df[df[output_var] == 1][var_name].mean()
                dfo.loc[dfo[var_name].isnull(), var_name] = valormalo
                dfo.loc[dfo[var_name] < (minimo - (stddev/10)), var_name] = valormalo            
                dfo.loc[dfo[var_name] > (maximo + (stddev/10)), var_name] = valormalo
            else:            
                valormalo = scistats.mode(df[df[output_var] == 1][var_name])[0][0]
                dfo.loc[dfo[var_name].isnull(), var_name] = valormalo
                dfo.loc[dfo[var_name] < minimo, var_name] = valormalo            
                dfo.loc[dfo[var_name] > maximo, var_name] = valormalo       
    
    return dfo



def CreacionRatios (df, dfo, list_inputs, list_if, iMaxRatios):
    print("Variable iniciales: " + str(len(df.columns)))
    iNumCols = len(df.columns)
    iNumRatios = 0

    for i in range(0, iNumCols):
        if (iNumRatios > iMaxRatios):
            break;
        vx = df.columns[i]

        if (vx in list_if and vx <> output_var):
            # x^2
            sNombre = 'ratioCuad#'+ vx
            df [sNombre] = df [vx] * df [vx]
            dfo[sNombre] = dfo[vx] * dfo[vx]
            list_inputs.add(sNombre)
            list_if.add(sNombre)
            iNumRatios = iNumRatios + 1

        for j in range(i+1, min(i+2, iNumCols)):
            vy = df.columns[j]    

            if (vx in list_if and vy in list_if):            
                # x+y
                sNombre = 'ratioSum#'+ vx + vy
                df [sNombre] = df [vx] + df [vy]
                dfo[sNombre] = dfo[vx] + dfo[vy]
                list_inputs.add(sNombre)
                list_if.add(sNombre)
                iNumRatios = iNumRatios + 1

                # (x-y)/y
                sNombre = 'ratioDif#'+ vx + vy
                df [sNombre] = np.where(df [vy]== 0, 0, (df [vx] - df [vy]) / df [vy])
                dfo[sNombre] = np.where(dfo[vy]== 0, 0, (dfo[vx] - dfo[vy]) / dfo[vy])
                list_inputs.add(sNombre)
                list_if.add(sNombre)
                iNumRatios = iNumRatios + 1

                # x*y
                sNombre = 'ratioMult#'+ vx + vy
                df [sNombre] = df [vx] * df [vy]
                dfo[sNombre] = dfo[vx] * dfo[vy]
                list_inputs.add(sNombre)
                list_if.add(sNombre)
                iNumRatios = iNumRatios + 1

                # x/y
                sNombre = 'ratioDiv#'+ vx + vy
                df [sNombre] = np.where(df [vy]== 0, 0, df [vx] / df [vy])
                dfo[sNombre] = np.where(dfo[vy]== 0, 0, dfo[vx] / dfo[vy])
                list_inputs.add(sNombre)
                list_if.add(sNombre)
                iNumRatios = iNumRatios + 1

                # x-y
                sNombre = 'ratioRest#'+ vx + vy
                df [sNombre] = df [vx] - df [vy]
                dfo[sNombre] = dfo[vx] - dfo[vy]
                list_inputs.add(sNombre)
                list_if.add(sNombre)
                iNumRatios = iNumRatios + 1

    print("Variable finales: " + str(len(df.columns)))
    
    return df, dfo, list_inputs, list_if



def Normalizacion (df, dfo, list_if):
    #NORMALIZACION DE VARIABLES
    for var_name in df.columns:
        if (var_name in list_if):
            media = df[df[var_name].notnull()][abs(df[var_name]) < np.inf][var_name].mean()
            desv  = df[df[var_name].notnull()][abs(df[var_name]) < np.inf][var_name].std()    

            if desv != 0:
                df [var_name] = (df [var_name] - media ) / desv               
                dfo[var_name] = (dfo[var_name] - media ) / desv     
                
    return df, dfo, list_if
