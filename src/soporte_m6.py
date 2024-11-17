# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway
from scipy.stats import bartlett
from scipy.stats import ttest_ind
import itertools as iter


def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]

        if type(categoria) == str:
    
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)
            
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
        
        else:
            
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para {categoria} son: ")
            display(dataframe_filtrado.describe().T)




#--------------------------------------------------------------------------------------------------------
def normalidad_shapiro (dataframe,group,datos):
        # Iterar sobre los grupos (por ejemplo, por 'tipo_dato')
    for tipo in dataframe[f'{group}'].unique():
        # Filtrar el DataFrame para cada grupo
        grupo = dataframe[dataframe[f'{group}'] == tipo]
        
        # Aplicar el test de Shapiro-Wilk para verificar normalidad
        stat, p_value = shapiro(grupo[f"{datos}"])
        
        # Mostrar los resultados para cada grupo
        print(f'Grupo: {tipo}')
        print(f'Estadístico Shapiro-Wilk: {stat}')
        print(f'Valor p: {p_value}')
        
        # Interpretación del valor p
        if p_value > 0.05:
            print("No se puede rechazar la hipótesis nula: los datos siguen una distribución normal")
        else:
            print("Se rechaza la hipótesis nula: los datos no siguen una distribución normal")
        
        print('---')

#-------------------------------------------------------------------------------------------------------------

def normalidad_kstest (dataframe,group,datos):
#Damos por hecho que los datos son independientes. Como hay más de 30 datos usaremos K-S para testear la normalidad de cada conjunto.

# Iterar sobre los grupos (por ejemplo, por 'tipo_dato')
    for tipo in dataframe[f"{group}"].unique():
        # Filtrar el DataFrame para cada grupo
        grupo = dataframe[dataframe[f"{group}"] == tipo]
        
        # Calcular la media y la desviación estándar del grupo
        media = grupo[f"{datos}"].mean()
        desviacion = grupo[f"{datos}"].std()
        
        # Aplicar el test de Kolmogorov-Smirnov para la distribución normal
        stat, p_value = kstest(grupo[f"{datos}"], 'norm', args=(media, desviacion))
        
        # Mostrar los resultados para cada grupo
        print(f'Grupo: {tipo}')
        print(f'Estadístico KS: {stat}')
        print(f'Valor p: {p_value}')
        
        # Interpretación del valor p
        if p_value > 0.05:
            print("No se puede rechazar la hipótesis nula: los datos siguen una distribución normal")
            return True
        else:
            print("Se rechaza la hipótesis nula: los datos no siguen una distribución normal")
            return False
        
        print('---')

#-----------------------------------------------------------------------------------------------------------------
def homocedasticidad_bartlett (dataframe,group,datos):

    # Agrupar los datos por 'tipo_dato'
    grupos = [dataframe[dataframe[f"{group}"] == tipo][f"{datos}"] for tipo in dataframe[f"{group}"].unique()]

    # Aplicar el test de Bartlett para comparar varianzas entre los grupos
    stat, p_value = bartlett(*grupos)

    # Mostrar los resultados del test
    print('Resultados del Test de Bartlett:')
    print(f'Estadístico de Bartlett: {stat}')
    print(f'Valor p: {p_value}')

    # Interpretación del valor p
    if p_value > 0.05:
        print("No se puede rechazar la hipótesis nula: las varianzas son homogéneas")
        return True
    else:
        print("Se rechaza la hipótesis nula: las varianzas no son homogéneas")
        return False


#-----------------------------------------------------------------------------------------------------------------
def homocedasticidad_levene (dataframe,group,datos):

    # Inicializamos un diccionario para almacenar los resultados
    resultados_levene = {}

    # Iterar sobre los grupos (por ejemplo, por 'tipo_dato')
    for tipo in dataframe[f"{group}"].unique():
        # Filtrar el DataFrame para cada grupo
        grupo = dataframe[dataframe[f"{group}"] == tipo]
        
        # Almacenamos los datos de cada grupo en una lista para aplicar el test de Levene
        resultados_levene[tipo] = grupo[f"{datos}"].values

    # Aplicamos el test de Levene para comparar las varianzas entre los grupos
    stat, p_value = levene(*resultados_levene.values())

    # Mostrar los resultados
    print(f'Estadístico Levene: {stat}')
    print(f'Valor p: {p_value}')

    # Interpretación del valor p
    if p_value > 0.05:
        print("No se puede rechazar la hipótesis nula: las varianzas son homogéneas entre los grupos")
        return True
    else:
        print("Se rechaza la hipótesis nula: las varianzas no son homogéneas entre los grupos")
        return False
#-----------------------------------------------------------------------------------------------------------------

def diferencias_anova (dataframe,group,datos):

    #Observamos que los datos son normales y homocedásticos, vamos a usar ANOVA para ver si hay diferencias 

    # Filtramos los datos por cada grupo
    grupos = [dataframe[dataframe[f"{group}"] == tipo][f"{datos}"] for tipo in dataframe[f"{group}"].unique()]

    # Aplicamos el test de ANOVA de una vía
    stat, p_value = f_oneway(*grupos)

    # Mostrar los resultados
    print(f'Estadístico ANOVA: {stat}')
    print(f'Valor p: {p_value}')

    # Interpretación del valor p
    if p_value > 0.05:
        print("No se puede rechazar la hipótesis nula: no hay diferencias significativas entre los grupos")
        return True
    else:
        print("Se rechaza la hipótesis nula: hay diferencias significativas entre los grupos")
        return False

#------------------------------------------------------------------------------------------------------------------

def parametrico (dataframe,group,datos):

    #Iteramos sobre los subgrupos, aplicamos el test de normalidad sobre cada subgrupo
    for tipo in dataframe[f"{group}"].unique():
        subgrupo = dataframe[dataframe[f"{group}"] == tipo]
        num_elementos = subgrupo.shape[0]
        if num_elementos > 30:
            print(f"El subgrupo tiene {num_elementos} elementos, usamos K-S")
            normal = normalidad_kstest (subgrupo,group,datos)
            print("---")
        else:
            print(f"El subgrupo tiene {num_elementos} elementos, usamos shapiro")
            normal = normalidad_shapiro (subgrupo,group,datos)
            print("---")


    if normal:
    #Si se cumple la condición de normalidad usamos bartlett. Sirve para más de dos grupos
        homocedasticidad=homocedasticidad_bartlett (dataframe,group,datos)
        print("El conjunto es normal")
    else:    
     #Puesto que levene sirve para más de dos grupos lo aplicamos si no hay normalidad:
        homocedasticidad = homocedasticidad_levene (dataframe,group,datos)
        print("El conjunto no cumple con la normalidad")
    print("---")

    if not homocedasticidad:
            print("Los conjuntos no son homocedásticos, hay que aplicar test no paramétrico")
    else:
            print("Los conjuntos son homocedásticos")
    

#-------------------------------------------------------------------------------------------------------------

def diferencias_por_grupos(dataframe, group, datos):
    # Obtener las categorías únicas de la columna 'temperatura'
    lista_categorias = [categoria for categoria in dataframe[group].unique()]

    # Generar todas las combinaciones posibles de 2 grupos
    combinaciones = list(iter.combinations(lista_categorias, 2))

    # Iterar sobre cada combinación de grupos
    for grupo1, grupo2 in combinaciones:
        # Filtrar los datos para los dos grupos
        datos_grupo1 = dataframe[dataframe[group] == grupo1][datos]
        datos_grupo2 = dataframe[dataframe[group] == grupo2][datos]
        
        # Realizar el test t-Student independiente
        t_stat, p_value = ttest_ind(datos_grupo1, datos_grupo2, equal_var=False)  

        # Imprimir los resultados del test
        print(f"Comparando los grupos '{grupo1}' y '{grupo2}':")
        print(f"  Estadístico t: {t_stat}")
        print(f"  Valor p: {p_value}")
        
        # Interpretar el valor p
        if p_value < 0.05:
            print("  Rechazamos la hipótesis nula: existe una diferencia significativa entre los grupos.")
        else:
            print("  No se rechaza la hipótesis nula: no existe una diferencia significativa entre los grupos.")
        print('---')
    