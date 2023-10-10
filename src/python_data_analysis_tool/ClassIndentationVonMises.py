"""
Análisis general de indentaciónes

Este programa sirve para:

- Contar átomos en contacto con el indentador
- Estimar área de contacto
- Calcular desplazamiento, strain, stress
- Gráficar curvas force-displacement, stress-strain, CN-displacement
  y CN-strain

Versiones de software para las que se escribió este programa

- Python 3.9.0
- NumPy 1.22.2
- Pandas 1.4.2
- Matplotlib 3.5.1
- Ovito 3.7.12

Elaborado por: Jesús Loera
Fecha: 16/06/23

"""
# Se define la clase "Indentation", se trata a cada simulación como
# un objeto con atributos y métodos ue pueden ejecutarse de
# manera independiente dependiendo del análisis requerido 

# Librerías requeridas
from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mlp
# Activate 'agg' backend for off-screen plotting.
mlp.use('Agg') 

class Indentation:

    """CONSTRUCTOR"""

    # Cada dump_indent está caractarizado por la orientación
    # y por la semilla de la simulación
    def __init__(self, dumpfile, indentfile):
        self.dumpfile = dumpfile
        self.indentfile = indentfile
        self.dataIndent = pd.read_csv(self.indentfile, sep='\s+', index_col=False, header=0, names=["timestep", "initdiam", "xeta", "force", "pote"])

    """MÉTODOS"""

    # Método para añadir la columna "displacement"
    def setDisplacement(self):
        self.dataIndent['displacement'] = self.dataIndent['initdiam'] + 4.0725- self.dataIndent['xeta']
        self.dataIndent['displacement'] = (self.dataIndent['displacement'] + abs(self.dataIndent['displacement'].min()))/10
        print("Cálculo de desplazamiento términado (nm)")

    # Método para añadir la columna "strain"
    def setStrain(self):
        self.dataIndent['strain'] = self.dataIndent['initdiam'] + 4.0725 - self.dataIndent['xeta']
        self.dataIndent['strain'] = (self.dataIndent['strain'] + abs(self.dataIndent['strain'].min()))/(self.dataIndent['initdiam'])
        print("Cálculo de strain términado")

    # Método para añadir columnas que dicen el % de atómos con un número de
    # coordinación arbitrario
    def setCoordinationAnalysis(self, cutradius):
        print('Realizando análisis de coordinación, puede tardar unos minutos')
        self.pipeline = import_file(self.dumpfile, multiple_frames = True)
        # cutoff radius for silicon: 2.60
        modifier = CoordinationAnalysisModifier(cutoff = cutradius, number_of_bins = 100)
        self.pipeline.modifiers.append(modifier)
        dfAnalysis = pd.DataFrame(columns=("timestep",1,2,3,4,5,6,7))
        for frame in range(self.pipeline.source.num_frames):
            data = self.pipeline.compute(frame)
            dfCoordination = pd.DataFrame(data.particles['Coordination'], columns=['coordination-number'])
            # Obtenemos la proporción % de cada número de cordinación y el timestep del frame
            numberOfAtoms = dfCoordination['coordination-number'].count()
            dfPercentCN = dfCoordination['coordination-number'].value_counts().div(numberOfAtoms)
            dfPercentCN['timestep'] = data.attributes['Timestep']
            dfAnalysis.loc[frame] = dfPercentCN
        self.dataIndent["percent-cn-1"] = dfAnalysis[1]
        self.dataIndent["percent-cn-2"] = dfAnalysis[2]
        self.dataIndent["percent-cn-3"] = dfAnalysis[3]
        self.dataIndent["percent-cn-4"] = dfAnalysis[4]
        self.dataIndent["percent-cn-5"] = dfAnalysis[5]
        self.dataIndent["percent-cn-6"] = dfAnalysis[6]
        self.dataIndent["percent-cn-7"] = dfAnalysis[7]
        print('Análisis de coordinación terminado')

    # Método que añade una columna con el número átomos en contacto
    # con el plano indentador
    def setContactAtoms(self):
        print('Realizando análisis de átomos de contacto, puede tardar unos minutos')
        self.pipeline = import_file(self.dumpfile, multiple_frames = True)
        contactAtoms = []
        for frame in range(self.pipeline.source.num_frames):
            dataDump = self.pipeline.compute(frame)
            xeta = self.dataIndent[ self.dataIndent['timestep'] == dataDump.attributes['Timestep'] ]['xeta'].iloc[0]
            # Los átomos arriba del indentador son los que están en contacto
            contactAtoms.append(np.count_nonzero(dataDump.particles['Position'][:,2] > xeta))
        self.dataIndent["contact-atoms"] = contactAtoms
        print('Análisis de átomos de contacto terminado')

    def setPercentageContactAtoms(self):
        print('Realizando análisis de porcentaje de átomos de contacto, puede tardar unos minutos')
        if (not 'contact-atoms' in self.dataIndent.columns):
            self.setContactAtoms()
        self.pipeline = import_file(self.dumpfile, multiple_frames = True)
        dataDump = self.pipeline.compute()
        numberOfAtoms = dataDump.particles.count
        self.dataIndent["percentage-contact-atoms"] = self.dataIndent["contact-atoms"].div(numberOfAtoms)
        print('Análisis de porcentaje átomos de contacto terminado')

    # Método para estimar el área de contacto con el plano indentador
    def setContactArea(self, atomic_area):
        if (not 'contact-atoms' in self.dataIndent.columns):
            self.setContactAtoms()
        print('Calculando área de contacto')
        # Se le asigna a cada átomo en contacto un área pi*r**2=17.35 al silicio
        self.dataIndent['contact-area'] = atomic_area*self.dataIndent['contact-atoms']
        print('Análisis de área de contacto terminado [amstrongs^2]')

    # Método para estimar el área de contacto con el plano indentador
    # de acuerdo a un área elíptica descrita por Eduardo Bringa 
    # "Mechanical Properties Obtained by Indentation of Hollow Pd Nanoparticles"
    def setEllipticalContactArea(self):
        print('Calculando área elíptica de contacto')
        self.pipeline = import_file(self.dumpfile, multiple_frames = True)
        contactEllipticalArea = []
        for frame in range(self.pipeline.source.num_frames):
            dataDump = self.pipeline.compute(frame)
            df_coords = pd.DataFrame(dataDump.particles.positions[:,:], columns=['x', 'y', 'z'])
            xeta = self.dataIndent[ self.dataIndent['timestep'] == dataDump.attributes['Timestep'] ]['xeta'].iloc[0]
            # Identificamos los átomos en la vecindand del plano indentador
            IndentNeigh = df_coords[ abs(df_coords['z'] - xeta) < 0.2 ]
            xmax = np.max(IndentNeigh['x'])
            xmin = np.min(IndentNeigh['x'])
            ymax = np.max(IndentNeigh['y'])
            ymin = np.min(IndentNeigh['y'])
            area = (np.pi/4.0)*(xmax-xmin)*(ymax-ymin)
            # Los átomos arriba del indentador son los que están en contacto
            contactEllipticalArea.append(area)
        self.dataIndent["contact-elliptical-area"] = contactEllipticalArea
        self.dataIndent["contact-elliptical-area"].fillna(0.0, inplace=True)
        print('Análisis de átomos de área elíptica de contacto términado [amstrongs^2]')

    # Método para cálcular el stress en GPa
    def setStress(self, atomic_area):
        if (not 'contact-area' in self.dataIndent.columns):
            self.setContactArea(atomic_area)
        print('Calculando stress por medio del área de contacto elíptica')
        self.dataIndent['stress'] = (self.dataIndent['force']/self.dataIndent['contact-area'])*160.2176  # En GPa
        print('Cálculo de stress terminado (gigapascales)')

    # Método para cálcular el stress en GPa por medio
    # del área de contacto elíptica
    def setEllipticalStress(self):
        if (not 'contact-elliptical-area' in self.dataIndent.columns):
            self.setEllipticalContactArea()
        print('Calculando stress')
        self.dataIndent['elliptical-stress'] = (self.dataIndent['force']/self.dataIndent['contact-elliptical-area'])*160.2176  # En GPa
        print('Cálculo de stress por medio del área de contacto elíptica terminado (gigapascales)')

    # Gráfica de la curva fuerza-desplazamiento
    def plotForceDisplacement(self, filename, title):
        if (not 'displacement' in self.dataIndent.columns):
            self.setDisplacement()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['displacement'], self.dataIndent['force']*(1.60218))   # En nano newtons
        plt.title(title)
        plt.ylabel("Force [nN]")
        plt.xlabel("Displacement [nm]")
        plt.savefig(filename)
        plt.clf()

    # Gráfica del número de átomos en contacto con el plano
    # indentador en función de strain
    def plotContactAtomsStrain(self, filename, title):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        if (not 'contact-atoms' in self.dataIndent.columns):
            self.setPercentageContactAtoms()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent['contact-atoms'])
        plt.title(title)
        plt.ylabel("Atoms in contact")
        plt.xlabel("Strain")
        plt.savefig(filename)
        plt.clf()

    # Gráfica del porcentaje de átomos en contacto con el plano
    # indentador en función de strain
    def plotPercentageContactAtomsStrain(self, filename, title):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        if (not 'percentage-contact-atoms' in self.dataIndent.columns):
            self.setPercentageContactAtoms()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent['percentage-contact-atoms'])
        plt.title(title)
        plt.ylabel("Percentage of atoms in contact %")
        plt.xlabel("Strain")
        plt.savefig(filename)
        plt.clf()

    # Gráfica de la curva Stress-Strain con la definicón
    # de área de contacto átomica
    def plotStressStrain(self, filename, title, atomic_area):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        if (not 'stress' in self.dataIndent.columns):
            self.setStress(atomic_area)
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent['stress'])
        plt.title(title)
        plt.ylabel("Stress [GPa]")
        plt.xlabel("Strain")
        plt.savefig(filename)
        plt.clf()

    # Gráfica de la curva Stress-Strain con la definicón
    # de área de contacto elíptica
    def plotEllipticalStressStrain(self, filename, title):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        if (not 'elliptical-stress' in self.dataIndent.columns):
            self.setStress()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent['elliptical-stress'])
        plt.title(title)
        plt.ylabel("Stress [GPa]")
        plt.xlabel("Strain")
        plt.savefig(filename)
        plt.clf()

    # Gráfica un análisis de coordinación en función
    # del desplazamiento
    def plotCoordinationAnalysis(self, filename, title):
        if (not 'displacement' in self.dataIndent.columns):
            self.setDisplacement()
        if (not 'percent-cn-4' in self.dataIndent.columns):
            self.setCoordinationAnalysis()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['displacement'], self.dataIndent["percent-cn-3"], label="CN=3")
        plt.plot(self.dataIndent['displacement'], self.dataIndent["percent-cn-4"], label="CN=4")
        plt.plot(self.dataIndent['displacement'], self.dataIndent["percent-cn-5"], label="CN=5")
        plt.plot(self.dataIndent['displacement'], self.dataIndent["percent-cn-6"], label="CN=6")
        plt.title(title)
        # Pote
        plt.ylabel("Atom percentage %")
        plt.xlabel("Displacement")
        plt.legend()
        plt.savefig(filename)
        plt.clf()

    # Gráfica un análisis de coordinación en función
    # de strain
    def plotCoordinationAnalysisV2(self, filename, title):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        if (not 'percent-cn-4' in self.dataIndent.columns):
            self.setCoordinationAnalysis()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent["percent-cn-3"], label="CN=3")
        plt.plot(self.dataIndent['strain'], self.dataIndent["percent-cn-4"], label="CN=4")
        plt.plot(self.dataIndent['strain'], self.dataIndent["percent-cn-5"], label="CN=5")
        plt.plot(self.dataIndent['strain'], self.dataIndent["percent-cn-6"], label="CN=6")
        plt.title(title)
        plt.ylabel("Atom percentage %")
        plt.xlabel("Strain")
        plt.legend()
        plt.savefig(filename)
        plt.clf()

    # Gráfica de la curva de energía potencial en función
    # de strain
    def plotPoteStrain(self, filename, title):
        if (not 'strain' in self.dataIndent.columns):
            self.setStrain()
        plt.style.use("ggplot")
        plt.plot(self.dataIndent['strain'], self.dataIndent["pote"])
        plt.title(title)
        plt.ylabel("Atom percentage %")
        plt.xlabel("Strain")
        plt.legend()
        plt.savefig(filename)
        plt.clf()
