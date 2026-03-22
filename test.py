
import os
import sys

# Root path inferred from the parent directory of this settings file
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

# Build all paths from ROOT_PATH
DLL_PATH = os.path.join(ROOT_PATH, "thermodynamics/Runtime/x64") + os.sep

# Important to do this BEFORE importing an apiwrapper.py module! Otherwise the DLLs will can not find the configs/!
os.chdir(DLL_PATH)
os.add_dll_directory(os.getcwd())
sys.path.append(ROOT_PATH)

from apiwrapper import *

material_type = JMP_MATERIAL_NICKEL_BASED_SUPERALLOY
composition = [58, 1, 1, 40]
elements = ["Ni", "Cr", "Al", "O"]
unit = JMP_COMPOSITION_UNIT_ATOMIC_PERCENT
temperature = 1100.0
unit_temperature = JMP_TEMPERATURE_UNIT_CELSIUS
calculation_type = JMP_SOLVER_CALCULATION_SINGLE_POINT

jmpSetMaterialType(material_type)
jmpSetAlloyElements(elements)
jmpSetCompositionUnit(unit)
jmpSetAlloyComposition(composition)
jmpSetSolverCalculationType(calculation_type)
jmpSetTemperatureUnit(unit_temperature)
jmpSetSolverTemperature(temperature)
jmpSetDefaultPhases()
jmpRunSolverCalculation()

# Read the austenite composition from "summary.out". The phase elements can be ignored as they match the
# alloy elements.
elements1, composition1 = jmpGetPhaseCompositionAt("GAMMA", temperature)
elements2, composition2 = jmpGetPhaseCompositionAt("MO_B2", temperature)
elements3, composition3 = jmpGetPhaseCompositionAt("SPINEL_AB2O4", temperature)

print(elements1, composition1)
print(elements2, composition2)
print(elements3, composition3)

temperature = 1100.0
composition = [60, 3, 3, 34]
elements = ["Ni", "Cr", "Al", "O"]
jmpSetMaterialType(material_type)
jmpSetAlloyElements(elements)
jmpSetCompositionUnit(unit)
jmpSetAlloyComposition(composition)
jmpSetSolverCalculationType(calculation_type)
jmpSetTemperatureUnit(unit_temperature)
jmpSetSolverTemperature(temperature)
jmpSetDefaultPhases()
jmpRunSolverCalculation()


elements1, composition1 = jmpGetPhaseCompositionAt("GAMMA", temperature)
print(elements1, composition1)
