from pathlib import Path
ScriptDir = Path(__file__).parent
from NewSpec.Core.Record import loadRecord, correctDC
from NewSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from NewSpec.Core.SaveIntegral import SaveIntegral


record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
record = correctDC(record)
print(SaveIntegral.Energy(record))

record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
print(SaveIntegral.Energy(record))

from BatSpec.QtApp.Visualize import update_function, update_spec2d, run_visualizer
update_function("record", record)
run_visualizer()




