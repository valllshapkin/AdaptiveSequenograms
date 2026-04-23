from pathlib import Path
ScriptDir = Path(__file__).parent
from BatSpec.Work.Record import loadRecord
from BatSpec.Work.SaveIntegral import SaveIntegralEnergy

record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
print(SaveIntegralEnergy(record))