from NewSpec.Core.Record import loadRecord
from pathlib import Path
import torch
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"

PATH = "/MainData/Repo/valllshapkin/MonoBat/BatSpec/Resources"
for i in glob(f"{PATH}/**/*.wav"):
    record = loadRecord(Path(i), device=device)
    print(record.end)
