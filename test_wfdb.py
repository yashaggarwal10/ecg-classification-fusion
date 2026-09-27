import wfdb

path = "data/ptbxl/records100/00000/00013_lr"

print("Trying to load:", path)

record = wfdb.rdrecord(path)

print("Loaded!")
print("Shape:", record.p_signal.shape)
