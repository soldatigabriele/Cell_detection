import os
import sys
import pandas as pd 

DIR = 'immagini_da_estrarre_roi'
DIVISE = os.path.join(DIR,'output')
GLCM_FILE = os.path.join(DIVISE,'GLCM.csv')
GLCM_TXT = os.path.join(DIVISE,'GLCM.txt')
MEASURES_FILE = os.path.join(DIVISE,'measures.csv')
FILE_NAME_XLSX = os.path.join(DIVISE,'summary_rois.xlsx')

f = open(GLCM_FILE, 'w')
TEXT = open(GLCM_TXT,'r').readlines()
for line in TEXT:
    line = line.split(',')
    line[5] = line[5][:-1]
    print(line)
    with open(GLCM_FILE, 'a') as f:
      f.write('{},{},{},{},{},{}\n'.format(line[0],line[1],line[2],line[3],line[4],line[5]))


# get data to be appended
df  = pd.read_excel(FILE_NAME_XLSX)
df2 = pd.read_csv(MEASURES_FILE)
df3 = pd.read_csv(GLCM_FILE)

print(df.head())
df["Mean"] = df2["Mean"]
df["Area"] = df2["Area"]
df["StdDev"] = df2["StdDev"]
df["Mode"] = df2["Mode"]
df["Min"] = df2["Min"]
df["Max"] = df2["Max"]
df["X"] = df2["Y"]
df["XM"] = df2["XM"]
df["YM"] = df2["YM"]
df["Perim."] = df2["Perim."]
df["Width"] = df2["Width"]
df["Height"] = df2["Height"]
df["Major"] = df2["Major"]
df["Minor"] = df2["Minor"]
df["Angle"] = df2["Angle"]
df["Circ."] = df2["Circ."]
df["Feret"] = df2["Feret"]
df["IntDen"] = df2["IntDen"]
df["Median"] = df2["Median"]
df["Skew"] = df2["Skew"]
df["Kurt"] = df2["Kurt"]
df["RawIntDen"] = df2["RawIntDen"]
df["FeretAngle"] = df2["FeretAngle"]
df["MinFeret"] = df2["MinFeret"]
df["AR"] = df2["AR"]
df["Round"] = df2["Round"]

df["Angular Second Moment"] = df3["Angular Second Moment"]
df["Contrast"] = df3["Contrast"]
df["Correlation"] = df3["Correlation"]
df["Inverse Difference Moment"] = df3["Inverse Difference Moment"]
df["Entrop"] = df3["Entrop"]
print(df.head())

excel_writer = pd.ExcelWriter(FILE_NAME_XLSX)
df.to_excel(excel_writer, 'summary_rois.csv', index=False)
excel_writer.save() 

if os.path.exists(MEASURES_FILE):
    os.unlink(MEASURES_FILE)
if os.path.exists(GLCM_TXT):
    os.unlink(GLCM_TXT)
if os.path.exists(GLCM_FILE):
    os.unlink(GLCM_FILE)
'''
# define what sheets to update
to_update = {"summary_rois.csv": df_append}

# load existing data
file_name = 'manipur1.xlsx'
excel_reader = pd.ExcelFile(file_name)

# write and update
excel_writer = pd.ExcelWriter(file_name)

for sheet in excel_reader.sheet_names:
    sheet_df = excel_reader.parse(sheet)
    append_df = to_update.get(sheet)

    if append_df is not None:
        sheet_df = pd.concat([sheet_df, append_df], axis=1)

    sheet_df.to_excel(excel_writer, sheet, index=False)

excel_writer.save()       
'''
