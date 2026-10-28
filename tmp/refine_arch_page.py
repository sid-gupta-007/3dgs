from pathlib import Path
from docx import Document
root=Path(r'C:\Users\Siddharth Gupta\Desktop\CODES-ALL\3dg')
path=root/'output'/'PanoGS_Project_Report.docx'; d=Document(path)
for p in d.paragraphs:
    if p.text=='5.5 Project Screenshots: Architecture and Interface':
        p.runs[0].text='5.5 PanoGS System Architecture Diagram'
    elif p.text.startswith('Insert a screenshot of the PanoGS interface'):
        p.runs[0].text='This diagram summarizes the local panorama reconstruction pipeline and the separate optional World Labs hosted-generation workflow.'
    elif p.text=='Figure 5.2  Project interface or architecture screenshot':
        p.runs[0].text='Figure 5.2  PanoGS architecture: local pipeline and optional hosted path'
    elif p.text.endswith('\t14') and '5.5 Project Screenshots: Architecture and Interface' in p.text:
        p.runs[0].text=p.text.replace('5.5 Project Screenshots: Architecture and Interface','5.5 PanoGS System Architecture Diagram')
    elif p.text=='Figure 5.2  Project interface or architecture screenshot':
        p.runs[0].text='Figure 5.2  PanoGS architecture: local pipeline and optional hosted path'
d.save(path); print(path)
