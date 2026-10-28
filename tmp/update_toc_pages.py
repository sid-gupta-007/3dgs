from pathlib import Path
import re
from docx import Document
from pypdf import PdfReader

root=Path(r'C:\Users\Siddharth Gupta\Desktop\CODES-ALL\3dg')
dp=root/'output'/'PanoGS_Project_Report.docx'; fp=root/'output'/'PanoGS_Project_Report.pdf'
d=Document(dp); pdf=PdfReader(str(fp))
pages=[' '.join((p.extract_text() or '').split()) for p in pdf.pages]
def norm(s): return ' '.join(re.sub(r'[^a-z0-9]+',' ',s.casefold()).split())
norm_pages=[norm(s) for s in pages]
updated=[]
for p in d.paragraphs:
    if not p.text.endswith('\tXX'): continue
    raw=p.text.rsplit('\t',1)[0]
    label=raw.replace('• ','',1).replace('   o ','',1)
    if label.lower().startswith('chapter '):
        n=label.split(':',1)[0].split()[-1]; key=f'chapter {n} {label.split(":",1)[1]}'
        start=7
    elif label.startswith('Appendix '):
        key=label.split(':',1)[1].strip(); key=f'appendix {label.split()[1][0]} {key}'
        start=22
    elif label=='References': key='references'; start=22
    else:
        key=label
        start=7
        if label.startswith('A.') or label.startswith('Appendix B'): start=23
    needle=norm(key)
    found=next((i+1 for i,t in enumerate(norm_pages[start:],start=start) if needle and needle in t),None)
    if found is None:
        raise RuntimeError(f'Could not locate TOC entry in PDF: {label!r} ({key!r})')
    p.runs[0].text=raw+'\t'+str(found)
    updated.append((label,found))
d.save(dp)
print(f'Updated {len(updated)} contents entries')
for row in updated: print(row[0],row[1])
