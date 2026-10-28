from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from xml.sax.saxutils import escape

root=Path(r'C:\Users\Siddharth Gupta\Desktop\CODES-ALL\3dg')
src=root/'output'/'PanoGS_Project_Report.docx'; dst=root/'output'/'PanoGS_Project_Report.pdf'
doc=Document(src); styles=getSampleStyleSheet()
base=ParagraphStyle('Body',parent=styles['BodyText'],fontName='Times-Roman',fontSize=10.5,leading=13.2,alignment=TA_JUSTIFY,spaceAfter=6,firstLineIndent=16)
center=ParagraphStyle('Center',parent=base,alignment=TA_CENTER,firstLineIndent=0)
right=ParagraphStyle('Right',parent=base,alignment=TA_RIGHT,firstLineIndent=0)
h1=ParagraphStyle('H1x',parent=styles['Heading1'],fontName='Times-Bold',fontSize=15,leading=18,alignment=TA_CENTER,spaceBefore=10,spaceAfter=8,keepWithNext=True)
h2=ParagraphStyle('H2x',parent=styles['Heading2'],fontName='Times-Bold',fontSize=12,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True)
h3=ParagraphStyle('H3x',parent=h2,fontSize=11,leading=13)
small=ParagraphStyle('Smallx',parent=base,fontSize=8.5,leading=10,firstLineIndent=0)
def para_html(p):
    if '\t' in p.text:
        label,number=p.text.rsplit('\t',1)
        dots=max(5,92-len(label)-len(number))
        return escape(label)+' <font color="#666666">'+('.'*dots)+'</font> '+escape(number)
    s=''
    for r in p.runs:
        t=escape(r.text).replace('\n','<br/>')
        if r.bold:t=f'<b>{t}</b>'
        if r.italic:t=f'<i>{t}</i>'
        s+=t
    return s or ' '
def choose(p):
    n=p.style.name if p.style else 'Normal'
    if '\t' in p.text:
        return ParagraphStyle('TOCx',parent=base,fontSize=8.8,leading=10.5,spaceAfter=1.3,firstLineIndent=0,leftIndent=0)
    if n=='Title':return ParagraphStyle('titlex',parent=center,fontName='Times-Bold',fontSize=21,leading=25,spaceAfter=17)
    if n=='Heading 1':return h1
    if n=='Heading 2':return h2
    if n=='Heading 3':return h3
    if n=='Small Note':return small
    if n=='Caption':return small
    if 'Bullet' in n or 'List' in n:return ParagraphStyle('listx',parent=base,leftIndent=18,firstLineIndent=-10,spaceAfter=3)
    if p.alignment==1:return center
    if p.alignment==2:return right
    return base
def page(canvas,document):
    canvas.saveState(); w,h=letter
    if document.page>1:
        canvas.setFont('Times-Roman',8); canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawRightString(w-0.85*inch,h-0.42*inch,'PanoGS | Project Report')
        canvas.drawCentredString(w/2,0.42*inch,str(document.page))
    canvas.restoreState()
story=[]; rel=doc.part.related_parts
body=doc.element.body
for child in body.iterchildren():
    if child.tag==qn('w:p'):
        # use matching python-docx paragraph wrapper
        from docx.text.paragraph import Paragraph as DocxParagraph
        p=DocxParagraph(child,doc)
        has_break=bool(child.xpath('.//w:br[@w:type="page"]'))
        if has_break:
            story.append(PageBreak()); continue
        # preserve the report's inline diagrams
        for blip in child.xpath('.//a:blip'):
            rid=blip.get(qn('r:embed'))
            if rid and rid in rel:
                image_path=root/'tmp'/f'pdf_embedded_{rid}.png'; image_path.write_bytes(rel[rid].blob)
                im=Image(str(image_path),width=6.2*inch,height=2.31*inch); im.hAlign='CENTER'; story.extend([Spacer(1,5),im,Spacer(1,5)])
        txt=para_html(p)
        if txt.strip():
            try: story.append(Paragraph(txt,choose(p)))
            except Exception: story.append(Paragraph(escape(p.text),choose(p)))
    elif child.tag==qn('w:tbl'):
        from docx.table import Table as DocxTable
        t=DocxTable(child,doc); data=[]
        is_slot=any('PASTE PROJECT IMAGE HERE' in cell.text or cell._tc.xpath('.//a:blip/@r:embed') for row in t.rows for cell in row.cells)
        for ri,row in enumerate(t.rows):
            cells=[]
            for cell in row.cells:
                image_ids=cell._tc.xpath('.//a:blip/@r:embed') if is_slot else []
                if image_ids:
                    rid=image_ids[0]; image_path=root/'tmp'/f'pdf_embedded_{rid}.png'; image_path.write_bytes(rel[rid].blob)
                    cells.append(Image(str(image_path),width=6.15*inch,height=3.54*inch)); continue
                value='<br/>'.join(escape(p.text).replace('\n','<br/>') for p in cell.paragraphs)
                color=colors.HexColor('#777777') if is_slot else (colors.white if ri==0 else colors.black)
                cells.append(Paragraph(value or ' ',ParagraphStyle('cell',parent=small,fontName='Times-Bold' if ri==0 and not is_slot else 'Times-Roman',textColor=color,alignment=TA_CENTER if is_slot else 0)))
            data.append(cells)
        if data:
            n=len(data[0]); widths=[6.5*inch/n]*n
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='CENTER')
            if is_slot:
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.white),('BOX',(0,0),(-1,-1),0.8,colors.HexColor('#7B8DA3')),('DASH',(0,0),(-1,-1),3,2),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            else:
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203957')),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F5F7FA')),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#BAC5D2')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([table,Spacer(1,7)])
SimpleDocTemplate(str(dst),pagesize=letter,rightMargin=.85*inch,leftMargin=.85*inch,topMargin=.72*inch,bottomMargin=.7*inch,title='PanoGS Project Report',author='PanoGS Project Team').build(story,onFirstPage=page,onLaterPages=page)
print(dst)
