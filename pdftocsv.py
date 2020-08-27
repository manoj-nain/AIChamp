#Ceating function to extract data from PDF using Pdfminer
#Updated to detect spaces 

import io

import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

# Perform layout analysis for all text
laparams = pdfminer.layout.LAParams()
setattr(laparams, 'all_texts', True)

def extract_text_from_pdf(pdf_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    if text:
        return text 

#Saving to csv
import csv
import os

def export_as_csv(pdf_path, csv_path):

    with open(csv_path, 'a',encoding= "utf-8",newline="\n") as csv_file:
        writer = csv.writer(csv_file)
        text = extract_text_from_pdf(pdf_path)
        text = text[:]
        words = text.split()
        writer.writerow([i," ".join(words)])
        

            
if __name__ == '__main__':
    
    for i in range(1,51):    
        profile = "Profile" + str(i)
        pdf_path = profile+".pdf"
        csv_path = 'Profile1.csv'
        export_as_csv(pdf_path, csv_path)