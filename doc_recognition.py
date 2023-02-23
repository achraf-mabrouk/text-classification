
import pandas as pd
from .aws_api.converter_pdf_to_imgs import *
from .aws_api.class_get_aws_lines import *
from .document_classification import document_classification
from fuzzywuzzy import fuzz
from pdf2image import pdfinfo_from_path
import platform
from glob import glob

keywords = ["attestation sur l'honneur", 'audit energetique', 'preconisation', 'certificat qualibat',
            'date de devis', 'geoportail', 'impots', 'liste des entreprises', "préconisés", 'facture', 'inspection',
            "contrat d'assistante", 'synthese daudit', "refus expres"]


def apply_OCR(img_path):
    textract = GetMainLinesAws(image=img_path)
    df = textract.get_df()
    return df


def df_to_str(df):
    return ' '.join(line['text'].lower() for line in df)


def get_nb_pages(pdf_path):
    # [0] because glob returns a list
    pdf = glob(os.path.join(pdf_path, "*.pdf"))[0]
    if platform.system() == 'Windows':
        pdf_info = pdfinfo_from_path(
            pdf, userpw=None, poppler_path='D:\\poppler-0.68.0\\bin')
    else:
        pdf_info = pdfinfo_from_path(pdf, userpw=None, poppler_path=None)
    return pdf_info['Pages']


# check if the document contains certificat keyword
def is_certif(data):
    # return true if "RGE" or "CEE" in data
    text = " ".join([line['text'].lower() for line in data])
    return ["certificat", "rge"] or ["certificat", "cee"] in text


# finding keyword according to similarity score
def find_keyword(df, keyword):
    for i in range(len(df)//2):
        if fuzz.ratio(keyword, df[i]['text'].lower()) >= 80 or keyword in df[i]['text'].lower():
            return keyword
    return None


def get_doc_type(df):
    for keyword in keywords:
        doc_type = find_keyword(df, keyword)
        if doc_type != None:
            if doc_type == "date de devis":
                doc_type = "devis"
            return doc_type

    if is_certif(df):
        return "certificat"
    else:
        return "couldn't recognize the document type!"


def recognize_document(pdf_dir_path, doc_filename):
    # pdf to images
    Converter(folder_needed=pdf_dir_path)
    imgs_path = os.path.join(pdf_dir_path, "img")
    # get text of page 1 for doc type recognition
    df = apply_OCR(glob(os.path.join(imgs_path, "*_output_0.jpg"))[0])
    doc = {}
    doc['file_name'] = doc_filename
    # predict doc type (ML model)
    text = df_to_str(df)
    df1 = pd.DataFrame(data={'text': text,
                             'num_pages': get_nb_pages(pdf_dir_path)
                             }, index=[0])
    # double check with the keyword approach
    doc['doc_type'] = document_classification(df1)
    if doc['doc_type'] == "devis" or doc['doc_type'] == "facture":
        doc['doc_type'] = get_doc_type(df)
    
    return doc
