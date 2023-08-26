"""
Count the number of words in a PDF file.
"""

import sys
import os
import re
import PyPDF2
import typer
from datetime import datetime

def main(pdf_file:str, output_file:str='wordcount.csv'):
    """
    Count the number of words in a PDF file.
    """
    pdf_file = os.path.abspath(pdf_file)
    if not os.path.exists(pdf_file):
        typer.echo(f"File not found: {pdf_file}")
        raise typer.Exit(code=1)
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('file,date,words\n')

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    typer.echo(f"Number of pages: {num_pages}")

    num_words = 0
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        text = re.sub(r'\s+', ' ', text)
        num_words += len(text.split())

    typer.echo(f"Number of words: {num_words}")

    with open(output_file, 'a') as f:
        f.write(f'{os.path.basename(pdf_file)},{datetime.now().strftime("%Y-%m-%dT%H:%M:00")},{num_words}\n')

if __name__ == "__main__":
    typer.run(main)