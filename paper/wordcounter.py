"""
Count the number of words in a PDF file.
"""

import sys
import os
import re
import PyPDF2
import typer
from datetime import datetime

chapters = [
    "Cover/Preliminaries",
    "1. introduction",
    "2. state of the art",
    "3. design of the solution",
    '4. methodology',
    "5. results",
    "6. conclusions",
    "regulatory framework",
    "bibliography",
    'appendix',
]


def main(pdf_file: str, output_file: str = "wordcount.csv", save: bool = False):
    """
    Count the number of words in a PDF file.
    """
    pdf_file = os.path.abspath(pdf_file)
    if not os.path.exists(pdf_file):
        typer.echo(f"File not found: {pdf_file}")
        raise typer.Exit(code=1)
    if not os.path.isfile(output_file) and save:
        with open(output_file, "w") as f:
            f.write("file,date,words\n")

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    typer.echo(f"Number of pages: {num_pages}")
    num_words = 0
    chapter_count = {}
    pages_count = {}
    chapter_counter = 0
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if (
            text[:50]
            .lower()
            .strip()
            .startswith(chapters[min(chapter_counter + 1, len(chapters) - 1)])
        ):
            chapter_counter += 1
        if chapters[chapter_counter] not in chapter_count:
            chapter_count[chapters[chapter_counter]] = [0, []]
        words = re.findall(r"[^\W_]+", text, re.MULTILINE)
        chapter_count[chapters[chapter_counter]][0] += len(words)
        chapter_count[chapters[chapter_counter]][1] += [page_num]
        pages_count[page_num] = len(words)
        num_words += len(words)
    typer.echo(f"Number of words: {num_words}")
    words_per_chapter = "\n\t".join(
        [
            f"{chapter.title()}: {words} words, {len(pages)} pages [{pages[0]+1}-{pages[-1]+1}]"
            for chapter, (words, pages) in chapter_count.items()
        ]
    )
    words_per_chapter = re.sub(
        "regulatory framework", "End Matters", words_per_chapter, flags=re.IGNORECASE
    )
    typer.echo(f"Word,Pages per chapter:\n\t{words_per_chapter}")
    typer.echo(
        f"Page with most words: {max(pages_count, key=pages_count.get)+1} ({pages_count[max(pages_count, key=pages_count.get)]} words)"
    )
    typer.echo(
        f"Page with least words: {min(pages_count, key=pages_count.get)+1} ({pages_count[min(pages_count, key=pages_count.get)]} words)"
    )
    num_pages = sum(
        1
        for i, p in pages_count.items()
        if p > 20
        and i not in chapter_count[chapters[0]][1]
        and i not in chapter_count[chapters[-1]][1]
    )
    typer.echo(
        f"Avg words per page (excluding preliminaries and bibliography): {num_words/num_pages:.2f}"
    )

    if save:
        with open(output_file, "a") as f:
            f.write(
                f'{os.path.basename(pdf_file)},{datetime.now().strftime("%Y-%m-%dT%H:%M:00")},{num_words},{num_pages} pages\n'
            )


if __name__ == "__main__":
    typer.run(main)
