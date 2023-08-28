# makefile for commands involving processing the LaTeX source

TEX_FILE = main
OUT_PDF = paper.pdf

# Default target 'help' displays the help message
.DEFAULT_GOAL := help

PHONY: help clean clean-aux compile count
	

clean: ## Clean up the directory of auxiliary Tex files excluding the PDF
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.dvi *.ps *.run.xml *.bcf *.synctex.gz *.blg *.bbl *.log *.aux *.out *.toc *.dvi *.ps \
		*.bcf *.run.xml *.synctex.gz pdfa.xmpi *.lof *.lot *latexmk *.fls *.fdb_latexmk *.tdo

clean-all: clean ## Clean up the directory of auxiliary Tex files including the PDF
	ls sections/* | grep -xv ".*.tex" | xargs rm

build: ## Compile and build the paper with pdflatex, biblatex
	pdflatex $(TEX_FILE).tex
	biber $(TEX_FILE) -e 
	pdflatex $(TEX_FILE).tex
	pdflatex $(TEX_FILE).tex
	make clean

clear_cache: ## Clear the cache of biber
	rm -rf `biber --cache`

count: ## Count the number of words in the paper, display the result, and save it to a file in append mode
	/opt/homebrew/Caskroom/miniforge/base/envs/tf-env/bin/python wordcounter.py main.pdf --save

push: count ## Push the paper to the repository
	make clean-all
	cd .. && git add paper && git commit -m "Updated paper" && git push

help: ## self documenting Makefile based on http: marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

