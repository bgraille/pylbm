# Makefile
#
# convert all .md files into .pdf using xelatex

SOURCE_DOCS := $(wildcard *.md)

EXPORTED_DOCS_PDF=$(SOURCE_DOCS:.md=.pdf)

PANDOC_OPTIONS=-s --number-sections \
	--filter pandoc-crossref \
	--filter pandoc-citeproc

PANDOC_PDF_OPTIONS=--template=template \
	--pdf-engine=xelatex

%.pdf : %.md
	pandoc $(PANDOC_OPTIONS) $(PANDOC_PDF_OPTIONS) -o $@ $<

.PHONY: all clean

pdf : $(EXPORTED_DOCS_PDF)
all : $(EXPORTED_DOCS_PDF)

clean:
	- rm $(EXPORTED_DOCS_PDF)
