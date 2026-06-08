MAIN = poster
LATEX = lualatex -interaction=nonstopmode -halt-on-error

pdf:
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f *.aux *.log *.nav *.snm *.toc *.out *.fdb_latexmk *.fls

.PHONY: pdf clean
