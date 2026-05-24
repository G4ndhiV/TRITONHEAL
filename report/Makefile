MAIN = main
LATEXMK = latexmk

pdf:
	$(LATEXMK) -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -C
	rm -f *.fdb_latexmk *.fls

.PHONY: pdf clean
