# This Makefile only covers the compilation of the final PDF.
# The analysis workflow is described in workflow/Snakefile.

collect:
	# Collect all results exported to TeX
	cp results/numbers/* paper/numbers

	# Collect all generated figures
	cp \
	results/figures/fig4_ctf_metrics.png \
	results/figures/fig5_experiments.png \
	results/figures/fig6_spurious_coherence.png \
	results/figures/fig7_connectivity_example.png \
	results/figures/fig8_rift_onestim.png \
	results/figures/figBsupp_imcoh_recovery.png \
	results/figures/figCsupp_inferred_params.png \
	results/figures/figDsupp_rfs_models.png \
	results/figures/fig4supp_max_ratios.png \
	results/figures/figEsupp_rift_noise_floor_onestim.png \
	results/figures/figEsupp_rift_noise_floor_twostim.png \
	results/figures/figEsupp_rift_twostim.png \
	paper/figures

	# For figures that were created/postprocessed manually, collect the components
	cp \
	results/figures/components/* \
	paper/figure_components

paper: paper/main.tex paper/authors.tex $(wildcard paper/sections/*.tex) $(wildcard paper/supplementary/*.tex) $(wildcard paper/figures/*.png) $(wildcard paper/numbers/*.tex) paper/supplementary.aux
	cd paper && pdflatex main.tex && bibtex main.aux && pdflatex main.tex && pdflatex main.tex

supplementary: paper/supplementary.tex paper/authors.tex $(wildcard paper/supplementary/*.tex) $(wildcard paper/figures/*.png) $(wildcard paper/numbers/*.tex)
	cd paper && pdflatex supplementary.tex && bibtex supplementary.aux && pdflatex supplementary.tex && pdflatex supplementary.tex
