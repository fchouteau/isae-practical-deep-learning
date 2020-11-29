build-docs-pdf:
	cd docs && reveal-md slides.md --print slides.pdf --css static/style.css

build-docs-serve:
	cd docs && watch -n 10 'reveal-md slides.md -w --css static/style.css --disable-auto-open'

build-docs-site:
	rm -rf ../docs
	cd docs && reveal-md slides.md --css static/style.css --static=../../docs --static-dirs=static --absolute-url https://fchouteau.github.io/isae-practical-deep-learning
	cp static/style.css ../../docs/css/style.css
	sed -i 's|<link rel="stylesheet" href="./_assets/static/style.css" />|<link rel="stylesheet" href="./css/style.css" />|g' ../../docs/slides.html
	cd ..

build-student-notebooks:
	jupytext --to notebook 1_hands_on.py --output ../1_hands_on.py.ipynb
	jupytext --to notebook 2_dealing_with_class_imbalance.py --output ../2_dealing_with_class_imbalance.ipynb
	jupytext --to notebook 3_sliding_window.py --output ../3_sliding_window.ipynb
