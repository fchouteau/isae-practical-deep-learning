serve-slides:
	cd docs; reveal-md . \
				-w \
				--disable-auto-open \
				--template static/template/reveal.html \
				--static-dirs static

build-slides:
	rm -rf ../docs || true
	cd docs; reveal-md . \
				--static ../../docs \
				--template static/template/reveal.html \
				--static-dirs static
	rm ../docs/index.html && mv ../docs/slides.html ../docs/index.html

build-pdf:
	docker run --rm -t --net=host -v /home/fchouteau/classes/isae-practical-deep-learning/teacher/docs:/slides astefanutti/decktape http://192.168.86.105:1948/slides.md#/ slides.pdf

build-student-notebooks:
	jupytext --to notebook 1_hands_on.py --output ../1_hands_on.ipynb
	jupytext --to notebook 2_dealing_with_class_imbalance.py --output ../2_dealing_with_class_imbalance.ipynb
	jupytext --to notebook 3_sliding_window.py --output ../3_sliding_window.ipynb
