rm -rf ../docs
cd docs && reveal-md slides.md --css static/style.css --static=../../docs --static-dirs=static --absolute-url https://fchouteau.github.io/isae-practical-deep-learning
cp static/style.css ../../docs/css/style.css
sed -i 's|@import url(font/fira-sans/fira.css);|@import url(../static/font/fira-sans/fira.css);|g' ../../docs/css/style.css
sed -i 's|<link rel="stylesheet" href="./_assets/static/style.css" />|<link rel="stylesheet" href="./css/style.css" />|g' ../../docs/slides.html
cd ..
