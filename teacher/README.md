# isae-practical-deep-learning Teacher resources

## Features

## Maintainers

## Usage

Inspiration taken from
https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-planespotting

TODO: Faire le readme

Package dans `src/`
Install gcp dans `gcp/`

Pour installer le dernier commit: `pip install https://storage.googleapis.com/isae-deep-learning/khumeia-0.1.0.dev0+master.tar.gz`

Site public: https://github.com/fchouteau/isae-practical-deep-learning

Site datalab: http://localhost:2224/bucket/gitlab-generated/tp_isae/master

Voir notebooks pour les exemples d'utilisation

Note: Khumeia est avant tout un exercice d'écrire d'un framework deep-image-sps-like-light.
Note: Peut être réutilisation à but de formation interne / "bac à sable" ? 

Build doc: installer deps dans requirements.txt puis `bash scripts/deploy_github_pages.sh`

## Documentation / Slides

In `docs/`

To watch slides for live editing:
```bash
cd docs && reveal-md slides.md -w --css static/css/theme.css
```

To build static site for documentation
```bash
cd docs && reveal-md slides.md --css static/css/theme.css --static=../../docs --static-dirs=static --absolute-url https://fchouteau.github.io/isae-practical-deep-learning
```

To build pdf version of slides
bash
cd docs && reveal-md slides.md --print slides.pdf --css static/css/theme.css
```
