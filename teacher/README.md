# ISAE-Practical-Deep-Learning, teacher resources

Contain the necessary scripts and tools to:
- generate dataset
- build the slides/docs

## How-to

- [build the documentation website](./build_docs_site.sh)
- [generate the slides in PDF](./build_docs_pdf.sh)
- [live edit the slides](./build_docs_serve.sh)
- [build student notebooks](./build_student_notebooks.sh)


## Student notebooks

Student notebooks are generated using jupytext python files. Solutions cells are not exported if they are tagged as "exercise"

See [build student notebook](./build_student_notebooks.sh) script (you need to )

## Installation of dev environment

you should have pytorch torchvision and skorch installed

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

to generate student notebooks you need `nb-filters-cells`:  `pip install nb-filter-cells`

## About

The idea of this course is based on the [excellent tutorial given by Martin Gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-planespotting)
