# ISAE-Practical-Deep-Learning, teacher resources

Contain the necessary scripts and tools to:
- generate dataset
- build the slides/docs

## How-to

- [build the documentation website](./build_docs_site.sh)
- [generate the slides in PDF](./build_docs_pdf.sh)
- [Live edit the slides](./build_docs_serve.sh)

- [Download Data](./get_data.py)
- [Explore the dataset](./data_exploration.py)
- [Generate Dataset Small](./generate_student_dataset_toy.py)
- [Generate Dataset Large](./generate_student_dataset_large.py)

## Student notebooks

Student notebooks are generated using jupytext python files. Solutions cells are not exported if they are tagged as "exercise"

See [build student notebook](./build_student_notebooks.sh) script (you need to `pip install nb-filter-cells`)

## Installation

install pytorch

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## About

The idea of this course is based on the [excellent tutorial given by Martin Gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-planespotting)

Khumeia was designed in 2018 to be an end-to-end framework for data processing. However, pedagogical considerations led it to be scrapped in 2019 and later, only to be used as dataset generation helper to generate the students' datasets.
