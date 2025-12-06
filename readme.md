# Predicting the Onset of Dementia Using Neural Networks: A Deep Learning Approach to Neuroimaging Analysis

Research Report: (insert link)
Abstract: (insert)


**This repo features multiple CNN structures preprocessed and trained to classify multiple indications of Alzheimer's Disease / Dementia.**

- MRI: Classifies multiple stages of Dementia into ['Non Demented', 'Very Mild', 'Mild', 'Moderate'].
- Biomarkers: Classifies patient records into ['No Dementia', 'Dementia'] based on corresponding preexisting health conditions.
- Speech: (insert)


Download Operation:
```
git clone git@github.com:abrizu/480-NN-Project-MRI.git
python -m venv .venv
.venv/scripts/activate
pip freeze > requirements.txt
```

**Due to size, data for both models must be downloaded separately.**

External Download - 
- Download the MRI Dataset from https://www.kaggle.com/datasets/ninadaithal/imagesoasis.
- Download the Biomarkers Dataset from https://www.kaggle.com/datasets/snmahsa/comprehensive-health-and-brain-imaging-dataset.
- Download the Speech Dataset from (insert link).

Extraction Operations - 
- Extract the MRI Dataset into ./MRIData folder. 
- Extract the Biomarker Dataset into ./BMData folder.
- Extract the Speech Dataset into ./SPData folder.

To Execute:
```
py functions/(mri.py, biomarkers.py, (insert))
```
