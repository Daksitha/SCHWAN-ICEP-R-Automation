
This repository contains the initial code base for our ICMI 2024 paper:

**Title:** Towards Automated Annotation of Infant-Caregiver Engagement Phases with Multimodal Foundation Models  
**Authors:** Daksitha Senel Withanage Don, Dominik Schiller, Tobias Hallmen, Silvan Mertes, Tobias Baur, Florian Lingenfelser, Mitho Müller, Lea Kaubisch, Prof. Dr. Corinna Reck, Elisabeth André  
**Conference:** [ICMI 2024](https://doi.org/10.1145/3678957.3685704)  
**DOI:** [10.1145/3678957.3685704](https://doi.org/10.1145/3678957.3685704)  

This repository supports the NOVA-DISCOVER ecosystem, designed for behavior analysis. Key feature extraction processes, including **DinoV2** and **Wav2Vec2-BERT**, are already integrated into the **DISCOVER** module, functioning as the back-end for the **NOVA** user interface.

## Repositories  

- **[DISCOVER](https://github.com/hcmlab/discover):** Backend module for behavior analysis.  
- **[NOVA](https://github.com/hcmlab/nova):** User interface for behavior analysis and annotation visualization.  

## Key Features  
- **Foundation Models for Feature Extraction**:  
  - DinoV2 for visual features  
  - Wav2Vec2-BERT for audio features  

- **Automated Annotation Pipeline**:  
  - Bidirectional LSTM for complex engagement phases  
  - Linear classifiers for unimodal feature encodings  

- **ICEP-R Label Integration**:  
  - Annotated dataset for further research  
  - Supports temporal alignment with video and audio signals  

## Installation  

Clone the repository and install the required dependencies:  

```bash  
git clone https://github.com/hcmlab/icep-r-automation.git  
cd icep-r-automation  
pip install -r requirements.txt  
```  

## Usage  

1. **Feature Extraction:**  
   Use pre-trained models (DinoV2 and Wav2Vec2-BERT) to extract features from video and audio data.  

2. **Model Training and Evaluation:**  
   Train bidirectional LSTM and linear models on extracted features for phase annotation.  

3. **Integration with NOVA-DISCOVER Ecosystem:**  
   Deploy trained models within the DISCOVER backend and visualize results in the NOVA UI.  

## Citation  

If you use this repository, please cite our paper:  

```bibtex  
@inproceedings{10.1145/3678957.3685704,
author = {Withanage Don, Daksitha Senel and Schiller, Dominik and Hallmen, Tobias and Mertes, Silvan and Baur, Tobias and Lingenfelser, Florian and M\"{u}ller, Mitho and Kaubisch, Lea and Reck, Prof. Dr. Corinna and Andr\'{e}, Elisabeth},
title = {Towards Automated Annotation of Infant-Caregiver Engagement Phases with Multimodal Foundation Models},
year = {2024},
isbn = {9798400704628},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3678957.3685704},
doi = {10.1145/3678957.3685704},
booktitle = {Proceedings of the 26th International Conference on Multimodal Interaction},
pages = {428–438},
numpages = {11},
keywords = {Automated annotation, Caregiver-infant interaction, Developmental psychology, Self-supervised learning, Still Face Paradigm},
location = {San Jose, Costa Rica},
series = {ICMI '24}
}
```  

## Contact  

For any questions or inquiries, please contact:  
**Daksitha Senel Withanage Don**  
Email: daksitha.withanage.don@uni-a.de
