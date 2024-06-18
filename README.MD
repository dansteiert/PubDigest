# PubDigest

## Abstract
### Background: 
In the modern era, the growth of scientific literature presents a daunting challenge for researchers to keep informed of advancements across multiple disciplines.
### Objective: 
We apply natural language processing (NLP) to design a software tool that combs PubMed literature, aiming to pinpoint potential drugs that could be repurposed.
### Methods: 
Using NLP, especially term associations through word embeddings, we explored unrecognized relationships between drugs and diseases. To illustrate the utility of this software, we focused on chronic thromboembolic pulmonary hypertension (CTEPH), a rare disease with limited numbers of scientific publications.
### Results: 
Our literature analysis identified key clinical features linked to CTEPH using term frequency-inverse document frequency (TF-IDF), a technique measuring a term's significance in a text corpus. This allowed us to map related diseases. One standout was venous thrombosis (VT), which showed strong textual links with CTEPH. Looking deeper, we discovered potential drugs for CTEPH by analyzing literature on both CTEPH and VT. An intriguing find were benzofuran-derivatives, particularly amiodarone, which displayed potential anti-thrombotic properties. In vitro tests confirmed its ability to reduce platelet aggregation significantly by 68% (p=0.018). However, real-world clinical data indicated that CTEPH patients on amiodarone faced a significant 15.9% higher mortality risk (p < 0.001).
### Conclusion: 
While NLP offers an innovative approach to interpret scientific literature, especially for drug repurposing, it's crucial to tread with caution. Our exploration with benzofuran-derivatives and CTEPH underscores this point. Thus, blending NLP with hands-on experiments and clinical trials can pave the way for safer drug repurposing, especially for rare diseases like CTEPH.


## Installation & Setup:
### Conda environment
 - conda env create -f environment.yml


### Base data - Manual (should not be necessary)
- Download drug data from CHEMBLE: 
  - https://www.ebi.ac.uk/chembl/g/#browse/drugs 
    - select all/ download as .csv; 
    - save csv file into the base_data directory as CHEMBL_medication_names.csv
    - call python misc.prep_CHEMBL()

- Donwload USAN stems:
  - https://www.ama-assn.org/about/united-states-adopted-names/united-states-adopted-names-approved-stems
    - save the xlsx file at base_data, called USAN_Stems.xlsx
    - call python misc.prep_USAN()


- Download a list of cities with their coordinates
  - https://simplemaps.com/data/world-cities 
    - save the list to base_data as worldcities.csv
    - This might need curation to achive better results
      - Some additional help can be the mapping provided in Configs Location

## Running
- check the configuration files in Configurations
  - add your Email Address in the PMC config. 
    - This Email is forwarded to the Administrators of the ENTREZ API.
  - Setup your Query config
  - Setup System config
  - Setup Workflow config
  - Setup Config_Main
    - Select here the desired configuations
  - Browse through the other configs if you desire to change them
- Run the program:
  - python main.py your_main_config.yml 

