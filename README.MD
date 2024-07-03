## Installation & Setup:
### Example
Check Run_Example_Small.ipynb for full setup and usage.
For a more detailed installation continue below.

### Conda environment
 - conda env create -f environment.yml

### Download required Data
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
### Example
Check Run_Example_Small.ipynb for full setup and usage

### Self-Setup
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

