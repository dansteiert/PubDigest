import urllib.request
import os
import shutil
from misc.prep_USAN_stem import prep_usan
def get_base_data():
    if not os.path.isfile(os.path.join(os.getcwd(), "base_data", "USAN_Stems.xlsx")):
        ## Download USAN Stems
        url = "https://www.ama-assn.org/system/files/stem-list-cumulative.xlsx"
        req = urllib.request.Request(url)
        req.add_header('User-Agent',
                       'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7')
        response = urllib.request.urlopen(req)
        data = response.read()  # a `bytes` object
        with open(os.path.join(os.getcwd(), "base_data", "USAN_Stems.xlsx"), "wb") as f:
            f.write(data)
        prep_usan()

    if not os.path.isfile(os.path.join(os.getcwd(), "base_data", "worldcities.csv")):
        # Download Worldcities
        urllib.request.urlretrieve(url="https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.77.zip",
                                   filename=os.path.join(os.getcwd(), "base_data", "worldcities.zip"))
        shutil.unpack_archive(filename=os.path.join(os.getcwd(), "base_data", "worldcities.zip"), extract_dir=os.path.join(os.getcwd(), "base_data"))



if __name__ == "__main__":
    get_base_data()