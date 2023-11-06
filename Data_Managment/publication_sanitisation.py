import re, unidecode, datetime, pickle, os, logging
import pandas as pd

from pymed.article import PubMedArticle
try:
    from nltk import pos_tag
    from nltk.tokenize import regexp_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn
except:
    logging.fatal("call nltk.download() and select all")
    exit()



def sanitise_publication(config: dict, PubMedArticle: PubMedArticle, iCite_dict: dict,
                         df_affiliation: pd.DataFrame, missing_ids: set):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    dict_mapping = {"pmid": "pubmed_id",
                    "title": "title",
                    "abstract": "abstract",
                    "keywords": "keywords",
                    "journal": "journal",
                    "methods": "methods",
                    "conclusions": "conclusions",
                    "results": "results",
                    "copyright": "copyright",
                    "doi": "doi",
                    }
    article_dict = PubMedArticle.toDict()
    clean_article_dict = {key: article_dict.get(val, None) for key, val in dict_mapping.items()}
    clean_article_dict["pmid"] = clean_article_dict["pmid"].split("\n")

    if not len(clean_article_dict["pmid"]) >= 1:
        logger.error(f"File has no identifier: {clean_article_dict}")
        return
    if not clean_article_dict['pmid'][0] in missing_ids:
        logger.info(f"File does already exist {clean_article_dict['pmid'][0]}")
        return
    clean_article_dict["identifier"] = clean_article_dict["pmid"][0]

    try:
        clean_article_dict["publication day"] = article_dict.get("publication_date",
                                                                 datetime.date.fromisoformat('3999-12-31')).day
        clean_article_dict["publication month"] = article_dict.get("publication_date",
                                                                   datetime.date.fromisoformat('3999-12-31')).month
        clean_article_dict["publication year"] = article_dict.get("publication_date",
                                                                  datetime.date.fromisoformat('3999-12-31')).year
    except Exception as e:
        try:
            clean_article_dict["publication year"] = int(article_dict.get("publication_date"))
            clean_article_dict["publication day"] = 31
            clean_article_dict["publication month"] = 12
        except:
            logger.error(f"{e}, Publication date is not readable {article_dict.get('publication_date')}")
            clean_article_dict["publication day"] = datetime.date.fromisoformat('3999-12-31').day
            clean_article_dict["publication month"] = datetime.date.fromisoformat('3999-12-31').month
            clean_article_dict["publication year"] = datetime.date.fromisoformat('3999-12-31').year
    try:
        [clean_article_dict.__setitem__(j.tag, j.text) for i in article_dict["xml"] for j in i]
    except Exception as e:
        logger.error(f"{e}; xml Key not in publication metadata {article_dict.keys()}")
    # might be in any case a smart thing to always retrieve the actual numbers?

    if config["Workflow"]["affiliation_search"]:
        df_countries_only = pd.DataFrame(data={"country": df_affiliation["country"].unique()})
        if article_dict["authors"] is not None:
            author_list = [i for i in article_dict["authors"] if not i.get("lastname") is None]
            try:
                [i.__setitem__("reference name", f"{i.get('lastname', '_')}, {i.get('firstname', '_')}")
                 for i in author_list]
            except Exception as e:
                logger.error(f"{e}; Error with Author Name {article_dict['authors']}")
            try:
                author_list = [
                    {**i, **affiliation_mapping(config=config, df_cities=df_affiliation, df_country=df_countries_only,
                                                affiliation=i["affiliation"]),
                     **{"author_position": idx, "first_author": True if idx == 0 else False,
                        "last_author": True if idx == len(article_dict["authors"]) - 1 else False,
                        "second_last_author": True if idx == len(article_dict["authors"]) - 2 else False,
                        "pmid": clean_article_dict.get("identifier", None)}}
                    for idx, i in enumerate(article_dict["authors"])]
            except Exception as e:
                logger.error(f"{e}; Error with author - affiliation mapping {author_list}")

            clean_article_dict["authors"] = author_list
    clean_article_dict["clean_text"] = process_article_text(article_dict=clean_article_dict, config=config)
    clean_article_dict["identifier"] = clean_article_dict["pmid"][0]
    clean_article_dict["citation_update_year"] = datetime.datetime.now().year
    clean_article_dict["citation_update_month"] = datetime.datetime.now().month
    clean_article_dict["citation_update_day"] = datetime.datetime.now().day
    iCite_dict = {f"iCite_{k}": v for k, v in iCite_dict.items()}
    clean_article_dict = {**clean_article_dict, **iCite_dict}
    target_dir = os.path.join(config["System"]["working_dir"], "processed_publications", *[i for i in str(clean_article_dict['identifier'])[:-1]])
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f"{clean_article_dict['identifier']}.pkl"),
              "wb") as f:
        pickle.dump(clean_article_dict, f)
    return {clean_article_dict['identifier']: os.path.join(target_dir, f"{clean_article_dict['identifier']}.pkl")}

    # return clean_article_dict


def affiliation_mapping(config: dict, df_cities: pd.DataFrame, df_country: pd.DataFrame, affiliation: str):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])
    if affiliation is None:
        return {}

    affiliation = unidecode.unidecode(affiliation)
    for aff in re.split("[\n]", affiliation):
        aff = re.sub(pattern="[0-9]|[\"\']", repl="", string=aff)
        aff = [i.strip(" .") for i in re.split("[,;\-/]", aff)]
        # </editor-fold>

        # <editor-fold desc="find Latitude and Longitude of corresponding cities">
        country = None
        city = None
        aff_id = None
        institute = None

        # <editor-fold desc="Get the Affiliation Country">
        for j in reversed(aff):
            country = find_country(search=j, df_country=df_country, config=config, country_dict=config["Locations"]["country_dict"])
            if country is not None:
                break
        # </editor-fold>

        # <editor-fold desc="Get Affiliation City">
        for j in reversed(aff):
            result = find_city(country=country, search=j, config=config, df_cities=df_cities,
                               city_dict=config["Locations"]["city_dict"])
            if result is not None:
                aff_id = result
                city = j
                break
        # </editor-fold>

        # <editor-fold desc="If no city is found, try associate with a country">
        if aff_id is None:
            if config["Debug"]["print_non_assigned_affiliations"]:
                logger.warning(f"No city found for this affiliation: {aff}, country: {country}")
            if country is not None:
                aff_id = find_country_only(search=country, df_cities=df_cities, config=config)


        if config["Debug"]["print_non_assigned_affiliations"]:
            if aff_id is None:
                logger.warning(f"No city and country found for this affiliation: {aff}")
        # </editor-fold>

        # <editor-fold desc="Find Institutions by keywords">
        for j in reversed(aff):
            if any(inst in j for inst in config["Locations"]["institute_names"]):
                institute = j
                break
        # </editor-fold>

        email = [j[: j.find("\n")] for i in aff if "@" in i for j in i.split(" ") if "@" in j]
        try:
            email = [i.strip(" .") for i in email][0]
        except:
            email = None

        try:
            return {**{"institute": institute, "email": email, "country": country, "city": city, "affiliation_id": aff_id},
                    **{i: df_cities.iloc[aff_id][i] for i in ["lat", "lng"]}}
        except Exception as e:
            if aff_id is not None:
                logger.error(f"{e} {aff_id} Error in affiliation matching {affiliation}; found entries: country: {country}; city; {city}; institute {institute}, with email {email}")
            return {**{"institute": institute, "email": email, "country": country, "city": city, "affiliation_id": aff_id},
                    **{i: None for i in ["lat", "lng"]}}
    #     affiliation_list.append({**{"original affiliation": aff, "institute": institute},
    #                              **{i: df_cities.iloc[aff_id][i] for i in ["country", "city", "longitude", "latitude"]}})
    # return affiliation_list


def find_country(search: str, config: dict, df_country: pd.DataFrame, country_dict: dict={}):

    '''
    Find if the country is represented in the database - or if the search string does not represent a country name
    :param search: The country as string
    :param con: Database connection instance
    :return: bool - True if the country is found in the db
    '''

    try:
        search = search.strip(" -()[];:.,")
        search = country_dict.get(search.lower(), search)
        data = df_country.query(f'country == "{search}"')
        if data.shape[0] > 0:
            return search
        for i in search.split(" "):
            if i != search:
                i = i.strip(" -()[];:.,")
                i = country_dict.get(i.lower(), i)
                data = df_country.query(f'country == "{i}"')
                if data.shape[0] > 0:
                    return i
        return None
    except Exception as e:
        logging.getLogger(config["System"]["logging"]["logger_name"]).error(f"{e} {search} error in query find country")
        return False


def find_city(country: str, search: str, config: dict, df_cities: pd.DataFrame, city_dict: pd.DataFrame={}):

    '''
    Find the id of the city in the database
    :param country: name of the country
    :param search: name of the city to search for
    :param con: Database connection instance
    :return: id of the city in the table Institute
    '''


    try:
        search = search.strip(" -()[];:.,")

        search = city_dict.get(search.lower(), search)
        if country is not None:
            data = df_cities.query(f'country == "{country}" & city == "{search}"')

            if data.shape[0] > 0:
                return data.iloc[0]["index"]
            for i in reversed(search.split(" ")):
                if i != search:
                    i = i.strip(" -()[];:.,")
                    i = city_dict.get(i.lower(), i)

                    data = df_cities.query(f'country == "{country}" & city == "{i}"')
                    if data.shape[0] > 0:
                        return data.iloc[0]["index"]

        else:
            data = df_cities.query(f'city == "{search}"')
            if data.shape[0] > 0:
                return data.iloc[0]["index"]
            for i in search.split(" "):
                if i != search:
                    i = i.strip(" -()[];:.,")
                    i = city_dict.get(i.lower(), i)
                    data = df_cities.query(f'city == "{i}"')
                    if data.shape[0] > 0:
                        return data.iloc[0]["index"]
    except Exception as e:
        logging.getLogger(config["System"]["logging"]["logger_name"]).error(f"{e} {search}, {country}, error in query find city")
        return None


def find_country_only(search: str, config: dict, df_cities: pd.DataFrame):

    '''
    Find the Id of the country, in the Institute table
    :param search: The country as string
    :param con: Database connection instance
    :return: Id of the element in the database
    '''
    data = df_cities.query(f'country == "{search}" & city != city') # city !=city checks if city is NaN
    if data.shape[0] > 0:
        return data.iloc[0]["index"]
    return None



def process_article_text(article_dict: dict, config: dict):
    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    text_data = [article_dict.get(i, None) for i in ["title", "abstract", "methods", "conclusions", "results"]]
    try:
        text_data = [*text_data, *article_dict.get("keywords", [])]
        text_data = [i for i in text_data if i is not None]
    except:
        logger.error(f"Acquiring keywords in the correct format: {article_dict.get('keywords', 'empty')}")
        text_data = [i for i in text_data if i is not None]

    if len(text_data) == 0:
        logger.info("No text found")
        return []
    try:
        text = " ".join(text_data)
        text = text.replace("\n", " ").replace("\r", " ")
    except Exception as e:
        logger.error(f"{e}, {text_data}, {[article_dict.get(i, None) for i in ['title', 'abstract', 'methods', 'keywords', 'conclusions', 'results']]}")
    if text in [" ", ""]:
        return []

    lemmatizer = WordNetLemmatizer()
    post_tag_dict = {"N": wn.NOUN, "R": wn.ADV, "J": wn.ADJ, "V": wn.VERB}

    tokens = regexp_tokenize(text, pattern='[\w\-]+|\$[\d\.]+|\S+')
    tokens = [i for i in tokens if re.match(
        pattern="(([0-9]+[\-_]+)|([0-9]*))[a-zA-Z]+(([\-_]+[0-9]+)|([0-9]*))", string=i)]
    tokens = [i if re.match(pattern="[a-zA-Z0-9]+[A-Z]+", string=i) else i.lower() for i in tokens]

    lemma = [lemmatizer.lemmatize(word=tag_tok[0], pos=post_tag_dict.get(tag_tok[1][0], wn.NOUN)) for tag_tok in
             pos_tag(tokens) if
             not post_tag_dict.get(tag_tok[1][0], None) is None]
    return lemma

if __name__ == "__main__":
    pass