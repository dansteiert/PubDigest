import logging
import pandas as pd


def generate_visualisations(config: dict, df_publications: pd.DataFrame, df_authors: pd.DataFrame):

    logger = logging.getLogger(config["System"]["logging"]["logger_name"])

    if config["Workflow"]["gen_plots"]:
        logger.info(f"Start Plotting")
        logger.info(f"Start Author and Publication Metadata")
        from Visualization.Metadata.Author_Metadata_Visualisations import plot_publications_by_author, plot_citations_by_author
        from Visualization.Metadata.Publication_Metadata_Visualisations import plot_publication_timeline, \
            plot_publication_timeline_in_intervals

        # plot_publications_by_year(df=df, config=config,
        #                           fontsize=10, figure_ratio=4 / 3, figure_scale=1)

        plot_publication_timeline(df_publications=df_publications, config=config,
                                  fontsize=15, figure_ratio=3 / 4, figure_scale=1, titlesize="xx-large")
        plot_publication_timeline_in_intervals(df_publications=df_publications, config=config,
                                               fontsize=15, figure_ratio=3 / 4, figure_scale=1, titlesize="xx-large")

        plot_publications_by_author(config=config, df_authors=df_authors, cutoff=40,
                                    fontsize=15, figure_ratio=2 / 1, figure_scale=2)
        plot_citations_by_author(config=config, df_authors=df_authors, cutoff=40,
                                 fontsize=15, figure_ratio=2 / 1, figure_scale=2)
        if config["Workflow"]["keyword_plotting"]:
            logger.info("Start Keyword Related Plotting")
            from Visualization.TF_IDF_Visualisations.heatmap import tfidf_heatmap
            from Visualization.TF_IDF_Visualisations.associated_disease_names import plot_associated_disease_names
            from Visualization.TF_IDF_Visualisations.wordcloud import generate_wordcloud
            disease = False
            for n_gram in config["NLP"]["n_gram_list"]:
                for med in [False, True]:
                    for abb in [False, True]:
                        logger.info(f"Keyword Plotting for ngram {n_gram}, abb: {abb}, med: {med}")
                        generate_wordcloud(config=config, n_gram=n_gram, med=med, abb=abb,
                                           fontsize=15, figure_ratio=3 / 4, figure_scale=1)
                        tfidf_heatmap(config=config, n_gram=n_gram, med=med, abb=abb, fontsize=15, figure_ratio=4 / 4,
                                      figure_scale=2,
                                      cutoff=25, titlesize="large", disease=False)
                        # TODO: Integrate
                        from Visualization.Keyword_Plotting import plot_keywords_as_venn_diagram, \
                            plot_keywords_by_timeinterval, plot_keywords_as_mosaik

                        plot_keywords_as_mosaik(config=config, n_gram=n_gram, med=med, abb=abb,
                                                cutoff=40,
                                                fontsize=16, figure_ratio=2 / 1.3, figure_scale=2)
                        plot_keywords_by_timeinterval(config=config, n_gram=n_gram, med=med, abb=abb, cutoff=20,
                                                      fontsize=16, figure_ratio=3/4, figure_scale=2,
                                                      titlesize="large")
                        plot_keywords_as_venn_diagram(config=config, n_gram=n_gram, med=med, abb=abb,
                                                      cutoff=40,
                                                      fontsize=16, figure_ratio=2 / 1.3, figure_scale=2)

                disease = True
                tfidf_heatmap(config=config, n_gram=n_gram, med=med, abb=abb, fontsize=15, figure_ratio=4 / 4,
                              figure_scale=2, disease=disease,
                              cutoff=25, titlesize="large")
            plot_associated_disease_names(config=config)

        if config["Workflow"]["affiliation_search"]:
            logger.info("Start Affiliation Related Plotting")
            from Visualization.Metadata.Publication_Metadata_Visualisations import plot_publications_by_x
            plot_publications_by_x(config=config, df_publications=df_publications, by="country", cutoff=40)
            plot_publications_by_x(config=config, df_publications=df_publications, by="city", cutoff=40)
            plot_publications_by_x(config=config, df_publications=df_publications, by="institute", cutoff=40)


        logger.info("Plotting Done")