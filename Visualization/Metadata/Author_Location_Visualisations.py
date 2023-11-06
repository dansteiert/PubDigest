

# TODO:!


def plot_citations_publications_by_author(base_dir: str, df_author: pd.DataFrame = None, cutoff: int = 40,
                                          categorial_colorfull_colors: list = ["#4daf4a", "#e41a1c", "#377eb8",
                                                                               "#984ea3",
                                                                               "#ff7f00", "#ffff33", "#a65628",
                                                                               "#f781bf",
                                                                               "#999999"],
                                          diverging_colors: str = "Spectral",
                                          color: str = "#4daf4a",
                                          figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10):
    # <editor-fold desc="Data loading of top x instances">
    if df_author is None:
        df_author = pd.read_csv(os.path.join(base_dir, "Author_Table.csv"), index_col=0)
    # </editor-fold>

    # <editor-fold desc="Prepare data">
    df_author = df_author[["all_publications", "citation_count", "name"]]
    # df_temp_publications = df_author.sort_values(by="all_publications", ascending=False).head(int(cutoff/2))
    # df_temp_citations = df_author.sort_values(by="citation_count", ascending=False).head(int(cutoff/2))
    # df = pd.concat([df_temp_citations, df_temp_publications])
    # df = df.drop_duplicates()
    # df = df.sort_values(by=["all_publications", "citation_count"], ascending=False)
    # # df["citation_count"] = np.log10(df["citation_count"])
    #
    # df_melt = df.melt(id_vars=["name"], value_vars=["all_publications", "citation_count"], value_name="value", var_name="metric")

    df_author["Citation/Publication"] = df_author["citation_count"] / df_author["all_publications"]
    df_author = df_author.sort_values(by=["Citation/Publication"], ascending=False)
    df_author.to_csv(os.path.join(base_dir, "Data_for_Plotting", "Citations and Publications.csv"))
    df_author = df_author.head(cutoff)
    df_melt = df_author.melt(id_vars=["name"], value_vars=["all_publications", "Citation/Publication"],
                             value_name="value", var_name="metric")

    # </editor-fold>

    # <editor-fold desc="select appropriate colors">
    if 2 > len(categorial_colorfull_colors):
        palette = sns.color_palette(diverging_colors, desat=1, n_colors=2)
    else:
        palette = categorial_colorfull_colors[:2]
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot"), exist_ok=True)
    set_figure_estetics(y_axis_elements=df_author.shape[0], figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    # <editor-fold desc="Horizontal barplot sorted for the most citations by an author">
    sns.barplot(data=df_author, y="name", x="Citation/Publication", color=color)
    # sns.barplot(data=df_melt, y="name", x="value", palette=palette, hue="metric")
    # </editor-fold>

    # <editor-fold desc="Figure esthetics">
    plt.title("Number of citations/publications")
    plt.ylabel("")
    plt.xlabel("")
    # plt.xscale("log")
    # plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Plot", "Publication_Citations_by_Author.png"), dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>


def plot_citations_by_author_and_country(df_author: pd.DataFrame = None, df_paper: pd.DataFrame = None,
                                         df_author_paper: pd.DataFrame = None, df_institut: pd.DataFrame = None,
                                         base_dir: str = None,
                                         top_x_authors_per_country: int = 20,
                                         top_x_countries: int = 5,
                                         color: str = "#4daf4a",
                                         background_colors: list = ["#000000", "#808000"],
                                         figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10
                                         ):
    # <editor-fold desc="Load Data">
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Citations_by_Author_and_country.csv")):
        df_citations = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Citations_by_Author_and_country.csv"),
                                   index_col=0)
    else:
        df_citations = data_citation_by_author_and_country(df_author=df_author, df_paper=df_paper,
                                                           df_author_paper=df_author_paper,
                                                           df_cities=df_institut, base_dir=base_dir)
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_Author.csv")):
        df_publications = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_Author.csv"),
                                      index_col=0)
    else:
        df_publications = data_publications_by_author(df=df_author, base_dir=base_dir)
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_country.csv")):
        df_country = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_country.csv"),
                                 index_col=0)
    else:
        df_country = data_publications_by_x(df=df_institut, base_dir=base_dir, by="country")
    # </editor-fold>

    # <editor-fold desc="keep only the top x countries and of each only the top y authors">
    country_order = list(df_country.head(top_x_countries).index)
    df_publications = df_publications[df_publications["country"].isin(country_order)]
    df_publications["country"] = pd.Categorical(df_publications["country"], reversed(country_order), ordered=True)
    df_publications = df_publications.sort_values(by=["country", "all_publications"], ascending=True)
    # </editor-fold>

    # <editor-fold desc="Sort authors by most publishing country how published most - keep only top x of each category">
    authors = list(df_publications.groupby(by="country").tail(top_x_authors_per_country)["name"])

    df_citations = df_citations[df_citations["country"].isin(country_order)]
    df_citations = df_citations[df_citations["name"].isin(authors)]
    df_citations["country"] = pd.Categorical(df_citations["country"], reversed(country_order), ordered=True)

    df_citations = df_citations.sort_values(by=["country", "all_publications"], ascending=False)
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot"), exist_ok=True)

    set_figure_estetics(y_axis_elements=top_x_authors_per_country * top_x_countries, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>
    ax = sns.stripplot(data=df_citations, x="citation_count", y="name", color=color, linewidth=1, size=5, alpha=0.5)

    # <editor-fold desc="Plot for each author the median citaion count">
    x_min, x_max = plt.gca().get_xlim()
    # calculate median
    median_width = 1

    for tick, text in zip(ax.get_yticks(), ax.get_yticklabels()):
        sample_name = text.get_text()  # "X" or "Y"

        # calculate the median value for all replicates of either X or Y
        median_val = df_citations[df_citations['name'] == sample_name]["citation_count"].median()
        # plot horizontal lines across the column, centered on the tick
        ax.plot([median_val, median_val], [tick - median_width / 2, tick + median_width / 2],
                lw=1, color='r', zorder=20)  # zorder: e.g. 2 does not plot at all!, so arbirary number chosen
    ax.plot([median_val, median_val], [ax.get_yticks()[-1] - median_width / 2, ax.get_yticks()[-1] + median_width / 2],
            lw=1, color='r', zorder=20, label="Median")
    # </editor-fold>

    # <editor-fold desc="Annotate each country with name and switching background">
    top_index = -0.5

    background_index = 0
    for i in list(country_order):
        # non-logarithmic depiction
        # plt.annotate(i, (x_max - (x_max / 4), top_index + (top_x_authors_per_country / 2)))
        plt.annotate(i, (np.power(10, (np.log10(x_max) - (np.log10(x_max) / (4 * 10 / fontsize)))),
                         top_index + (top_x_authors_per_country / 2)))
        plt.fill_between(x=range(int(x_min), int(np.ceil(x_max))), y1=top_index,
                         y2=top_index + top_x_authors_per_country,
                         color=background_colors[background_index], alpha=0.2)
        top_index += top_x_authors_per_country
        background_index = (background_index + 1) % 2
    # </editor-fold>

    # <editor-fold desc="Figure esthetics and saving">
    leg = plt.legend(loc="lower center", fontsize="medium")
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.grid(b=True, which="major", axis="both")
    plt.xscale("log")
    plt.xlabel("Citations per publication")
    plt.ylabel("")
    plt.title("Citations by Author")
    plt.tight_layout()

    plt.savefig(os.path.join(base_dir, "Plot", "Citations_by_Author_and_Country.png"), dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>


def plot_publications_by_author_and_country(df_author: pd.DataFrame = None, df_institut: pd.DataFrame = None,
                                            base_dir: str = None,
                                            top_x_authors_per_country: int = 20,
                                            top_x_countries: int = 5,
                                            categorial_colorfull_colors: list = ["#4daf4a", "#e41a1c", "#377eb8",
                                                                                 "#984ea3",
                                                                                 "#ff7f00", "#ffff33", "#a65628",
                                                                                 "#f781bf",
                                                                                 "#999999"],
                                            diverging_colors: str = "Spectral",
                                            background_colors: list = ["#000000", "#808000"],
                                            figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10,
                                            titlesize: str = "xx-large"):
    # <editor-fold desc="Load necessary data">
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_Author.csv")):
        df = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_Author.csv"), index_col=0)
    else:
        df = data_publications_by_author(df=df_author, base_dir=base_dir)
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_country.csv")):
        df_country = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_country.csv"),
                                 index_col=0)
    else:
        df_country = data_publications_by_x(df=df_institut, base_dir=base_dir, by="country")
    # </editor-fold>

    # <editor-fold desc="keep only the top x countries and of each only the top y authors">
    country_order = list(df_country.head(top_x_countries).index)
    df = df[df["country"].isin(country_order)]
    df["country"] = pd.Categorical(df["country"], reversed(country_order), ordered=True)
    df = df.sort_values(by=["country", "all_publications"], ascending=True)
    # df = df.sort_values(by=["country"], ascending=True)
    df = df.groupby(by="country").tail(top_x_authors_per_country).reset_index(drop=True)
    # </editor-fold>

    # <editor-fold desc="select appropriate colors">
    if 3 > len(categorial_colorfull_colors):
        palette = sns.color_palette(diverging_colors, desat=1, n_colors=3)
    else:
        palette = categorial_colorfull_colors[:3]
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot"), exist_ok=True)
    set_figure_estetics(y_axis_elements=df.shape[0], figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    # <editor-fold desc="Plot horizontal stacked Barplots">

    plt.barh(y=df["name"], width=df["publications_as_first_author"],
             label="First authored", color=palette[0], edgecolor="gray")
    plt.barh(y=df["name"], width=df["publications_as_institute_author"],
             left=df["publications_as_first_author"],
             label="Last authored", color=palette[1], edgecolor="gray")
    plt.barh(y=df["name"], width=df["count_middle"],
             left=df["bottom_other"],
             label="Co-authored", color=palette[2], edgecolor="gray")
    # positions = [i * 2 for i in df.shape[0]]
    # plt.barh(y=positions, width=df["publications_as_first_author"], height=1,
    #               label="First authored", color=palette[0], edgecolor="gray", tick_label=df["name"])
    # plt.barh(y=positions, width=df["publications_as_institute_author"],
    #               left=df["publications_as_first_author"], height=1,
    #               label="Last authored", color=palette[1], edgecolor="gray", tick_label=df["name"])
    # plt.barh(y=positions, width=df["count_middle"],
    #               left=df["bottom_other"], height=1,
    #               label="Co-authored", color=palette[2], edgecolor="gray", tick_label=df["name"])
    # plt.yticks(df["name"])
    # </editor-fold>

    # <editor-fold desc="Annotate each country with Name and set switches background for the given background colors">
    top_index = -0.5
    # for box positioning
    x_min, x_max = plt.gca().get_xlim()

    background_index = 0
    for i in list(reversed(country_order)):
        elements = df[df["country"] == i].shape[0]
        plt.annotate(i, (x_max - (x_max / (4 * 10 / fontsize)), top_index + (elements / 2)))
        plt.fill_between(x=range(int(x_min), int(np.ceil(x_max))), y1=top_index, y2=top_index + elements,
                         color=background_colors[background_index], alpha=0.2)
        top_index += elements
        background_index = (background_index + 1) % 2
    # </editor-fold>

    # <editor-fold desc="Figure esthetics and saving">
    plt.legend(loc="lower right", fontsize="medium", handlelength=2.5, markerscale=1)
    plt.xlabel("Publication count")
    plt.title("Publications by Author")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Plot", "Publications_by_Author_and_Country.png"), dpi=400, transparent=True)
    plt.close("all")
    # </editor-fold>


def plot_publications_by_first_last_author_and_country(df_author: pd.DataFrame = None, df_cities: pd.DataFrame = None,
                                                       base_dir: str = None, cutoff: int = 20,
                                                       country_filter: bool = True,
                                                       top_x_countries: int = 5, top_x_country_labels: int = 10,
                                                       categorial_colorfull_colors: list = ["#4daf4a", "#e41a1c",
                                                                                            "#377eb8",
                                                                                            "#984ea3", "#ff7f00",
                                                                                            "#ffff33",
                                                                                            "#a65628", "#f781bf",
                                                                                            "#999999"],
                                                       diverging_colors: str = "Spectral",
                                                       figure_ratio: float = 2 / 1, figure_scale: float = 1,
                                                       fontsize: int = 10, titlesize: str = "xx-large"):
    # <editor-fold desc="Load data">
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_author_and_country.csv")):
        df = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_author_and_country.csv"),
                         index_col=0)
    else:
        df = data_publications_by_author_and_country(df_author=df_author, df_cities=df_cities, base_dir=base_dir)
    # </editor-fold>

    # <editor-fold desc="Retrieve the top x countries">
    if country_filter:
        country_list = list(
            df.groupby(by="country")["index"].count().sort_values(ascending=False).head(top_x_countries).index)
        df = df[df["country"].isin(country_list)]
    else:
        country_list = df["country"].unique().tolist()
    # </editor-fold>

    # <editor-fold desc="Map all countries not in the selected x countries to 'other'">
    if len(country_list) > top_x_country_labels:
        df["country"] = df.apply(
            lambda x: x["country"] if x["country"] in country_list[:top_x_country_labels] else "other", axis=1)
        country_list = [*country_list[:top_x_country_labels], "other"]
    # </editor-fold>

    # <editor-fold desc="Select the appropriate color palette">
    if len(country_list) + 1 > len(categorial_colorfull_colors):
        palette = sns.color_palette(diverging_colors, desat=1, n_colors=len(country_list))
    else:
        palette = categorial_colorfull_colors[:len(country_list)]
    # </editor-fold>

    # <editor-fold desc="Change column names for later easier usage">
    df = df.rename(
        columns={"publications_as_first_author": "First Author", "publications_as_institute_author": "Last Author"})
    # </editor-fold>

    # <editor-fold desc="Set Figure sizes and respective labels">
    os.makedirs(os.path.join(config["base_dir"], "Plot"), exist_ok=True)
    set_figure_estetics(y_axis_elements=cutoff, figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    for i in ["First Author", "Last Author"]:
        # <editor-fold desc="Sort by most publishing author and generate a scatterplot">
        df = df.sort_values(by=i, ascending=False)
        sns.scatterplot(data=df.head(cutoff), x="all_publications", y="name", hue="country", hue_order=country_list,
                        palette=palette,
                        size=i, sizes=(100, 1000), alpha=0.5)
        # </editor-fold>

        # <editor-fold desc="Figure esthetics as saving">
        plt.grid(b=True, which="major", axis="both")
        plt.title(f"Authors by Publications - Scaled by {i}")
        plt.xlabel("Publication count")
        plt.ylabel("Author name")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(os.path.join(base_dir, "Plot", f"Publications_by_Author_and_Country_{i}.png"), dpi=400,
                    transparent=True)
        plt.close("all")
        # </editor-fold>


def radar_graph_authors_in_countries(base_dir: str, df_author: pd.DataFrame, df_cities: pd.DataFrame,
                                     top_x_countries: int = 5,
                                     categorial_colorfull_colors: list = ["#4daf4a", "#e41a1c", "#377eb8",
                                                                          "#984ea3", "#ff7f00", "#ffff33",
                                                                          "#a65628", "#f781bf", "#999999"],
                                     diverging_colors: str = "Spectral",
                                     figure_ratio: float = 2 / 1, figure_scale: float = 1, fontsize: int = 10):
    os.makedirs(os.path.join(config["base_dir"], "Plot"), exist_ok=True)
    # <editor-fold desc="Set Figure sizes and respective labels">
    set_figure_estetics(figure_ratio=figure_ratio,
                        figure_scale=figure_scale, fontsize=fontsize)
    # </editor-fold>

    # <editor-fold desc="Load data">
    if os.path.isfile(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_author_and_country.csv")):
        df = pd.read_csv(os.path.join(base_dir, "Data_for_Plotting", "Publications_by_author_and_country.csv"),
                         index_col=0)
    else:
        df = data_publications_by_author_and_country(df_author=df_author, df_cities=df_cities, base_dir=base_dir)
    # </editor-fold>

    # <editor-fold desc="Generate the three categories total authors, #first authored - and last authored articles, for the top x countries">
    df_groupby = df.groupby(by="country")
    temp_df = pd.DataFrame()
    temp_df["authorships"] = df_groupby["name"].count()
    temp_df["first_author"] = df_groupby["publications_as_first_author"].sum()
    temp_df["last_author"] = df_groupby["publications_as_institute_author"].sum()
    temp_df = temp_df.sort_values(by="authorships", ascending=False).head(top_x_countries)
    # </editor-fold>

    # TODO: assign colors
    # <editor-fold desc="appropriate sized color palette">
    if temp_df["country"].unique().shape[0] > len(categorial_colorfull_colors):
        palette = sns.color_palette(diverging_colors, desat=1, n_colors=temp_df["country"].unique().shape[0])
    else:
        palette = categorial_colorfull_colors[:temp_df["country"].unique().shape[0]]
    # </editor-fold>

    # <editor-fold desc="Figure generation">
    fig = go.Figure()
    categories = temp_df.columns
    for index, row in temp_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row,
            theta=categories,
            fill='toself',
            name=index
        ))
    # </editor-fold>

    # <editor-fold desc="esthetics and saving">
    fig.write_html(os.path.join(base_dir, "Plot", "Radar_Chart_Authors_by_Country.html"))
    fig.write_image(os.path.join(base_dir, "Plot", "Radar_Chart_Authors_by_Country.png"))
    fig.show()
    # </editor-fold>

