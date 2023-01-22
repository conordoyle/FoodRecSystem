from Food_Rec_System_Engine import df_x_sn, y
import seaborn as sns
import matplotlib.pyplot as plt
from copy import copy
import plotly.express as px

def plot_clusters():
    # plot scatterplot of clusters
    sns.scatterplot(data=df_x_sn, x="pca0", y="pca1", s=100, hue=y, palette='Set2')
    plt.gcf().set_size_inches(12, 6)

    sns.set(font_scale=2)

    plt.suptitle('Foods Clustered by all Nutritional Values')

    sns.set(font_scale=2)



def plot_pca_clusters():
    # create temp dataframes for the plotly graph
    df_removed_categories = copy(df_x_sn)
    df_temp = copy(df_x_sn)

    # limit down to the top 50 most frequent categories
    kept_categories = df_removed_categories['food_categories'].value_counts().iloc[:50]

    # removed rows that are in unecessary categories
    for i in range(len(df_temp)):
        if df_temp.iloc[i]['food_categories'] not in kept_categories:
            df_removed_categories.drop(i, axis=0, inplace=True)


    # create PCA plot using plotly scatter
    fig = px.scatter(df_removed_categories, x='pca0', y='pca1', hover_data=['name'], color='food_categories', width=1700, height=500)
    fig.update_layout(yaxis_range=[-5,25])
    fig.update_layout(xaxis_range=[-7,35])
    fig.show()

    #export to HTML
    fig.write_html('cluster_pca.html')


if __name__ == "__main__":
    plot_clusters()
    plot_pca_clusters()
