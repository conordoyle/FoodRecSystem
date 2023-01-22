from Food_Rec_System_Engine import *

def plot_box_whisker():
    # display distribution of calories for top ten food categories
    fig = px.box(cleaned_df, x="food_categories", y="calories", title='Calorie Distribution of Top 10 Food Categories')
    fig.show()


def plot_info_graphs():
    # loop through nutritional value ratio
    for nutr_value in list(grams_df.columns)[2:]:

        # create figure
        fig = go.Figure()

        # loop through each food category
        for food_cat in list(grams_df['food_categories'].unique()):
            data = grams_df.loc[grams_df['food_categories'] == food_cat, nutr_value]
            fig.add_trace(go.Box(x=list(data), name=food_cat, boxpoints=False))

        # update figure
        fig.update_layout(title_text=nutr_value, xaxis_title='grams per calorie',
                          yaxis_title='food categories', showlegend=False)

        # show figure
        fig.show()


def plot_elbow_curve():
    # plot line plot of mean distance to centroid for n clusters from 2-7
    plt.plot(mean_d_dict.keys(), mean_d_dict.values())
    # create labels
    plt.xlabel('number of clusters')
    plt.ylabel('mean dist^2 to centroid')

if __name__ == "__main__":
    plot_box_whisker()
    plot_info_graphs()
    plot_elbow_curve()