# Import Packages
import rasterio
import fiona
import os

import pyproj
if os.path.isdir(r'C:\Users\!lbrown\Miniconda3\envs\hsi36\Library\share\proj'):
    pyproj.datadir.set_data_dir(r'C:\Users\!lbrown\Miniconda3\envs\hsi36\Library\share\proj')
else:
    pyproj.datadir.set_data_dir(r'C:\ProgramData\Miniconda3\envs\hsi36\Library\share\proj')

import numpy as np
import pandas as pd
import rasterio.mask as rmsk
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
# import plotly.express as px

from rasterio.vrt import WarpedVRT

from sklearn.preprocessing import StandardScaler
import umap
import umap.plot

# Bokeh
from bokeh.palettes import Spectral10, Category20
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider, Range1d
from bokeh.layouts import column, row, gridplot
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

# todo: hsi39 import conflict bokeh & umap.plot regarding registering colormap
# todo: chip_d_ss currently dislikes derived bands
# todo: hsi39 import conflict bokeh & umap.plot regarding registering colormap
# todo: extend spectral signature plotter to have a clickable legend
# todo: hover tool extend to polygon FIDs
# todo: lasso tool coupled to highlight polygons


def rgb_to_hex(red, green, blue):
    return f'#{red:02x}{green:02x}{blue:02x}'


# Define Functions - structural
def parse_lookup_table(lookup_csv):
    class_field = 'Class_Code'
    species_field = 'Common Name'
    red, green, blue = 'R', 'G', 'B'
    lookup = pd.read_csv(lookup_csv)
    try:
        species_d = dict(
            list(
                zip(
                    lookup[class_field],
                    lookup[species_field]
                )
            )
        )
        colors_d = dict(
            list(
                zip(
                    lookup[species_field],
                    lookup.apply(lambda colors: rgb_to_hex(colors[red], colors[green], colors[blue]), axis=1).tolist()
                )
            )
        )
        species_d[-99] = 'Removed'
        return species_d, colors_d
    except Exception as e:
        print(f"Error: Unable to handle column name {e}")
        print("Current standard field names for class code and class name are: \n"
              "'Class_Code' and 'Common Name'")


def extract_chips(input_img, input_shp, class_field, species_d, chip=False):
    class_dict = {}
    class_dict_ss = {}

    # extract elements from the shapefile
    with fiona.open(input_shp, "r") as c:
        # %time
        # print("stopwatch for compiling geometries to list")
        shapes = [feature["geometry"] for feature in c]  # list geometries
        # 3d -> 2d shape conversion (for bokeh, mainly)
        shapes = [
            {
                "type": g["type"],
                "coordinates": [[xyz[0:2] for xyz in p] for p in g["coordinates"]],
            }
            for g in shapes if g['type'] == 'Polygon'
        ]

        class_vals = [feature['properties'][class_field] for feature in c]  # list class values per polygon

        tree_tags = []

        for feature in c:
            if 'tree_tag' not in feature['properties'].keys():
                # print(feature['properties'].keys())
                feature['properties'].setdefault('tree_tag', 'no_tag')
                # print(f"new keys: {feature['properties'].keys()}")
                # print(f"value for current tree tag: {feature['properties']['tree_tag']}")
            tree_tags.append(feature['properties']['tree_tag'])

        # print(tree_tags)

        # check species counts, warn user about low sample size
        species = list(
            map(species_d.get, class_vals))  # derive list of species from the class values present in polygon set
        species_ct_d = {item: species.count(item) for item in species}  # or create a dictionary of species counts
        rare = []
        for k, v in species_ct_d.items():
            if v < 20:
                rare.append(k)  # create list of classes with <20 samples
        print(
            "Warning: The following species have low sample size (species ct under 20), \n"
            "which may impact the visualization: ")
        for i in rare:
            print(f"{i}; Count: {species_ct_d[i]}")  # return a warning to the user

    # get chip for each polygon, pool pixels
    count = 0
    debug = 0
    with WarpedVRT(rasterio.open(input_img), dtype='float32') as rds:
        for shp, val, tag in zip(shapes, class_vals, tree_tags):

            out_img, out_transform = rmsk.mask(rds, [shp], nodata=np.nan,
                                               crop=True)  # grab array of pixels for each poly
            out_meta = rds.meta

            # Create dictionary of pooled pixels per class
            bands = rds.count
            spect_all = out_img.reshape(bands, -1)  # collapse 3D to 2D array: bands x pixels

            # drop pixels with nan values (outside poly)
            spectra = spect_all[:, ~np.all(np.isnan(spect_all), axis=0)].reshape(bands, -1).T
            spectra[spectra == 0] = np.nan  # 0 values represent pixel x band combos with no value. Change to nan

            tags = np.full((spectra.T.shape[1], 1), tag)
            tagged_spectra = np.concatenate([tags.T, spectra.T]).T

            if val not in class_dict:
                class_dict[val] = [pixel for pixel in tagged_spectra]  # list of n-dimensional pixels
            else:
                class_dict[val] += [pixel for pixel in tagged_spectra]

            # Create dictionary of pooled pixels per class for non-derived bands only
            out_band_info = dict(list(zip(rds.indexes, rds.descriptions)))
            if debug < 1:
                print(f"\nthis is your out_band_info!\n{out_band_info}\n")
                debug += 1

            derived_band_keys = ["MNF ", "Dissimilarity ", "NDVI ", "Range ", "PC ", "PCA ", "Index"]

            non_derived = {}
            if out_band_info is not None:
                for k, v in out_band_info.items():
                    if any(key in v for key in derived_band_keys if v is not None):
                        continue
                    else:
                        non_derived[k] = v
                band_select = list(non_derived.keys())
            # todo: identify when band_select is zero
            if len(band_select) > 0:
                out_img_sub = out_img[np.array(band_select) - 1, :, :]
                bands = len(band_select)
                spect_all = out_img_sub.reshape(bands, -1)

                # drop pixels with nan values (outside poly)
                spectra = spect_all[~np.isnan(spect_all)].reshape(bands, -1).T
                spectra[spectra == 0] = np.nan

            if val not in class_dict_ss:
                class_dict_ss[val] = [pixel for pixel in spectra]  # list of pixels
            else:
                class_dict_ss[val] += [pixel for pixel in spectra]

            # output chip as tif
            if chip is True:
                out_meta.update({"driver": "GTiff",
                                 "height": out_img.shape[1],
                                 "width": out_img.shape[2],
                                 "transform": out_transform})

                count += 1
                with rasterio.open(f"chips\\RGB.byte.masked.{count}.tif", "w", **out_meta) as dest:
                    dest.write(out_img)
                print(f"Individual raster chips written out to folder ~/chips")

    return class_dict, class_dict_ss, species_ct_d


def pixel_df(chip_d, species_dict, output_extracted_data=None, mission_prefix=None, filter_percent=0.05):
    print('preparing pixel dataframe')
    band_ct = len(chip_d[list(chip_d.keys())[0]][0])

    column_names = ['class_code', 'tree_tag']
    column_names += [str(x) for x in range(1, band_ct)]

    px_df = pd.DataFrame(((k, *x) for k, v in chip_d.items() for x in v), columns=column_names)

    px_df['class_code'] = pd.to_numeric(px_df['class_code'], downcast='integer')  # class code data type to int
    px_df['species'] = px_df['class_code'].map(species_dict)  # map species name by class code based on lookup table
    drop_species = ['Removed', 'NoData', 'Unknown']
    px_df.drop(px_df[px_df['species'].isin(drop_species)].index, inplace=True)
    px_df.replace(['None', 'nan'], np.nan, inplace=True)  # nan finds its way in with type conversions
    for band in px_df.columns:
        null_count = px_df[band].isna().sum()
        if null_count / len(px_df[band]) > filter_percent:
            px_df.drop(band, axis=1, inplace=True)
            print(f"column {band} is more than {filter_percent * 100}% null values: {null_count}... dropping it")
    px_df.dropna(axis=0, inplace=True)
    # filter lots of NaNs
    if output_extracted_data is not None:
        px_df.to_csv(os.path.join(
            output_extracted_data,
            f"{mission_prefix}_umap_extracted_training_data" + f"_nullthresh{int(filter_percent * 100)}.csv")
        )
    return px_df


# Define Functions - plotting
def spectral_signature_plot(chips, classnames, band_list=None, species_select=None, stdev=False, ss_fp=False):
    band_ct = np.array(chips[list(chips.keys())[0]]).shape[1]
    if band_list is not None:
        bandlist = band_list
    else:
        bandlist = list(range(1, band_ct))
    if species_select is None:
        species_select = list(chips.keys())
    plt.figure(figsize=(16, 9), dpi=150)
    for i in species_select:
        if classnames[i] in ['NoData', 'Unknown', 'Removed']:
            continue

        plt.plot(bandlist,
                 np.nanmean(chips[i], axis=0)[bandlist],
                 label=classnames[i])

        if stdev:
            plt.fill_between(bandlist,
                             (np.nanmean(chips[i], axis=0) - np.nanstd(chips[i], axis=0))[bandlist],
                             (np.nanmean(chips[i], axis=0) + np.nanstd(chips[i], axis=0))[bandlist],
                             alpha=0.4)

    plt.title('Spectral Signature by Class')
    plt.ylabel('Reflectance')
    plt.xlabel('Spectral Band')
    plt.xticks(np.arange(min(bandlist), max(bandlist) + 1, 1))
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    if ss_fp is True:
        plt.savefig(ss_fp, dpi=300, format='png')

    return plt.show()


def spectral_pairplot(band_select_list, px_df, pp_fp=False):
    band_ct = px_df.shape[1]
    if band_select_list:
        band_select = [str(x) for x in band_select_list]
    else:
        band_select = [str(x) for x in list(range(1, band_ct - 1))]
    # print(band_select)
    if len(band_select) > 10:
        print("For effective use of this visualization, please select 10 or fewer bands")
        print("Defaulting to first 10 bands")
        del band_select[10:]
    band_select.extend(['species'])
    # print(band_select)
    plt_df = px_df.filter(band_select, axis=1)
    palette = sns.color_palette("gist_rainbow", len(plt_df.species.unique()))
    sns.pairplot(plt_df, hue='species', palette=palette)

    if pp_fp is True:
        plt.savefig(pp_fp, format='png')

    return plt.show()


def umap_vis_plot(px_df, shapefile, band_select_list, species_select_list, colors, u_fp=False, debug=False):
    df = px_df.copy()
    species_list = df.species.unique()
    colors_select = list(map(colors.get, species_select_list))
    if species_select_list:
        for sp in species_select_list:
            if sp not in species_list:
                print(f"current species {sp} is not in {species_list}")
        df.drop(df[~df['species'].isin(species_select_list)].index, inplace=True)
    band_ct = df.shape[1]
    if band_select_list:
        band_select = [str(x+1) for x in band_select_list]
        if len(band_select) > band_ct:
            print("Band selection exceeds band count of input image")
            print("Defaulting to all bands")
        else:
            band_select.extend(['species', 'tree_tag', 'class_code'])
            df.drop(columns=[col for col in df if col not in band_select], inplace=True)
            if debug:
                print(f"current dataframe columns: \n{df.columns}")
    df.dropna(inplace=True)
    pixel_data = df.drop(['species', 'tree_tag', 'class_code'], axis=1).values

    # ValueError: Found array with 0 sample(s) (shape=(0, 106)) while a minimum of 1 is required by StandardScaler.

    scaled_class_data = StandardScaler().fit_transform(pixel_data)
    reducer = umap.UMAP(metric='manhattan', n_neighbors=18, random_state=52)  # euclidean, chebyshev
    embedding = reducer.fit_transform(scaled_class_data)

    df.loc[:, 'UMAP_1'] = embedding[:, 0]
    df.loc[:, 'UMAP_2'] = embedding[:, 1]

    # bokeh

    # source: https://www.kaggle.com/code/yohanb/nmf-visualized-using-umap-and-bokeh/notebook
    df["colors"] = df["species"].map(colors)

    sources = {}

    for species_name in df.species.unique():
        sources[species_name] = ColumnDataSource(
            data=dict(
                x=pd.DataFrame(df.loc[df['species'] == species_name, ['UMAP_1']]).squeeze().tolist(),
                y=pd.DataFrame(df.loc[df['species'] == species_name, ['UMAP_2']]).squeeze().tolist(),
                species=pd.DataFrame(df.loc[df['species'] == species_name, ['species']]).squeeze().tolist(),
                colors=pd.DataFrame(df.loc[df['species'] == species_name, ['colors']]).squeeze().tolist(),
                tree_tags=pd.DataFrame(df.loc[df['species'] == species_name, ['tree_tag']]).squeeze().tolist()
            )
        )

    hover_emb = HoverTool(names=["df"], tooltips="""
        <div style="margin: 10">
            <div style="margin: 0 auto; width:300px;">
                <span style="font-size: 12px; font-weight: bold;">Species:</span>
                <span style="font-size: 12px">@species</span>
                <span style="font-size: 12px; font-weight: bold;">Tree-tag:</span>
                <span style="font-size: 12px">@tree_tags</span>
            </div>
        </div>
        """)
    tools_emb = [hover_emb, 'lasso_select', 'tap', 'pan', 'box_zoom', 'wheel_zoom', 'reset']
    plot_emb = figure(plot_width=900, plot_height=900, tools=tools_emb, title='species UMAP embedding',
                      output_backend="webgl")
    # for species_name, color in zip(df.species.unique(), Category20[len(df.species.unique())]):
    for species_name, color in zip(df.species.unique(), colors_select):
        plot_emb.circle(x='x', y='y', size=3.5, color=color,  # fill_color='colors',
                        muted_alpha=0.04, source=sources[species_name], name="df",
                        legend_label=species_name)

    plot_emb.x_range = Range1d(-8, 18)
    plot_emb.y_range = Range1d(-8, 18)

    plot_emb.legend.location = "top_left"
    plot_emb.legend.click_policy = "mute"  # "hide"

    plot_emb.toolbar.autohide = True

    # plot training polys

    gt_shp = gpd.read_file(shapefile).to_crs(3857)

    gt_bokeh = convert_geopandas_to_bokeh_format(gt_shp)
    # print(gt_bokeh.data)

    p = figure(plot_width=900, plot_height=900, output_backend="webgl",
               x_axis_type="mercator", y_axis_type="mercator",
               title="extracted ground truth polygons \n"
                     + "pulled from: " + os.path.basename(shapefile))
    p.patches('x', 'y', source=gt_bokeh, name="grid", line_width=2.5, color='red')  # fill_color="red",
    # todo: plot hsi swaths with faint alpha
    # show(p)
    tile_provider = get_provider(CARTODBPOSITRON)
    p.add_tile(tile_provider)
    p.match_aspect = True

    # layout = column(plot_emb)
    # show(layout)
    grid = gridplot([[plot_emb, p]])
    show(grid)

    number = np.random.randint(1, 1e4)
    # print(f"i've got your number {number}")
    output_file(rf'U:\Users\umaps\species_umap_interactive_{number}.html',
                title=f'species UMAP embedding {number}')
    if u_fp is True:
        plt.savefig(u_fp, bbox_inches='tight', format='png')  # , bbox_extra_artists=(lgd,)

    pix_df_test = df.drop(['class_code', 'species', 'UMAP_1', 'UMAP_2', 'colors'], axis=1)
    pix_df_refl = pix_df_test.loc[:, pix_df_test.columns != 'tree_tag'].apply(pd.to_numeric)
    pix_df_numeric = pd.concat([pix_df_test.loc[:, pix_df_test.columns == 'tree_tag'].reset_index(drop=True),
                                pix_df_refl.reset_index(drop=True)], axis=1)
    tree_crown_mean = pix_df_numeric.groupby('tree_tag').mean()

    return df, tree_crown_mean  # plt.show()


def convert_geopandas_to_bokeh_format(gdf):
    """
    Function to convert a GeoPandas GeoDataFrame to a Bokeh
    ColumnDataSource object.

    :param: (GeoDataFrame) gdf: GeoPandas GeoDataFrame with polygon(s) under
                                the column name 'geometry.'

    :return: ColumnDataSource for Bokeh.
    """
    gdf = gdf[gdf.geom_type != 'MultiPolygon']
    gdf_new = gdf.drop('geometry', axis=1).copy()
    gdf_new['x'] = gdf.apply(get_geometry_coords,
                             geom='geometry',
                             coord_type='x',
                             shape_type='polygon',
                             axis=1)

    gdf_new['y'] = gdf.apply(get_geometry_coords,
                             geom='geometry',
                             coord_type='y',
                             shape_type='polygon',
                             axis=1)

    return ColumnDataSource(gdf_new)


def get_geometry_coords(feature, geom, coord_type, shape_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.

    :param: (GeoPandas Series) row : The row of each of the GeoPandas DataFrame.
    :param: (str) geom : The column name.
    :param: (str) coord_type : Whether it's 'x' or 'y' coordinate.
    :param: (str) shape_type
    """

    # Parse the exterior of the coordinate
    if shape_type == 'polygon':
        exterior = feature[geom].exterior
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list(exterior.coords.xy[0])

        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list(exterior.coords.xy[1])

    elif shape_type == 'point':
        exterior = feature[geom]

        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return exterior.coords.xy[0][0]

        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return exterior.coords.xy[1][0]
