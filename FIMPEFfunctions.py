
#%%
#Defining the libraries
import numpy as np
import rasterio
import plotly.graph_objects as go
from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame
from shapely.ops import unary_union
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import geopandas as gpd
from plotly.subplots import make_subplots
from rasterio.plot import show,show_hist
from rasterio.warp import calculate_default_transform, reproject, Resampling,transform
from rasterio.mask import mask
from rasterio import features
from rasterio.features import shapes, geometry_mask
from rasterio.io import MemoryFile
from rasterio.crs import CRS
from shapely.geometry import shape,box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import time
from shapely.geometry import mapping, shape, Polygon
from pyproj import Transformer
from pathlib import Path
import os


# User-defined AOI
def AOI(benchmark_path, shapefile_path, save_dir):
    with rasterio.open(benchmark_path) as src:
        data = src.read(1)
        
        if data.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
            data = data.astype('float32')   
        crs = src.crs  
        bounding_geom = gpd.read_file(shapefile_path)
        bounding_geom = bounding_geom.to_crs(crs)
        
        bounding_geom = [geom for geom in bounding_geom.geometry]
        
    return bounding_geom

#%% Determination of Smallest extent

def get_smallest_raster_path(benchmark_path, *candidate_paths):
    def get_raster_shape(raster_path):
        with rasterio.open(raster_path) as src:
            return src.shape
    all_paths = [benchmark_path] + list(candidate_paths)
    smallest_raster = None
    smallest_size = float('inf')

    for raster_path in all_paths:
        shape = get_raster_shape(raster_path)
        size = shape[0] * shape[1]
        if size < smallest_size:
            smallest_size = size
            smallest_raster = raster_path
    return smallest_raster

def smallest_extent(raster_path, save_dir):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs.to_string()
    bounding_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    #Save the smallest extent boundary
    boundary_shapefile = os.path.join(save_dir, 'boundary.shp')
    gdf = gpd.GeoDataFrame({'geometry': [bounding_geom]}, crs=crs)
    gdf.to_file(boundary_shapefile, driver='ESRI Shapefile')
    return [mapping(bounding_geom)]
    
# Convex Hull
def convex_hull(raster_path , save_dir):
    #raster_path = get_smallest_raster_path(benchmark_path, *candidate_paths)
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        transform = src.transform
        nodata_value = src.nodata
        crs = src.crs

        if raster_data.dtype not in ['int16', 'int32', 'uint8', 'uint16', 'float32']:
            raster_data = raster_data.astype('float32')
        
    raster_data = np.where(raster_data > 0, 1, 0).astype('uint8')
    mask = raster_data == 1

        # Extract shapes from raster
    feature_generator = shapes(raster_data, mask=mask, transform=transform)
    polygons = [shape(feature[0]) for feature in feature_generator]

        # Create GeoDataFrame from polygons
    gdf = GeoDataFrame({'geometry': polygons}, crs=crs)
    gdf.to_file('Shapes1.shp')
    #gdf = gpd.GeoDataFrame(valid_shapes, geometry='geometry', crs=crs)
    union_geom = unary_union(gdf.geometry)
    bounding_geom = union_geom.convex_hull
    #bounding_geom=[bounding_geom]
    #x=union_geom.convex_hull
    bounding_gdf = gpd.GeoDataFrame({'geometry': [bounding_geom]}, crs=gdf.crs)
    boundary_shapefile = os.path.join(save_dir, 'boundary.shp')
    bounding_gdf.to_file(boundary_shapefile, driver='ESRI Shapefile')

    return [mapping(bounding_geom)]
# %%
def eval(out_image1,out_image2):
    merged=out_image1+out_image2
    unique_values, counts = np.unique(merged, return_counts=True)
    class_pixel_counts = dict(zip(unique_values, counts))
    class_pixel_counts
    TN=class_pixel_counts[1]
    FP=class_pixel_counts[2]
    FN=class_pixel_counts[3]
    TP=class_pixel_counts[4]
    TPR=(TP/(TP+FN))
    FNR=(FN/(TP+FN))
    Acc=(TP+TN)/(TP+TN+FP+FN)
    Prec=TP/(TP+FP)
    sen=TP/(TP+FN)
    F1_score=2*(Prec*sen)/(Prec+sen)
    CSI=(TP/(TP+FN+FP))
    POD=(TP/(TP+FN))
    FPR=(FP/(FP+TN))
    return unique_values,TN,FP,FN,TP,TPR,FNR,Acc,Prec,sen,CSI,F1_score,POD,FPR,merged

# %%
#Function for the evalution of the model
def evaluate_flood_inundation(benchmark_path, candidate_paths, gdf, folder, method, shapefile=None):
    # Lists to store evaluation metrics
    csi_values = []
    TN_values = []
    FP_values = []
    FN_values = []
    TP_values = []
    TPR_values = []
    FNR_values = []
    Acc_values = []
    Prec_values = []
    sen_values = []
    F1_values = []
    POD_values = []
    FPR_values = []
    Merged = []
    Unique = []
    
           # Dynamically call the specified method
    method = globals().get(method)
    if method is None:
        raise ValueError(f"Method '{method}' is not defined.")

    #Save the smallest extent boundary and cliped FIMS
    save_dir = os.path.join(folder, f'{method.__name__}')
    os.makedirs(save_dir, exist_ok=True)

    # Get the smallest matched raster extent and make a boundary shapefile
    smallest_raster_path = get_smallest_raster_path(benchmark_path, *candidate_paths)
    
    if method.__name__== "AOI":
        bounding_geom = AOI(benchmark_path, shapefile,save_dir)
    else:
        bounding_geom= method(smallest_raster_path,save_dir=save_dir)

    # Read and process benchmark raster
    with rasterio.open(benchmark_path) as src1:
        out_image1, out_transform1 = mask(src1, bounding_geom, crop=True, all_touched=True)
        benchmark_nodata = src1.nodata
        benchmark_crs = src1.crs
        b_profile=src1.profile
        out_image1[out_image1 == benchmark_nodata] = 0
        out_image1= np.where(out_image1 > 0, 2, 0).astype(np.float32)
        gdf = gdf.to_crs(benchmark_crs)
        shapes1 = [geom for geom in gdf.geometry]
        mask1 = features.geometry_mask(shapes1, transform=out_transform1, invert=True, out_shape=out_image1.shape[1:])
        extract_b = np.where(mask1, out_image1, 0)
        extract_b = np.where(extract_b > 0, 1, 0)
        idx_pwb = np.where(extract_b == 1)
        out_image1[idx_pwb] = 0
        
        clipped_benchmark = os.path.join(save_dir, 'Benchmark.tif')
        b_profile.update({
         'height': out_image1.shape[1],
        'width': out_image1.shape[2],
         'transform': out_transform1
            })
          
        with rasterio.open(clipped_benchmark , 'w', **b_profile) as dst:
            #dst.transform =out_transform1
            #dst.crs = benchmark_crs
            dst.write(np.squeeze(out_image1), 1)                                

    def resize_image(source_image, source_transform, source_crs, target_crs, target_shape, target_transform):
        target_image = np.empty(target_shape, dtype=source_image.dtype)
        reproject(
            source=source_image,
            destination=target_image,
            src_transform=source_transform,
            dst_transform=target_transform,
            src_crs=source_crs,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )
        return target_image

    # Process each candidate file
    for idx, candidate_path in enumerate(candidate_paths):
        base_name = os.path.splitext(os.path.basename(candidate_path))[0]
        with rasterio.open(candidate_path) as src2:
            candidate = src2.read(1)
            candidate_nodata = src2.nodata
            candidate_transform = src2.transform
            candidate_meta = src2.meta.copy()
            candidate_crs = src2.crs
            c_profile = src2.profile
            candidate[candidate == src2.nodata] = 0 
            candidate = np.where(candidate > 0, 2, 1).astype(np.float32)
            with MemoryFile() as memfile:
                with memfile.open(**candidate_meta) as mem2:
                    mem2.write(candidate, 1)
                    dst_transform, width, height = rasterio.warp.calculate_default_transform(
                        mem2.crs, benchmark_crs, mem2.width, mem2.height, *mem2.bounds
                    )
                    dst_meta = mem2.meta.copy()
                    dst_meta.update({
                        'crs': benchmark_crs,
                        'transform': dst_transform,
                        'width': width,
                        'height': height
                    })

                    with MemoryFile() as memfile_reprojected:
                        with memfile_reprojected.open(**dst_meta) as mem2_reprojected:
                            for i in range(1, mem2.count + 1):
                                reproject(
                                    source=rasterio.band(mem2, i),
                                    destination=rasterio.band(mem2_reprojected, i),
                                    src_transform=mem2.transform,
                                    src_crs=mem2.crs,
                                    dst_transform=dst_transform,
                                    dst_crs=benchmark_crs,
                                    resampling=Resampling.nearest
                                )
                            out_image2, out_transform2 = mask(mem2_reprojected, bounding_geom, crop=True, all_touched=True)
                            out_image2 = np.where(out_image2 == candidate_nodata, 0, out_image2)
                            #out_image2 = np.where(out_image2 == 0, -1, out_image2)
                            #out_image2 = np.where(out_image2 == 3, 2, out_image2)
                             # Save the clipped candidate raster
                            clipped_candidate = os.path.join(save_dir, f'{base_name}.tif')
                            b_profile.update({
                               'height': out_image1.shape[1],
                                'width': out_image1.shape[2],
                                 'transform': out_transform1
                                  })
                            with rasterio.open(clipped_candidate, 'w', **b_profile) as dst:
                                dst.write(np.squeeze(out_image2), 1)
                                #dst.transform = out_transform2
                                #dst.crs = b_profile['crs']
                         
                            
                            mask2 = features.geometry_mask(shapes1, transform=out_transform2, invert=True, out_shape=(out_image2.shape[1], out_image2.shape[2]))
                            extract_c = np.where(mask2, out_image2, 0)
                            extract_c = np.where(extract_c > 0, 1, 0)
                            idx_pwc = np.where(extract_c == 1)
                            out_image2[idx_pwc] = -1
                            out_image2_resized = resize_image(out_image2, out_transform2, mem2_reprojected.crs, benchmark_crs, out_image1.shape, out_transform1) 
                           
                            merged = out_image1 + out_image2_resized
                            
           
                         
   # Get Evaluation Metrics
            unique_values, TN, FP, FN, TP, TPR, FNR, Acc, Prec, sen, CSI, F1_score, POD, FPR, merged = eval(out_image1,                  out_image2_resized)
                            
         # Append values to the lists
            csi_values.append(CSI)
            TN_values.append(TN)
            FP_values.append(FP)
            FN_values.append(FN)
            TP_values.append(TP)
            TPR_values.append(TPR)
            FNR_values.append(FNR)
            Acc_values.append(Acc)
            Prec_values.append(Prec)
            sen_values.append(sen)
            F1_values.append(F1_score)
            POD_values.append(POD)
            FPR_values.append(FPR)
            Merged.append(merged)
            Unique.append(unique_values)

    results = {
        "CSI_values": csi_values,
        "TN_values": TN_values,
        "FP_values": FP_values,
        "FN_values": FN_values,
        "TP_values": TP_values,
        "TPR_values": TPR_values,
        "FNR_values": FNR_values,
        "Acc_values": Acc_values,
        "Prec_values": Prec_values,
        "sen_values": sen_values,
        "F1_values": F1_values,
        "POD_values": POD_values,
        "FPR_values": FPR_values,
        # 'Merged': Merged,
        #  'Unique': Unique
    }
    for candidate_idx, candidate_path in enumerate(candidate_paths):
        base_name = os.path.splitext(os.path.basename(candidate_path))[0]
        merged_raster = Merged[candidate_idx]
    
        # Handle raster dimensions
        if merged_raster.ndim == 3:
            band = merged_raster.squeeze()
        elif merged_raster.ndim == 2:
            band = merged_raster
        else:
            raise ValueError(f"Unexpected number of dimensions in Merged[{candidate_idx}].")
    
    # Construct the file name dynamically
        output_filename = os.path.join(save_dir, f"Contingency_{base_name}.tif")
    
    # Save the raster
        with rasterio.open(output_filename, "w", **b_profile) as dst:
            dst.write(band, 1)
            dst.transform = out_transform1
            dst.crs = benchmark_crs


               
    # Dynamically assign column names based on the number of candidate paths
    #Saving it into dataframe
    candidate_names = [os.path.splitext(os.path.basename(path))[0] for path in candidate_paths]
    df = pd.DataFrame.from_dict(results, orient='index')
    num_candidates = len(candidate_paths)
    #df.columns = [f'FIM{idx + 1}' for idx in range(num_candidates)]
    df.columns = candidate_names
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Metrics'}, inplace=True)
    # Save the DataFrame
    csv_file = os.path.join(save_dir, 'EvaluationMetrics.csv')
    df.to_csv(csv_file, index=False)
    print(f'Evaluation metrics saved to {csv_file}')
    return results
# %%
#Function to plot the metrics of the FIM
def PlotMetrics(csv_path, save_dir, method_name):
    metrics_df = pd.read_csv(csv_path)
    # Extract relevant metrics
    metrics = metrics_df.loc[metrics_df['Metrics'].isin(['CSI_values', 'POD_values', 'Acc_values', 'Prec_values', 'F1_values'])].copy()
    # Renaming fot the plot
    metrics.loc[:, 'Metrics'] = metrics['Metrics'].replace({
        'CSI_values': 'CSI',
       'POD_values': 'POD',
        'Acc_values': 'Accuracy',
       'Prec_values': 'Precision',
        'F1_values': 'F1 Score'
    })
    # Dynamically determine the numeric column (assume the first numeric column is the score column)
    score_column = metrics.select_dtypes(include='number').columns[2]
    # Round the scores for better presentation
    metrics[score_column] = metrics[score_column].round(2)
    # Create the bar plot
    fig = px.bar(
        metrics,
        x=score_column,
        y='Metrics',
        title='Performance Metrics',
        labels={score_column: 'Score'},
        text=score_column,
        color='Metrics',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    # Update layout for better aesthetics
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    # Set the background colors to transparent
    fig.update_layout(
    yaxis_title='Metrics',
    xaxis_title='Score',
    showlegend=False,
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showline=True, linewidth=2, linecolor='black'),
    yaxis=dict(showline=True, linewidth=2, linecolor='black'),
    height=350,
    width=900,
    title_font=dict(family='Arial', size=24, color='black'),  # Removed weight
    xaxis_title_font=dict(family='Arial', size=20, color='black'),
    yaxis_title_font=dict(family='Arial', size=20, color='black'),
    font=dict(family='Arial', size=18, color='black')
      )
    # Adding a black border
    fig.add_trace(px.scatter(x=[None], y=[None]).data[0])
    fig.update_traces(marker=dict(line=dict(color='black', width=1)))
    # Save the plot as a PNG
    folder_name = os.path.basename(os.path.normpath(save_dir))
    if folder_name == method_name:
        final_dir = os.path.join(save_dir, f"{method_name}")
        print(final_dir)
        output_path = os.path.join(final_dir, 'PerformanceMetrics.png')
        fig.write_image(output_path, scale= 500/96)
        fig.write_image(output_path)
        print(f"Performance metrics chart is saved as PNG at {output_path}")
        fig.show()
    else:
        print(f"There is no folder with {method_name}. Please Check twice")
# %%
def getContingencyMap(raster_path):
    # Load the raster
    with rasterio.open(raster_path) as src:
        band1 = src.read(1)
        transform = src.transform
        src_crs = src.crs
        nodata_value = src.nodatavals[0] if src.nodatavals else None  # Get the NoData value if available

# Create a new array for color mapping, default to "No Data" (1)
    combined_flood = np.full_like(band1, fill_value=1, dtype=int)  # Initialize with "No Data"

# Map pixel values to colors
    combined_flood[band1 == -1] = 0  # Black (Permanent Water)
    combined_flood[band1 == 0] = 1   # White (No Data)
    combined_flood[band1 == 1] = 2   # Grey (True Negative)
    combined_flood[band1 == 2] = 3   # Green (False Positive)
    combined_flood[band1 == 3] = 4   # Blue (False Negative)
    combined_flood[band1 == 4] = 5   # Red (True Positive)

# Handle NoData explicitly, mapping it to "No Data" class (1)
    if nodata_value is not None:
        combined_flood[band1 == nodata_value] = 1

    rows, cols = np.indices(band1.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)

    dst_crs = "EPSG:4326"
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    flat_xs, flat_ys = xs.ravel(), ys.ravel()
    longitudes, latitudes = transformer.transform(flat_xs, flat_ys)
    xs_dd = np.array(longitudes).reshape(xs.shape)
    ys_dd = np.array(latitudes).reshape(ys.shape)

    # Define the color map and normalization
    flood_colors = ["black", "white", "grey", "green", "blue", "red"]  # 6 classes
    flood_cmap = mcolors.ListedColormap(flood_colors)
    flood_norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 6.5, 1), ncolors=len(flood_colors))

# Plot the raster with transformed coordinates
    plt.figure(figsize=(12, 11))
    plt.imshow(combined_flood, cmap=flood_cmap, norm=flood_norm, interpolation="none", extent=(
        xs_dd.min(), xs_dd.max(), ys_dd.min(), ys_dd.max()
    ))

# Create legend patches
    value_labels = {
    0: "Permanent Water",
    1: "No Data",
    2: "True Negative",
    3: "False Positive",
    4: "False Negative",
    5: "True Positive"
    }
    legend_patches = [
        Patch(color=flood_colors[i], label=value_labels[i], edgecolor="black", linewidth=1.5)
        for i in range(len(flood_colors))
    ]

# Add legend and labels
    plt.legend(handles=legend_patches, loc="lower left")
    plt.xlabel("Longitude", fontsize=14, fontweight="bold")
    plt.ylabel("Latitude", fontsize=14, fontweight="bold")
    plt.tick_params(axis="both", labelsize=14, width=1.5)

# Adjust tick formatting
    x_ticks = np.linspace(xs_dd.min(), xs_dd.max(), 5)
    y_ticks = np.linspace(ys_dd.min(), ys_dd.max(), 5)
    plt.xticks(x_ticks, [f"{abs(tick):.2f}" for tick in x_ticks])
    plt.yticks(y_ticks, [f"{abs(tick):.2f}" for tick in y_ticks])
    plt.legend(handles=legend_patches, loc='lower left')
    plt.xlabel('Longitude', fontsize=14, fontweight='bold')
    plt.ylabel('Latitude', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', which='both', labelsize='14', width=1.5)
    # Save the plot in the same directory as the raster with 500 DPI
    output_path = os.path.join(os.path.dirname(raster_path), 'Contingency_Map.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')

    plt.show()


#%%
def Changeintogpkg(input_path, output_dir, layer_name):
    input_path = str(input_path)
    # Check if the file is already in GPKG format
    if input_path.endswith('.gpkg'):
        return input_path
    else:
        # Convert to GPKG format if it's not
        gdf = gpd.read_file(input_path)
        output_gpkg = os.path.join(output_dir, f'{layer_name}.gpkg')
        gdf.to_file(output_gpkg, driver='GPKG')
        return output_gpkg

# Get the Flooded Building Count

def GetFloodedBuildingCountInfo(building_fp_path, study_area_path, raster1_path, raster2_path, contingency_map,save_dir, method_name):
    output_dir = os.path.dirname(building_fp_path)
    
    # Convert files to GPKG if necessary
    building_fp_gpkg = Changeintogpkg(building_fp_path, output_dir, 'building_footprint')
    
    # Load the building footprint
    building_gdf = gpd.read_file(building_fp_gpkg)
    study_area_gdf = gpd.read_file(study_area_path)
    
    # Reproject both layers to the same CRS
    if building_gdf.crs != study_area_gdf.crs:
        building_gdf = building_gdf.to_crs(study_area_gdf.crs)
    
    # Clip the building footprint to the study area
    clipped_buildings = gpd.overlay(building_gdf, study_area_gdf, how='intersection')
    clipped_buildings['centroid'] = clipped_buildings.geometry.centroid
    
    # Initialize a dictionary to store the counts
    centroid_counts = {'Benchmark': 0, 'Candidate': 0, 'False Positive': 0, 'False Negative': 0, 'True Positive': 0}

    def count_centroids_in_raster(raster_path, label):
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            transform = src.transform
            
            for centroid in clipped_buildings['centroid']:
                # Get row, col of centroid in raster space
                row, col = src.index(centroid.x, centroid.y)
                
                # Check if the value at that location matches the expected pixel values
                if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
                    pixel_value = raster_data[row, col]
                    if label in ['Benchmark', 'Candidate']:
                        if pixel_value == 2:  # False Positive
                            centroid_counts[label] += 1
                    else:
                        if pixel_value == 2:
                            centroid_counts['False Positive'] += 1
                        elif pixel_value == 3:
                            centroid_counts['False Negative'] += 1
                        elif pixel_value == 4:
                            centroid_counts['True Positive'] += 1
   
    # Identify Benchmark and Candidate rasters based on file name
    if 'benchmark' in str(raster1_path).lower():
        count_centroids_in_raster(raster1_path, 'Benchmark')
        count_centroids_in_raster(raster2_path, 'Candidate')
    elif 'candidate' in str(raster2_path).lower():
        count_centroids_in_raster(raster1_path, 'Candidate')
        count_centroids_in_raster(raster2_path, 'Benchmark')
        
    # Count for the third raster (contingency map)
    if 'contingency' in str(contingency_map).lower():
        count_centroids_in_raster(contingency_map, 'Contingency')

    # Percentage calculation
    total_buildings = len(clipped_buildings)
    percentages = {key: (count / total_buildings) * 100 for key, count in centroid_counts.items()}
    
    # Prepare data for the second plot (third raster counts)
    third_raster_labels = ['False Positive', 'False Negative', 'True Positive']
    third_raster_counts = [centroid_counts['False Positive'], centroid_counts['False Negative'], centroid_counts['True Positive']]
    
    # Plotting the result using Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Building Counts on Different FIMs", "Contingency Flooded Building Counts"))

    # Add Candidate bar for the first plot
    fig.add_trace(go.Bar(
        x=['Candidate'], y=[centroid_counts['Candidate']], 
        text=[f"{centroid_counts['Candidate']}"], 
        textposition='auto',
        marker_color='#1c83eb', 
        marker_line_color='black', 
        marker_line_width=1,
        name=f"Candidate ({percentages['Candidate']:.2f}%)"
    ), row=1, col=1)

    # Add Benchmark bar for the first plot
    fig.add_trace(go.Bar(
        x=['Benchmark'], y=[centroid_counts['Benchmark']], 
        text=[f"{centroid_counts['Benchmark']}"], 
        textposition='auto',
        marker_color='#a4490e', 
        marker_line_color='black', 
        marker_line_width=1,
        name=f"Benchmark ({percentages['Benchmark']:.2f}%)"
    ), row=1, col=1)

    # Add bars for the second plot (third raster counts)
    for i in range(len(third_raster_labels)):
        fig.add_trace(go.Bar(
            x=[third_raster_labels[i]], y=[third_raster_counts[i]], 
            text=[f"{third_raster_counts[i]}"], 
            textposition='auto',
            marker_color=['#ff5733', '#ffc300', '#28a745'][i],
            marker_line_color='black', 
            marker_line_width=1,
            name=f"{third_raster_labels[i]} ({percentages[third_raster_labels[i]]:.2f}%)"
        ), row=1, col=2)

    # Customizing layout
    fig.update_layout(
    title="Flooded Building Counts",
    xaxis_title="Inundation Surface",
    yaxis_title="Flooded Building Counts",
    width=1100,  
    height=400, 
    xaxis=dict(showline=True, linewidth=2, linecolor='black'), 
    yaxis=dict(showline=True, linewidth=2, linecolor='black'),
    xaxis2=dict(showline=True, linewidth=2, linecolor='black'),
    yaxis2=dict(showline=True, linewidth=2, linecolor='black'),  
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',  
    showlegend=True,
    title_font=dict(family='Arial', size=24, color='black'),  # Removed weight
    xaxis_title_font=dict(family='Arial', size=20, color='black'),  # Removed weight
    yaxis_title_font=dict(family='Arial', size=20, color='black'),  # Removed weight
    font=dict(family='Arial', size=18, color='black') 
        )
    
    # Save counts to CSV
    counts_data = {
        'Category': ['Candidate', 'Benchmark', 'False Positive', 'False Negative', 'True Positive'],
        'Building Count': [centroid_counts['Candidate'], centroid_counts['Benchmark'], 
                           centroid_counts['False Positive'], centroid_counts['False Negative'], 
                           centroid_counts['True Positive']]
    }
    counts_df = pd.DataFrame(counts_data)
    csv_file_path = os.path.join(output_dir, 'FBCounts.csv')
    counts_df.to_csv(csv_file_path, index=False)
    # Save the plot as PNG
    #output_path = os.path.join(os.path.dirname(raster1_path), 'BuildingCounts.png')
    folder_name = os.path.basename(os.path.normpath(save_dir))
    final_dir = os.path.join(save_dir, f"{method_name}")
    print(final_dir)
    output_path = os.path.join(final_dir, 'BuildingCounts.png')
    fig.write_image(output_path, scale= 500/96)
    fig.write_image(output_path)
    print(f"Performance metrics chart is saved as PNG at {output_path}")
    fig.show()
    

#%%