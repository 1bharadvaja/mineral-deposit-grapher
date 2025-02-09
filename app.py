from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import torch
import random
import plotly.graph_objects as go
import plotly.offline as pyo
from torch import nn
from torch.utils.data import Dataset, DataLoader
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def compute_sample_coordinates(df):
    """
    Compute 3D coordinates for each assay sample.

    The collar provides the surface (or entry) coordinates (Easting, Northing, Elevation).
    Each sample interval has a 'From_m' and 'To_m' along the drill hole.
    We compute the mid-depth along the drill hole, then use the hole's dip and azimuth
    (from the collar data) to compute horizontal and vertical offsets.

    Parameters:
        df (pd.DataFrame): Merged DataFrame containing collar and sample data.

    Returns:
        np.ndarray: x, y, z coordinates and the target metric (e.g., Zn_pct) as 1D arrays.
    """
    # Compute mid-depth for each sample
    df['MidDepth_m'] = (df['From_m'] + df['To_m']) / 2.0

    # Convert angles from degrees to radians.
    # Assume "Dip" is negative (downward). Use the absolute value for computing distances.
    dip_rad = np.deg2rad(np.abs(df['Dip'].astype(float)))
    az_rad  = np.deg2rad(df['Azimuth'].astype(float))

    # Compute horizontal distance along the drill hole (projected onto a horizontal plane)
    horizontal_distance = df['MidDepth_m'] * np.cos(dip_rad)
    # Compute vertical distance (how far down the sample is from the collar elevation)
    vertical_distance   = df['MidDepth_m'] * np.sin(dip_rad)

    # Offsets in the horizontal plane.
    # Assume azimuth is measured in degrees clockwise from north.
    east_offset  = horizontal_distance * np.sin(az_rad)
    north_offset = horizontal_distance * np.cos(az_rad)

    # Compute final 3D coordinates:
    # x: collar Easting + east_offset
    # y: collar Northing + north_offset
    # z: collar Elevation minus vertical_distance (since vertical_distance is downward)
    x = df['Easting'].astype(float) + east_offset
    y = df['Northing'].astype(float) + north_offset
    z = df['Elevation'].astype(float) - vertical_distance

    # For this example we use "Zn_pct" as the target metric. Adjust as needed.
    target = df['Zn_pct'].astype(float).values

    return x.values, y.values, z.values, target



def load_and_merge(collar_csv, sample_csv):
    """
    Load the collar and sample CSV files and merge them on "HoleID".

    Parameters:
        collar_csv (str): Path to the collar CSV.
        sample_csv (str): Path to the sample CSV.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Assume the CSV files are comma-separated
    collar_df = pd.read_csv(collar_csv, encoding='utf-8-sig')
    sample_df = pd.read_csv(sample_csv, encoding='utf-8-sig')

    # Strip whitespace from column names
    collar_df.columns = [col.strip() for col in collar_df.columns]
    sample_df.columns = [col.strip() for col in sample_df.columns]

    # Merge on "HoleID". Use inner join so only holes with both collar and sample data are kept.
    merged = pd.merge(sample_df, collar_df, on="HoleID", how="inner")
    return merged


class MineralDataset(Dataset):
    """
    PyTorch Dataset for the mineral data.
    Loads the CSV files, computes 3D coordinates, and prepares input/target pairs.
    """
    def __init__(self, collar_csv, sample_csv):
        self.df = load_and_merge(collar_csv, sample_csv)
        # Use all available data (do not subsample)
        self.x, self.y, self.z, self.targets = compute_sample_coordinates(self.df)
        # Stack coordinates into a (N, 3) array.
        self.data = np.column_stack((self.x, self.y, self.z))

        nan_mask = ~np.isnan(self.data).any(axis=1) & ~np.isnan(self.targets)
        if np.sum(~nan_mask) > 0:
            print(f"Removing {np.sum(~nan_mask)} datapoints containing NaN values.")
        self.data = self.data[nan_mask]
        self.targets = self.targets[nan_mask]




    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # This is an array: [x, y, z]
        target = self.targets[idx]
        # Convert to torch tensors.
        sample = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return sample, target

class MLP(nn.Module):
    """
    A simple multilayer perceptron (MLP) to predict Zn_pct from 3D coordinates.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


@app.route('/upload', methods=['POST'])
def upload():
    # 'datafile' is the name attribute in the HTML input element.
    uploaded_files = request.files.getlist('datafile')

    for file in uploaded_files:
        if file.filename != '':
            # Process each file here.
            # For example, you might save the file:
            file.save(f"uploads/{file.filename}")
    threshold = 0
    
    merged = load_and_merge(f"uploads/{uploaded_files[0].filename}", f"uploads/{uploaded_files[2].filename}")
    filtered_df = merged[merged['Zn_pct'].astype(float) > threshold]
    if filtered_df.empty:
        print(f"No data points found with Zn_pct above {threshold}.")
        return
    x, y, z, target = compute_sample_coordinates(filtered_df)

    x_lower_interval = (np.min(x), np.max(x) - 10000)
    print((np.min(x), np.max(x) - 400))
    #y_lower_interval = (0, 1300)
    y_lower_interval = (np.min(y), np.max(y) - 10000)
    z_lower_interval = (0, 1200)
    z_lower_interval = (np.min(z), np.max(z) - 10000)
    print((np.min(z), np.max(z) - 400))

    x_width = 10000
    y_width = 10000
    z_width = 10000

    x_min = random.uniform(*x_lower_interval)
    y_min = random.uniform(*y_lower_interval)
    z_min = random.uniform(*z_lower_interval)


    x_range = (x_min, x_min + x_width)    # Easting bounds
    y_range = (y_min, y_min + y_width)   # Northing bounds
    z_range = (z_min, z_min + z_width)       # Elevation bounds (vertical)



    # Set the resolution: number of points along each axis.
    resolution = 20  # You can increase this for a finer grid.

    # Create grid points along each axis.
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    z_vals = np.linspace(z_range[0], z_range[1], resolution)

    # Create a 3D grid of coordinates.
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

    # Flatten the grid into a list of coordinates of shape (N, 3)
    coords = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # ---------------------------------------------------------
    # 2. Load the Trained Model
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=3, hidden_dim=64, output_dim=1).to(device)
    model.load_state_dict(torch.load("mineral_model.pth", map_location=device))
    model.eval()

    # ---------------------------------------------------------
    # 3. Perform Inference
    # ---------------------------------------------------------
    # Convert the grid coordinates to a torch tensor.
    inputs = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        # Model outputs will have shape (N, 1); flatten to (N,)
        predictions = model(inputs.to(device)).cpu().numpy().flatten()

    # ---------------------------------------------------------
    # 4. Plot the Inferred Data in 3D
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot. The color of each point corresponds to the predicted Zn_pct.
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=predictions, cmap='viridis', marker='o')
    pred_3d = predictions.reshape(X.shape)


    mask = pred_3d >= threshold

    trace = go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=pred_3d.flatten(),
        isomin=50,
        opacity=1,
        surface_count=2,
        colorscale='Viridis'
    )
    
    colors = np.empty(mask.shape, dtype=object)
    colors[mask] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask, facecolors=colors, edgecolor='k')



    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation")
    #fig.colorbar(sc, ax=ax, label="Predicted Zn_pct")
    #plt.clim(0, 50)
    plt.title("3D Inference of Mineral Deposit (Predicted Zn_pct)")

    buf = BytesIO()
    plt.savefig(buf, format = 'png')
    buf.seek(0)

    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)

    layout = go.Layout(
        title="Predicted Zn_pct",
        scene=dict(
            xaxis=dict(title="X Axis"),
            yaxis=dict(title="Y Axis"),
            zaxis=dict(title="Z Axis")
        )
    )
    fig = go.Figure(data=[trace], layout=layout)

    # Render the result page and pass the base64 image data.
    plot_div = pyo.plot(fig, output_type='div', include_plotlyjs=True)

    return render_template('result.html', plot=plot_div, image_data = image_base64)


if __name__ == 'main':
    app.run(debug=True)








        