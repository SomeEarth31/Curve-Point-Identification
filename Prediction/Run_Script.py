import numpy as np
import pandas as pd
from scipy.stats import circmean

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.exceptions import ConvergenceWarning

import sys
import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def add_surrounding_points_features(df):
    # Create columns for previous two points and next two points
    df['x_prev_2'] = df['X'].shift(2)
    df['y_prev_2'] = df['Y'].shift(2)
    df['x_prev_1'] = df['X'].shift(1)
    df['y_prev_1'] = df['Y'].shift(1)
    df['x_next_1'] = df['X'].shift(-1)
    df['y_next_1'] = df['Y'].shift(-1)
    df['x_next_2'] = df['X'].shift(-2)
    df['y_next_2'] = df['Y'].shift(-2)
    
    return df

def adaptive_sparsity_measure(row, dense_spacing=10):
    distances = [
        np.linalg.norm(np.array([row['X'] - row['x_prev_1'], row['Y'] - row['y_prev_1'], 0])),
        np.linalg.norm(np.array([row['x_next_1'] - row['X'], row['y_next_1'] - row['Y'], 0]))
    ]
    median_distance = np.median(distances)
    return median_distance / dense_spacing

def local_density_variation(row):
    distances = [
        np.linalg.norm(np.array([row['x_prev_1'] - row['x_prev_2'], row['y_prev_1'] - row['y_prev_2'], 0])),
        np.linalg.norm(np.array([row['X'] - row['x_prev_1'], row['Y'] - row['y_prev_1'], 0])),
        np.linalg.norm(np.array([row['x_next_1'] - row['X'], row['y_next_1'] - row['Y'], 0])),
        np.linalg.norm(np.array([row['x_next_2'] - row['x_next_1'], row['y_next_2'] - row['y_next_1'], 0]))
    ]
    return np.std(distances) / np.mean(distances)

def compute_curvature(p1, p2, p3):
    points = np.array([p1, p2, p3])
    
    # Check if points are collinear
    if np.abs(np.cross(points[1] - points[0], points[2] - points[0])) < 1e-8:
        return 0  # Return 0 curvature for straight lines
    
    # Fit a circle to the points
    x_m = np.mean(points[:, 0])
    y_m = np.mean(points[:, 1])
    u = points[:, 0] - x_m
    v = points[:, 1] - y_m
    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuv + Suvv, Svvv + Suuu])/2
    
    try:
        uc, vc = np.linalg.solve(A, B)
        xc = x_m + uc
        yc = y_m + vc
        R = np.sqrt((points[:, 0]-xc)**2 + (points[:, 1]-yc)**2).mean()
        return 1/R if R > 1e-10 else 0
    except np.linalg.LinAlgError:
        # If matrix is singular, points are likely collinear
        return 0


def compute_local_linearity(p1, p2, p3, p4, p5):
    points = np.array([p1, p2, p3, p4, p5])
    X = points[:, 0].reshape(-1, 1)
    Y = points[:, 1]
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model = HuberRegressor(max_iter=100, epsilon=1.5).fit(X, Y)
        r_squared = model.score(X, Y)
    except Exception:
        # If HuberRegressor fails, fall back to LinearRegression
        model = LinearRegression().fit(X, Y)
        r_squared = model.score(X, Y)
    
    return r_squared


def compute_angle_consistency(p1, p2, p3, p4, p5):

    def angle_between(v1, v2):
        v1_3d = np.append(v1, 0)
        v2_3d = np.append(v2, 0)
        return np.arctan2(np.linalg.norm(np.cross(v1_3d, v2_3d)), np.dot(v1, v2))

    vectors = [
        np.array(p2) - np.array(p1),
        np.array(p3) - np.array(p2),
        np.array(p4) - np.array(p3),
        np.array(p5) - np.array(p4)
    ]
    
    angles = [angle_between(vectors[i], vectors[i+1]) for i in range(len(vectors)-1)]
    
    # Use circular mean and standard deviation
    mean_angle = circmean(angles)
    angle_deviation = np.std(angles)
    
    # Normalize consistency measure
    consistency = 1 - (angle_deviation / np.pi)
    
    return consistency

def compute_angle(p1, p2, p3, epsilon=1e-6):
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1], 0])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], 0])
    
    # Use cross product for more stable angle calculation
    cross_prod = np.cross(v1, v2)
    dot_prod = np.dot(v1, v2)
    
    angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
    return np.degrees(angle)

def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p2) - np.array(p1))

def compute_slope(p1, p2, epsilon=1e-6):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx + epsilon)

def compute_slope_change(p1, p2, p3):
    slope1 = compute_slope(p1, p2)
    slope2 = compute_slope(p2, p3)
    return np.abs(np.arctan2(np.sin(slope2-slope1), np.cos(slope2-slope1)))

features = ['Angle', 'Distance','Slope_Change',
            'Curvature','Angle_Consistency','Local_Linearity']

def Calc_Features(df2, add, mult):
    # Add features to the dataframe
    angles2 = []
    distances2 = []
    curvature2 = []
    slope_change2 = []
    angle_consistency2 = []
    local_linearity2 = []
    for i in range(len(df2)):  # Starting from the second point and ending at the second-to-last point
        p5 = (df2['x_prev_2'][i], df2['y_prev_2'][i])
        p1 = (df2['x_prev_1'][i], df2['y_prev_1'][i])
        p2 = (df2['X'][i], df2['Y'][i])
        p3 = (df2['x_next_1'][i], df2['y_next_1'][i])
        p4 = (df2['x_next_2'][i], df2['y_next_2'][i])

        angles2.append(compute_angle(p1, p2, p3))
        curvature2.append(compute_curvature(p1,p2,p3))
        slope_change2.append(compute_slope_change(p1,p2,p3))
        distances2.append(compute_distance(p2, p1))
        angle_consistency2.append(compute_angle_consistency(p5,p1,p2,p3,p4))
        local_linearity2.append(compute_local_linearity(p5,p1,p2,p3,p4))

    # Add the features to the dataframe
    df2['Angle'] = angles2
    df2['Curvature'] = curvature2
    df2['Slope_Change'] = slope_change2
    df2['Distance'] = distances2
    df2['Angle_Consistency'] = angle_consistency2
    df2['Local_Linearity'] =local_linearity2

    adaptive_sparsity = df2['adaptive_sparsity'].to_numpy()
    local_density_variation = df2['local_density_variation'].to_numpy()

    #print(df2)

    df2 = df2[features]
    df2 = (df2 - add)/mult 
    df2['local_density_variation'] = local_density_variation
    df2['adaptive_sparsity'] = adaptive_sparsity
    return df2

# Defining main function
def main():
    if len(sys.argv) == 1:
        name = "input.csv"
    else:
        name = sys.argv[1]
    print("Processing data...")

    df1 = pd.read_csv(name)

    df = df1.copy() 
    df = add_surrounding_points_features(df)
    df = df.dropna()
    df['adaptive_sparsity'] = df.apply(adaptive_sparsity_measure, axis=1)
    df['local_density_variation'] = df.apply(local_density_variation, axis=1)
    df = df.reset_index(drop=True)
    print("Loaded data with spatial calcaultions:")
    print(df)


    add = pd.read_pickle('add_coef.pkl')
    mult = pd.read_pickle('mult_coef.pkl')
    X = Calc_Features(df, add, mult)
    print("Spatial Features from the loaded data set:")
    print(X)

    # Load the model from the file
    loaded_model = joblib.load('random_forest_model.pkl')

    # Predict using the loaded model
    y_pred_loaded = loaded_model.predict(X)
    
    y_pred = np.concatenate(([y_pred_loaded[0]],[y_pred_loaded[0]], y_pred_loaded, [y_pred_loaded[-1]],[y_pred_loaded[-1]]))

    df1['Curve'] = y_pred

    print("Output: ")
    print(df1)

    df1.to_csv('output.csv', index=False)

    print("Done! Output saved to output.csv")

if __name__ == "__main__":
    main()
