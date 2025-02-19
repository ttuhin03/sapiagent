import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.max_rows', None)      # Show all rows if needed (optional)
pd.set_option('display.width', None)         # Let the notebook handle line wrapping
pd.set_option('display.max_colwidth', None)  


def smoothness(values):
    n = len(values)
    total_diff = 0.0

    if n < 2:
        return 0.0

    for i in range(1, n):
        total_diff += abs(values[i] - values[i-1])
    return total_diff / (n - 1)



end_df = pd.DataFrame()

import os

# Base path where your user directories are located
pfadDaten = "/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/sapimouse_ownhumandata"

# List to collect CSV file paths
csv_paths = []

# Walk through the directory tree
for root, dirs, files in os.walk(pfadDaten):
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)
            csv_paths.append(full_path)


for path in csv_paths:

    pfadDaten = path#"/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/sapimouse_ownhumandata/user4/session_2024_12_30_3min.csv"

    df = pd.read_csv(pfadDaten)


    # 1. Sort by timestamp if not sorted
    df = df.sort_values('client timestamp')

    # 2. Compute the time difference to the previous row
    df['time_diff'] = df['client timestamp'].diff().fillna(0)

    # 3. Define a "new_chunk" marker where conditions are met
    #    Condition A: time gap > 4000
    #    Condition B: state == 'Released'
    df['new_chunk'] = (
    (df['time_diff'] > 4000)    # large gap
    | (df['state'] == 'Released')  # or row is 'Released'
)

    # 4. Convert that boolean into a cumulative sum
    #    Each True increments the chunk ID
    df['chunk_id'] = df['new_chunk'].cumsum()

# 5. (Optional) If you prefer the row with Released to be 
#    included in the preceding chunk rather than marking 
#    the start of the new chunk, you can adjust the logic as needed. 
#    For example, you might shift the condition or handle it differently.
#    But in this version, chunk_id changes on the same row that has "Released".

# Now each chunk is all rows that have the same chunk_id:
# For example, group by chunk_id:
    groups = df.groupby('chunk_id')

    rows = []

    for chunk_id, group_data in groups:
    # Calculate distance between consecutive points
        group_data['distance'] = (
        (group_data['x'].diff()**2 + group_data['y'].diff()**2) ** 0.5
    ).fillna(0)
    
    # Calculate time_diff between consecutive points; assumed to be in group_data already
    # group_data['time_diff'] = ...
    
    # Calculate per-row velocity to get min and max
    # Avoid division by zero by replacing 0 with NaN or a small number if needed
    # Here, we’ll just replace time_diff == 0 with NaN:
        group_data.loc[group_data['time_diff'] == 0, 'time_diff'] = float('nan')
        group_data['row_velocity'] = group_data['distance'] / group_data['time_diff']

    # Replace NaN back to 0 for velocity, if desired
        group_data['row_velocity'] = group_data['row_velocity'].fillna(0)

    # Differenz der Geschwindigkeiten:
        velocity_diff = group_data['row_velocity'].diff()  # v[i] - v[i-1]

    # Beschleunigung: diff in px/s / time_diff (ms) => px/s^2
        group_data['row_acceleration'] = (velocity_diff / group_data['time_diff']) * 1000000
        group_data['row_acceleration'] = group_data['row_acceleration'].fillna(0)

    # Chunk-level total time
        total_time = group_data['time_diff'].sum(skipna=True)
    # Chunk-level total distance
        total_distance = group_data['distance'].sum()
    # Chunk-level average velocity (distance / time)
        velocity_chunk = total_distance / total_time if total_time > 0 else 0

    # Filter out rows where row_velocity is 0
        nonzero_velocities = group_data.loc[group_data['row_velocity'] != 0, 'row_velocity']

    # Compute the min from the non-zero velocities
        velocity_min = nonzero_velocities.min() if not nonzero_velocities.empty else 0
        velocity_max = group_data['row_velocity'].max()
        velocity_mean = group_data['row_velocity'].mean()
        velocity_var = group_data['row_velocity'].var()

    # Beschleunigung auf Chunk-Ebene
        acc_min = group_data['row_acceleration'].min()
        acc_max = group_data['row_acceleration'].max()
        acc_mean = group_data['row_acceleration'].mean()
        acc_var = group_data['row_acceleration'].var()



    # Direkte Distanz zwischen erstem und letztem Punkt
        first_x, first_y = group_data.iloc[0]['x'], group_data.iloc[0]['y']
        last_x, last_y   = group_data.iloc[-1]['x'], group_data.iloc[-1]['y']
        direct_distance = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)

    # Beispiel-Dauer (Summe aller time_diff)
        duration = total_time  # kann nach Bedarf in andere Einheiten (Min / Std) umgerechnet werden

    # Effizienz als direkte Distanz / Gesamtzeit
    # (falls du eine andere Definition für „Effizienz“ brauchst, entsprechend anpassen)
        efficiency = direct_distance / duration if duration > 0 else 0



    # If you also want the absolute difference (|delta_x|):
        group_data['abs_delta_x'] = group_data['x'].diff().abs()
        smoothness_v1 = group_data['abs_delta_x'].mean()

        
        smoothness_v2 = smoothness(group_data['x'].values)











    # Werte in das rows-Dictionary übernehmen
        rows.append({
        'chunk_id': chunk_id,
        'geschwindigkeit': velocity_chunk * 1000,
        'geschwindigkeit_min': velocity_min * 1000,
        'geschwindigkeit_max': velocity_max * 1000,
        'geschwindigkeit_mean': velocity_mean * 1000,
        'geschwindigkeit_var': velocity_var * 1000,
        'dauer': duration,
        'direkte_distanz': direct_distance,
        'effizienz': efficiency,
        'totale_distanz': total_distance,
        'smoothness_v1': smoothness_v1,
        'smoothness_v2': smoothness_v2,
        'beschleunigung_min':  acc_min,
        'beschleunigung_max':  acc_max,
        'beschleunigung_mean': acc_mean,
        'beschleunigung_var':  acc_var,

    })

# Finally, build your DataFrame from the rows
    df_temp = pd.DataFrame(rows)

# ------------------------------------------------------
# 4) Z-SCORE UND MINMAX-SCALING FÜR JEDES FEATURE
# ------------------------------------------------------
# Legen Sie fest, welche Spalten Sie skalieren möchten
    columns_to_scale = [
    'geschwindigkeit', 'geschwindigkeit_min', 'geschwindigkeit_max',
    'geschwindigkeit_mean', 'geschwindigkeit_var',
    'beschleunigung_min', 'beschleunigung_max',
    'beschleunigung_mean', 'beschleunigung_var',
    'dauer', 'direkte_distanz', 'effizienz', 'totale_distanz',
    'smoothness_v1', 'smoothness_v2'
]

# 4a) Z-SCORE => (X - mean)/std
    for col in columns_to_scale:
        mean_val = df_temp[col].mean()
        std_val  = df_temp[col].std()
    # Falls std_val=0 => Division durch 0 vermeiden
        if std_val == 0:
            df_temp[col + '_zscore'] = 0
        else:
            df_temp[col + '_zscore'] = (df_temp[col] - mean_val) / std_val

# 4b) MIN-MAX-SCALING => (X - min)/(max - min)
    for col in columns_to_scale:
        min_val = df_temp[col].min()
        max_val = df_temp[col].max()
        if max_val == min_val:
            df_temp[col + '_minmax'] = 0  # oder 1
        else:
            df_temp[col + '_minmax'] = (df_temp[col] - min_val) / (max_val - min_val)

       # Append df_temp to end_df
    end_df = pd.concat([end_df, df_temp], ignore_index=True)



end_df['is_anomaly'] = 0
end_df = end_df.dropna()
print("Human Dataframe Shape")
print(end_df.shape)





bot_df = pd.DataFrame()

import os

# Base path where your user directories are located
pfadDaten = "/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/feature_berechnung_ownhuman/bot_nutzbar"

# List to collect CSV file paths
csv_paths = []

# Walk through the directory tree
for root, dirs, files in os.walk(pfadDaten):
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)
            csv_paths.append(full_path)


for path in csv_paths:

    pfadDaten = path#"/Users/tuhin/Desktop/Bachelorarbeit/sapiagent/sapimouse_ownhumandata/user4/session_2024_12_30_3min.csv"

    df = pd.read_csv(pfadDaten)


    # 1. Sort by timestamp if not sorted
    df = df.sort_values('client timestamp')

    # 2. Compute the time difference to the previous row
    df['time_diff'] = df['client timestamp'].diff().fillna(0)

    # 3. Define a "new_chunk" marker where conditions are met
    #    Condition A: time gap > 4000
    #    Condition B: state == 'Released'
    df['new_chunk'] = (
    (df['time_diff'] > 4000)    # large gap
    | (df['state'] == 'Released')  # or row is 'Released'
)

    # 4. Convert that boolean into a cumulative sum
    #    Each True increments the chunk ID
    df['chunk_id'] = df['new_chunk'].cumsum()

# 5. (Optional) If you prefer the row with Released to be 
#    included in the preceding chunk rather than marking 
#    the start of the new chunk, you can adjust the logic as needed. 
#    For example, you might shift the condition or handle it differently.
#    But in this version, chunk_id changes on the same row that has "Released".

# Now each chunk is all rows that have the same chunk_id:
# For example, group by chunk_id:
    groups = df.groupby('chunk_id')

    rows = []

    for chunk_id, group_data in groups:
    # Calculate distance between consecutive points
        group_data['distance'] = (
        (group_data['x'].diff()**2 + group_data['y'].diff()**2) ** 0.5
    ).fillna(0)
    
    # Calculate time_diff between consecutive points; assumed to be in group_data already
    # group_data['time_diff'] = ...
    
    # Calculate per-row velocity to get min and max
    # Avoid division by zero by replacing 0 with NaN or a small number if needed
    # Here, we’ll just replace time_diff == 0 with NaN:
        group_data.loc[group_data['time_diff'] == 0, 'time_diff'] = float('nan')
        group_data['row_velocity'] = group_data['distance'] / group_data['time_diff']

    # Replace NaN back to 0 for velocity, if desired
        group_data['row_velocity'] = group_data['row_velocity'].fillna(0)

    # Differenz der Geschwindigkeiten:
        velocity_diff = group_data['row_velocity'].diff()  # v[i] - v[i-1]

    # Beschleunigung: diff in px/s / time_diff (ms) => px/s^2
        group_data['row_acceleration'] = (velocity_diff / group_data['time_diff']) * 1000000
        group_data['row_acceleration'] = group_data['row_acceleration'].fillna(0)

    # Chunk-level total time
        total_time = group_data['time_diff'].sum(skipna=True)
    # Chunk-level total distance
        total_distance = group_data['distance'].sum()
    # Chunk-level average velocity (distance / time)
        velocity_chunk = total_distance / total_time if total_time > 0 else 0

    # Filter out rows where row_velocity is 0
        nonzero_velocities = group_data.loc[group_data['row_velocity'] != 0, 'row_velocity']

    # Compute the min from the non-zero velocities
        velocity_min = nonzero_velocities.min() if not nonzero_velocities.empty else 0
        velocity_max = group_data['row_velocity'].max()
        velocity_mean = group_data['row_velocity'].mean()
        velocity_var = group_data['row_velocity'].var()

    # Beschleunigung auf Chunk-Ebene
        acc_min = group_data['row_acceleration'].min()
        acc_max = group_data['row_acceleration'].max()
        acc_mean = group_data['row_acceleration'].mean()
        acc_var = group_data['row_acceleration'].var()



    # Direkte Distanz zwischen erstem und letztem Punkt
        first_x, first_y = group_data.iloc[0]['x'], group_data.iloc[0]['y']
        last_x, last_y   = group_data.iloc[-1]['x'], group_data.iloc[-1]['y']
        direct_distance = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)

    # Beispiel-Dauer (Summe aller time_diff)
        duration = total_time  # kann nach Bedarf in andere Einheiten (Min / Std) umgerechnet werden

    # Effizienz als direkte Distanz / Gesamtzeit
    # (falls du eine andere Definition für „Effizienz“ brauchst, entsprechend anpassen)
        efficiency = direct_distance / duration if duration > 0 else 0



    # If you also want the absolute difference (|delta_x|):
        group_data['abs_delta_x'] = group_data['x'].diff().abs()
        smoothness_v1 = group_data['abs_delta_x'].mean()

        
        smoothness_v2 = smoothness(group_data['x'].values)











    # Werte in das rows-Dictionary übernehmen
        rows.append({
        'chunk_id': chunk_id,
        'geschwindigkeit': velocity_chunk * 1000,
        'geschwindigkeit_min': velocity_min * 1000,
        'geschwindigkeit_max': velocity_max * 1000,
        'geschwindigkeit_mean': velocity_mean * 1000,
        'geschwindigkeit_var': velocity_var * 1000,
        'dauer': duration,
        'direkte_distanz': direct_distance,
        'effizienz': efficiency,
        'totale_distanz': total_distance,
        'smoothness_v1': smoothness_v1,
        'smoothness_v2': smoothness_v2,
        'beschleunigung_min':  acc_min,
        'beschleunigung_max':  acc_max,
        'beschleunigung_mean': acc_mean,
        'beschleunigung_var':  acc_var,

    })

# Finally, build your DataFrame from the rows
    df_temp = pd.DataFrame(rows)

# ------------------------------------------------------
# 4) Z-SCORE UND MINMAX-SCALING FÜR JEDES FEATURE
# ------------------------------------------------------
# Legen Sie fest, welche Spalten Sie skalieren möchten
    columns_to_scale = [
    'geschwindigkeit', 'geschwindigkeit_min', 'geschwindigkeit_max',
    'geschwindigkeit_mean', 'geschwindigkeit_var',
    'beschleunigung_min', 'beschleunigung_max',
    'beschleunigung_mean', 'beschleunigung_var',
    'dauer', 'direkte_distanz', 'effizienz', 'totale_distanz',
    'smoothness_v1', 'smoothness_v2'
]

# 4a) Z-SCORE => (X - mean)/std
    for col in columns_to_scale:
        mean_val = df_temp[col].mean()
        std_val  = df_temp[col].std()
    # Falls std_val=0 => Division durch 0 vermeiden
        if std_val == 0:
            df_temp[col + '_zscore'] = 0
        else:
            df_temp[col + '_zscore'] = (df_temp[col] - mean_val) / std_val

# 4b) MIN-MAX-SCALING => (X - min)/(max - min)
    for col in columns_to_scale:
        min_val = df_temp[col].min()
        max_val = df_temp[col].max()
        if max_val == min_val:
            df_temp[col + '_minmax'] = 0  # oder 1
        else:
            df_temp[col + '_minmax'] = (df_temp[col] - min_val) / (max_val - min_val)

       # Append df_temp to end_df
    bot_df = pd.concat([bot_df, df_temp], ignore_index=True)



bot_df['is_anomaly'] = 1
bot_df = bot_df.dropna()
print("Bot Dataframe Shape")
print(bot_df.shape)



# Combine human and bot dataframes
#end_df = pd.concat([end_df, bot_df], ignore_index=True)

# Print the shape of the combined dataframe
#print("Combined Dataframe Shape")
#print(end_df.shape)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer

# ------------------------------------------------
# 1. Separate the dataset into normal vs anomaly
# ------------------------------------------------
df_all = end_df.copy()  # or rename `end_df` if you prefer

# We don't train on is_anomaly=1 at all.
df_normal = df_all
df_anomaly = bot_df

# ------------------------------------------------
# 2. Train/Test Split - Normal data only
# ------------------------------------------------
# We keep part of the normal data for training, part for testing
# E.g., 80% train, 20% test (only from normal)
train_normal_df, test_normal_df = train_test_split(df_normal, test_size=0.2, random_state=42)

# The anomaly data is *all* used for testing
test_anomaly_df = df_anomaly

# OPTIONAL: If you want to keep chunk_id for reference, you can re-insert or handle it differently.
# For the actual features, typically we drop IDs before modeling.

# ------------------------------------------------
# 3. Prepare features (Imputation, scaling, etc.)
#    for TRAIN
# ------------------------------------------------

# a) Identify the columns you actually want to train on
#    (Exclude chunk_id or any label column.)
feature_cols = ['geschwindigkeit', 'geschwindigkeit_min',
       'geschwindigkeit_max', 'geschwindigkeit_mean', 'geschwindigkeit_var',
       'dauer', 'direkte_distanz', 'effizienz', 'totale_distanz',
       'smoothness_v1', 'smoothness_v2', 'beschleunigung_min',
       'beschleunigung_max', 'beschleunigung_mean', 'beschleunigung_var',
       'geschwindigkeit_zscore', 'geschwindigkeit_min_zscore',
       'geschwindigkeit_max_zscore', 'geschwindigkeit_mean_zscore',
       'geschwindigkeit_var_zscore', 'beschleunigung_min_zscore',
       'beschleunigung_max_zscore', 'beschleunigung_mean_zscore',
       'beschleunigung_var_zscore', 'dauer_zscore', 'direkte_distanz_zscore',
       'effizienz_zscore', 'totale_distanz_zscore', 'smoothness_v1_zscore',
       'smoothness_v2_zscore', 'geschwindigkeit_minmax',
       'geschwindigkeit_min_minmax', 'geschwindigkeit_max_minmax',
       'geschwindigkeit_mean_minmax', 'geschwindigkeit_var_minmax',
       'beschleunigung_min_minmax', 'beschleunigung_max_minmax',
       'beschleunigung_mean_minmax', 'beschleunigung_var_minmax',
       'dauer_minmax', 'direkte_distanz_minmax', 'effizienz_minmax',
       'totale_distanz_minmax', 'smoothness_v1_minmax', 'smoothness_v2_minmax',
       ]

# Extract train features
X_train = train_normal_df[feature_cols]

# Impute missing values (mean) on the training set
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# ------------------------------------------------
# 4. Train the One-Class SVM (Normal data only)
# ------------------------------------------------
clf = OneClassSVM(kernel='rbf', nu=0.1, gamma='auto')
clf.fit(X_train_scaled)

# ------------------------------------------------
# 5. Prepare TEST set (Normal + Anomaly)
# ------------------------------------------------

# (a) Combine normal-test + anomaly into one test DataFrame
test_df_combined = pd.concat([test_normal_df, test_anomaly_df], axis=0)

# Keep track of ground-truth labels (0 = normal, 1 = anomaly)
y_test = np.where(test_df_combined['is_anomaly'] == 1, 1, 0)

# For reference, keep the original index or chunk_id if desired
test_indices = test_df_combined.index

# (b) Build the feature matrix from the same columns
X_test = test_df_combined[feature_cols]

# (c) Impute and scale using the *same* transformers fit on training
X_test_imputed = imputer.transform(X_test)
X_test_scaled  = scaler.transform(X_test_imputed)

# ------------------------------------------------
# 6. Predict and Evaluate
# ------------------------------------------------

# One-Class SVM output: +1 for inliers (normal), -1 for outliers (anomaly)
y_pred_raw = clf.predict(X_test_scaled)

# Map to 0 = normal, 1 = anomaly
y_pred = np.where(y_pred_raw == 1, 0, 1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Identify each category
tn_idx = (y_test == 0) & (y_pred == 0)  # True Normal
fp_idx = (y_test == 0) & (y_pred == 1)  # False Anomaly
fn_idx = (y_test == 1) & (y_pred == 0)  # False Normal
tp_idx = (y_test == 1) & (y_pred == 1)  # True Anomaly

print("\nCounts:")
print("True Normal (TN):", np.sum(tn_idx))
print("False Anomaly (FP):", np.sum(fp_idx))
print("False Normal (FN):", np.sum(fn_idx))
print("True Anomaly (TP):", np.sum(tp_idx))

# Optionally, retrieve original row/indices
tn_rows = test_indices[tn_idx]
fp_rows = test_indices[fp_idx]
fn_rows = test_indices[fn_idx]
tp_rows = test_indices[tp_idx]

print("\nRow Indices:")
print("False Anomaly (FP):", fp_rows.tolist())
print("False Normal  (FN):", fn_rows.tolist())
print("True Anomaly  (TP):", tp_rows.tolist())


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


# -------------------------------
# 7. 3D Visualization with PCA
# -------------------------------
# Reduce the test set to 3 principal components.
pca = PCA(n_components=3)
X_test_pca = pca.fit_transform(X_test_scaled)

fig = go.Figure()

# Plot True Normals in blue.
fig.add_trace(go.Scatter3d(
    x=X_test_pca[tn_idx, 0],
    y=X_test_pca[tn_idx, 1],
    z=X_test_pca[tn_idx, 2],
    mode='markers',
    marker=dict(size=4, color='blue'),
    name='True Normal (TN)'
))

# Plot False Positives (False Anomalies) in orange.
fig.add_trace(go.Scatter3d(
    x=X_test_pca[fp_idx, 0],
    y=X_test_pca[fp_idx, 1],
    z=X_test_pca[fp_idx, 2],
    mode='markers',
    marker=dict(size=4, color='orange'),
    name='False Anomaly (FP)'
))

# Plot False Negatives (False Normals) in green.
fig.add_trace(go.Scatter3d(
    x=X_test_pca[fn_idx, 0],
    y=X_test_pca[fn_idx, 1],
    z=X_test_pca[fn_idx, 2],
    mode='markers',
    marker=dict(size=4, color='green'),
    name='False Normal (FN)'
))

# Plot True Anomalies in red.
fig.add_trace(go.Scatter3d(
    x=X_test_pca[tp_idx, 0],
    y=X_test_pca[tp_idx, 1],
    z=X_test_pca[tp_idx, 2],
    mode='markers',
    marker=dict(size=4, color='red'),
    name='True Anomaly (TP)'
))

fig.update_layout(
    title="3D PCA Visualization of One-Class SVM Results",
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    )
)

fig.show()