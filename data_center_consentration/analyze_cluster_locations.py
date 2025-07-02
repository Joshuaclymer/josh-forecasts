import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the data
df = pd.read_csv('data.csv')

# Check for location information
print(f"Total clusters in dataset: {len(df)}")
print(f"Clusters with location data: {df['Location'].notna().sum()}")
print(f"Clusters with latitude/longitude: {df[['latitude', 'longitude']].notna().all(axis=1).sum()}")

# Look for clusters in the same location
print("\n--- Clusters by Location ---")

# Group by location and count
location_counts = df['Location'].value_counts()
locations_with_multiple = location_counts[location_counts > 1]

print(f"\nNumber of locations with multiple clusters: {len(locations_with_multiple)}")
print("\nTop locations with multiple clusters:")
print(locations_with_multiple.head(10))

# For locations with multiple clusters, show details
print("\n--- Details of Locations with Multiple Clusters ---")
for location, count in locations_with_multiple.head(10).items():
    clusters = df[df['Location'] == location]
    print(f"\nLocation: {location} ({count} clusters)")
    print(clusters[['Name', 'Owner', 'First Operational Date', 'H100 equivalents']].head(10))

# Check for clusters with same coordinates
print("\n--- Clusters by Coordinates ---")
if df[['latitude', 'longitude']].notna().all(axis=1).sum() > 0:
    # Create a location key from lat/long
    df['coord_key'] = df.apply(
        lambda row: f"{row['latitude']:.6f},{row['longitude']:.6f}" 
        if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None, 
        axis=1
    )
    
    # Count clusters per coordinate
    coord_counts = df['coord_key'].value_counts()
    coords_with_multiple = coord_counts[coord_counts > 1]
    
    print(f"\nNumber of coordinates with multiple clusters: {len(coords_with_multiple)}")
    print("\nTop coordinates with multiple clusters:")
    print(coords_with_multiple.head(10))
    
    # For coordinates with multiple clusters, show details
    print("\n--- Details of Coordinates with Multiple Clusters ---")
    for coord, count in coords_with_multiple.head(10).items():
        if pd.isna(coord):
            continue
        clusters = df[df['coord_key'] == coord]
        print(f"\nCoordinates: {coord} ({count} clusters)")
        print(clusters[['Name', 'Owner', 'Location', 'First Operational Date', 'H100 equivalents']].head(10))

# Check for clusters with same owner
print("\n--- Clusters by Owner ---")
owner_counts = df['Owner'].value_counts()
owners_with_multiple = owner_counts[owner_counts > 1]

print(f"\nNumber of owners with multiple clusters: {len(owners_with_multiple)}")
print("\nTop owners with multiple clusters:")
print(owners_with_multiple.head(10))

# Analyze if clusters from the same owner are in the same location
print("\n--- Analysis of Owner's Clusters by Location ---")
for owner, count in owners_with_multiple.head(5).items():
    if pd.isna(owner):
        continue
    owner_clusters = df[df['Owner'] == owner]
    owner_locations = owner_clusters['Location'].value_counts()
    
    print(f"\nOwner: {owner} ({count} clusters)")
    print(f"Number of distinct locations: {len(owner_locations)}")
    print("Top locations:")
    print(owner_locations.head(5))
    
    # Show a sample of clusters for this owner
    print("\nSample clusters:")
    print(owner_clusters[['Name', 'Location', 'First Operational Date', 'H100 equivalents']].head(5))

# Summary
print("\n--- Summary ---")
print(f"Total clusters: {len(df)}")
print(f"Total unique locations: {len(location_counts)}")
print(f"Locations with multiple clusters: {len(locations_with_multiple)} ({len(locations_with_multiple)/len(location_counts)*100:.1f}% of locations)")
print(f"Total clusters in shared locations: {locations_with_multiple.sum()} ({locations_with_multiple.sum()/len(df)*100:.1f}% of clusters)")
