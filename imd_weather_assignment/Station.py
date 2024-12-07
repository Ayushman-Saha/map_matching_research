import pandas as pd
import numpy as np


def calculate_station_metrics():
    # Read the updated stations data
    data = pd.read_csv("updated_cleaned_stations.csv")

    # Calculate daily rainfall
    data['avg_daily_rainfall'] = data['Mean Monthly Total Rainfall\xa0[\xa0in\xa0mm\xa0]'] / 30

    # Calculate visibility metrics
    visibility_cols = [
        'No. Of Days With Visibility Upto 1 km',
        'No. Of Days With Visibility 1 - 4 km',
        'No. Of Days With Visibility 4 - 10 km',
        'No. Of Days With Visibility 10 - 20 km',
        'No. Of Days With Visibility > 20 km'
    ]

    # Melt visibility data
    melted_visibility = data.melt(
        id_vars=['Station', 'Month'],
        value_vars=visibility_cols,
        var_name='avg_visibility_range',
        value_name='days'
    )

    # Clean up visibility ranges
    melted_visibility['avg_visibility_range'] = \
    melted_visibility['avg_visibility_range'].str.extract(r'(\d.*km|Upto 1 km|> 20 km)')[0]
    melted_visibility['avg_visibility_range'] = melted_visibility['avg_visibility_range'].replace({
        'Upto 1 km': '<1 km',
        '> 20 km': '>20 km'
    })

    # Calculate mean visibility
    def calculate_mean_visibility_range(visibility_range):
        if '<' in visibility_range:
            return 0.5  # Mean of <1 km
        elif '>' in visibility_range:
            return 25  # Mean of >20 km
        else:
            low, high = map(int, visibility_range.replace(' km', '').split('-'))
            return (low + high) / 2

    melted_visibility['M'] = melted_visibility['avg_visibility_range'].apply(calculate_mean_visibility_range)
    melted_visibility['W'] = 1 / melted_visibility['M']
    melted_visibility['Xi'] = melted_visibility['W'] * melted_visibility['days']

    # Calculate monthly visibility averages
    visibility_summary = melted_visibility.groupby(['Station', 'Month'])['Xi'].sum().reset_index()
    visibility_summary['Xi_divided_by_30'] = visibility_summary['Xi'] / 30
    visibility_summary['avg_visibility'] = 1 / visibility_summary['Xi_divided_by_30']

    # Calculate cloud cover metrics
    cloud_cover_cols = [
        'No. Of Days With Cloud Amount (All Clouds) 0 oktas',
        'No. Of Days With Cloud Amount (All Clouds) Trace - 2 oktas',
        'No. Of Days With Cloud Amount (All Clouds) 3 - 5 oktas',
        'No. Of Days With Cloud Amount (All Clouds) 6 - 7 oktas',
        'No. Of Days With Cloud Amount (All Clouds) 8 oktas'
    ]

    # Melt cloud cover data
    melted_cloud = data.melt(
        id_vars=['Station', 'Month'],
        value_vars=cloud_cover_cols,
        var_name='cloud_cover_range',
        value_name='days'
    )

    # Clean up cloud cover ranges
    melted_cloud['cloud_cover_range'] = melted_cloud['cloud_cover_range'].str.extract(r'(\d.*oktas)')[0]
    melted_cloud['cloud_cover_range'] = melted_cloud['cloud_cover_range'].replace({
        '0 oktas': '0',
        '8 oktas': '8',
        '3 - 5 oktas': '3-5',
        '6 - 7 oktas': '6-7',
        '2 oktas': '0-2'
    })

    # Calculate mean cloud cover
    def calculate_mean_cloud_cover(cloud_range):
        if '-' in cloud_range:
            low, high = map(int, cloud_range.split('-'))
            return (low + high) / 2
        else:
            return float(cloud_range)

    melted_cloud['M'] = melted_cloud['cloud_cover_range'].apply(calculate_mean_cloud_cover)
    melted_cloud['W'] = melted_cloud['M']
    melted_cloud['Xi'] = melted_cloud['W'] * melted_cloud['days']

    # Calculate monthly cloud cover averages
    cloud_summary = melted_cloud.groupby(['Station', 'Month'])['Xi'].sum().reset_index()
    cloud_summary['avg_cloud_cover'] = cloud_summary['Xi'] / 30

    # Combine all metrics into final dataset
    final_summary = pd.merge(
        visibility_summary[['Station', 'Month', 'avg_visibility']],
        cloud_summary[['Station', 'Month', 'avg_cloud_cover']],
        on=['Station', 'Month']
    )

    # Add average daily rainfall
    rainfall_summary = data.groupby(['Station', 'Month'])['avg_daily_rainfall'].first().reset_index()
    final_summary = pd.merge(final_summary, rainfall_summary, on=['Station', 'Month'])

    # Add latitude and longitude
    coords = data.groupby('Station')[['latitude', 'longitude']].first().reset_index()
    final_summary = pd.merge(final_summary, coords, on='Station')

    # Save the final summary
    final_summary.to_csv('monthly_station_metrics.csv', index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of stations: {len(final_summary['Station'].unique())}")
    print(f"Number of months: {len(final_summary['Month'].unique())}")
    print("\nAverage values across all stations:")
    print(f"Average visibility: {final_summary['avg_visibility'].mean():.2f} km")
    print(f"Average cloud cover: {final_summary['avg_cloud_cover'].mean():.2f} oktas")
    print(f"Average daily rainfall: {final_summary['avg_daily_rainfall'].mean():.2f} mm")


if __name__ == "__main__":
    calculate_station_metrics()