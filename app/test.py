import pandas as pd

def frank_path():
    with open('data.txt', 'r') as file:
        data_string = file.read()

    rows = data_string.strip().split('\n')
    data = []
    for row in rows:
        elements_ = row.split()
        time = elements_[0]
        northern_limit = f"{elements_[1]} {elements_[2]} {elements_[3]} {elements_[4]}"
        southern_limit = f"{elements_[5]} {elements_[6]} {elements_[7]} {elements_[8]}"
        central_line = f"{elements_[9]} {elements_[10]} {elements_[11]} {elements_[12]}"
        diam_sun = elements_[13]
        sun_path = f"{elements_[14]} {elements_[15]}"
        line = elements_[16]
        time_duration = elements_[17]
        data.append([time, northern_limit, southern_limit, central_line, diam_sun, sun_path, line, time_duration])

    df = pd.DataFrame(data, columns=["Universal", "Northern Limit", "Southern Limit", "Central Line", "Diam. Sun", "Sun Path", "Line", "Duration"])

    central_line = df["Central Line"].str.split(expand=True)
    central_line[0] = central_line[[0, 1]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    central_line[1] = central_line[[2, 3]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)


    central_line = central_line.drop([2, 3], axis=1)
    def convert_to_degrees(coordinate):
        direction = coordinate[-1]
        degrees, minutes = coordinate[:-1].split()
        decimal_degrees = float(degrees) + float(minutes) / 60
        if direction in ['S', 'W']:
            decimal_degrees *= -1
        return decimal_degrees


    for index, row in central_line.iterrows():
        lat, lon = row[0], row[1]
        lat_deg = convert_to_degrees(lat)
        lon_deg = convert_to_degrees(lon)
        central_line.loc[index] = [lat_deg, lon_deg]

    return central_line
