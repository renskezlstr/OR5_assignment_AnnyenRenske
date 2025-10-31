import pandas as pd


routes = {
    1: [91, 86, 111, 108, 103, 77, 74, 82, 52, 31, 30, 25, 26, 47, 39, 21, 18, 37, 42, 49, 51, 56, 64, 63, 66, 90, 92],
    2: [83, 84, 96, 100, 99, 116, 119, 110, 107, 104, 109, 105, 106, 98, 102, 95, 93, 94, 97, 75, 70, 62, 57, 59, 58, 68, 65, 69, 61, 50, 44, 45, 43, 41, 35, 32, 27, 24, 12, 11, 9, 4, 7],
    3: [85, 114, 113, 115, 120, 118, 117, 89, 71, 46, 29, 28, 23, 20, 19, 16, 17, 22, 15, 3, 10, 13, 34, 36, 33, 38, 40, 48, 53, 54, 60, 81, 88, 87, 80, 78, 76],
    4: [101, 112, 79, 67, 72, 73, 55, 8, 1, 14, 6, 5, 2]
}


data = []
for bezorger, locs in routes.items():
    prev = None
    for loc in locs:
        afstand = 0 if prev is None else abs(loc - prev)
        data.append([bezorger, loc, afstand])
        prev = loc

df = pd.DataFrame(data, columns=["bezorger", "locatie", "afstand"])

excel_path = "/mnt/data/bezorgers_routedata.xlsx"
df.to_excel(excel_path, index=False)

excel_path
