import math

# get location data
def get_location():
    crane_location_list = []
    unit_location_list = []
    trailer_location_list = [[4, 6], [4, 10], [17, 1], [22, 4], [22, 9], [22, 14]]

    for x in range(6, 22):
        for y in range(4, 16):
            crane_location_list.append([x, y])
    for x in range(14, 22):
        crane_location_list.append([x, 3])
    for x in range(17, 22):
        crane_location_list.append([x, 2])
    for x in range(20, 22):
        crane_location_list.append([x, 1])
    for x in range(6, 22):
        crane_location_list.append([x, 16])
        crane_location_list.append([x, 17])
    for x in range(8, 22):
        crane_location_list.append([x, 18])
    for x in range(10, 22):
        crane_location_list.append([x, 19])
    for x in range(12, 22):
        crane_location_list.append([x, 20])
    for x in range(13, 22):
        crane_location_list.append([x, 21])
    for x in range(15, 22):
        crane_location_list.append([x, 22])
    for x in range(19, 22):
        crane_location_list.append([x, 23])
    for x in range(7, 9):
        for y in range(5, 9):
            unit_location_list.append([x, y])
            crane_location_list.remove([x, y])
    for x in range(10, 12):
        for y in range(8, 13):
            unit_location_list.append([x, y])
            crane_location_list.remove([x, y])
    for x in range(17, 21):
        for y in range(4, 6):
            unit_location_list.append([x, y])
            crane_location_list.remove([x, y])
    for x in range(15, 20):
        for y in range(8, 10):
            unit_location_list.append([x, y])
            crane_location_list.remove([x, y])
    for x in range(14, 18):
        unit_location_list.append([x, 13])
        crane_location_list.remove([x, 13])
    for x in range(15, 19):
        unit_location_list.append([x, 14])
        crane_location_list.remove([x, 14])
    for x in range(12, 14):
        unit_location_list.append([x, 17])
        crane_location_list.remove([x, 17])
    for x in range(12, 16):
        unit_location_list.append([x, 18])
        crane_location_list.remove([x, 18])
    for x in range(14, 16):
        unit_location_list.append([x, 19])
        crane_location_list.remove([x, 19])

    for i in crane_location_list:
        crane_unit_distance = []
        boolean_list = []
        for x in unit_location_list:
            crane_unit_distance.append(get_distance(i, x))
        for x in crane_unit_distance:
            boolean_list.append(x < 1.5)
        if any(boolean_list):
            crane_location_list.remove(i)
        else:
            continue
    more_location = [
        [6, 4],
        [6, 8],
        [9, 6],
        [9, 12],
        [10, 7],
        [11, 16],
        [12, 7],
        [12, 13],
        [13, 14],
        [14, 9],
        [14, 15],
        [14, 16],
        [15, 7],
        [15, 20],
        [17, 10],
        [17, 12],
        [18, 12],
        [19, 15],
        [20, 7],
    ]
    remove_location = [
        [6, 5],
        [6, 7],
        [9, 5],
        [9, 7],
        [9, 9],
        [9, 11],
        [12, 8],
        [12, 12],
        [14, 3],
        [15, 3],
        [15, 10],
        [16, 5],
        [16, 7],
        [16, 15],
        #[17, 2],
        [17, 3],
        [17, 7],
        [17, 15],
        [18, 7],
        [18, 13],
        [19, 3],
        [19, 11],
        [19, 14],
        # [20, 1],
        # [20, 2],
        [20, 6],
        # [21, 1],
        [21, 5],
        [12, 19],
        [12, 20],
        [15, 21],
        [18, 2],
        # [14, 21],
        [17, 22],
        [10, 18],
        # [10, 19],
        [9, 18],
        # [8, 18],
        # [6, 17],
    ]
    for i in more_location:
        crane_location_list.append(i)
    for i in remove_location:
        crane_location_list.remove(i)
    return crane_location_list, unit_location_list, trailer_location_list


# get distance
def get_distance(location_1, location_2):
    distance_x = location_1[0] - location_2[0]
    distance_y = location_1[1] - location_2[1]
    distance = math.sqrt((distance_x ** 2 + distance_y ** 2))
    return distance


crane_location_list, unit_location_list, trailer_location_list = get_location()
print(unit_location_list)