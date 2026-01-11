def calculate_distance(loc1, loc2):
    return ((loc1.lat - loc2.lat)**2 + (loc1.lng - loc2.lng)**2) ** 0.5
