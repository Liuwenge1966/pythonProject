import math

def calculate_angle(self, vector_a,vector_b):
    dot_product = vector_a[0]*vector_b[0] + vector_a[1]*vector_b[1] + vector_a[2]*vector_b[2]
    magnitude_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[3]**2)
    magnitude_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2 + vector_b[3]**2)
    angle = math.acos(dot_product / (magnitude_a * magnitude_b))
    return math.degrees(angle)

