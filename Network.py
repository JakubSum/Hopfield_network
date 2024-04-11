import numpy as np

# Wzorce liter A, C, X, I
p1 = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1])
p2 = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
p3 = np.array([1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1])
p4 = np.array([-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1])

# Inicjalizacja sieci - przykładowe wagi
weights = np.zeros((len(p1), len(p1)))

# Algorytm
def update_weights(weights, x):
    for i in range(len(x)):
        for j in range(len(x)):
            weights[i][j] += x[i] * x[j]

def activation_function(weights, x):
    return np.sign(np.dot(weights, x))

def recognize_pattern(weights, x, letter):
    max_iterations = 100
    current_iteration = 0
    while True:
        y = activation_function(weights, x)
        if np.array_equal(y, x):
            print(f"Testowany wzorzec ({letter}): {x}")
            print(f"Poprawiony wzorzec ({letter}): {y} (Litera: {get_letter(y)})")
            print("Stabilność osiągnięta.")
            break
        x = y
        current_iteration += 1
        if current_iteration >= max_iterations:
            print(f"Przekroczono maksymalną liczbę iteracji dla wzorca ({letter}).")
            break

def get_letter(pattern):
    # Sprawdzamy, który z wzorców pasuje do rozpoznanego patternu
    if np.array_equal(pattern, p1):
        return 'A'
    elif np.array_equal(pattern, p2):
        return 'C'
    elif np.array_equal(pattern, p3):
        return 'X'
    elif np.array_equal(pattern, p4):
        return 'I'
    else:
        return 'Nieznana litera'

# Uczenie sieci - aktualizacja wag
update_weights(weights, p1)
update_weights(weights, p2)
update_weights(weights, p3)
update_weights(weights, p4)

# Testowanie algorytmu - odtwarzanie wzorców
test_pattern1 = np.array([-1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1])
test_pattern2 = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1])
test_pattern3 = np.array([-1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1])
test_pattern4 = np.array([1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1])
recognize_pattern(weights, test_pattern1, 'Test')
recognize_pattern(weights, test_pattern2, 'Test')
recognize_pattern(weights, test_pattern3, 'Test')
recognize_pattern(weights, test_pattern4, 'Test')
