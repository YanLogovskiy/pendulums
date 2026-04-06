import numpy as np


def rungekut(func, t_0, t_1, x_0):
    num_of_it = 10
    step = (t_1 - t_0) /num_of_it
    x_cur = x_0

    # k_1 = f(x_n, y_n)
    # k_2 = f(x_n + h/2, y_n + h/2 * k_1);
    # k_3 = f(x_n + h/2, y_n + h/2 * k_2);
    # k_4 = f(x_n + h, y_n + h * k_3);

    for i in range(num_of_it):
        k_1 = func(x_cur)
        k_2 = func(np.add(x_cur, np.dot(k_1, step/2)))
        k_3 = func(np.add(x_cur, np.dot(k_2, step/2)))
        k_4 = func(np.add(x_cur, np.dot(k_3, step)))

        # x_{n+1} = x_n + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)

        # k_1 + 2*k_2
        temp_1 = np.add(k_1, np.dot(k_2, 2))
        # 2*k_3 + k_4
        temp_2 = np.add(np.dot(k_3, 2), k_4)
        # h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
        temp_3 = np.dot(np.add(temp_1, temp_2), step/6)

        x_cur = np.add(x_cur, temp_3)

    return x_cur

