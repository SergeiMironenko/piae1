import numpy as np
import scipy as sp
from scipy import optimize
import scipy.stats
import matplotlib.pyplot as plt


# Общий класс произвольного критерия оптимальности
class Opt:
    def __init__(self):
        self.value = None
        self.rank = 0
        self.rankorder = 1  # rankorder == 1 (max), rankorder == -1 (min)


# Класс одного плана
class Plan:
    def __init__(self):
        self.D = Opt()
        self.A = Opt()
        self.E = Opt()
        self.F2 = Opt()
        self.L = Opt()
        self.MV = Opt()
        self.G = Opt()

    def get_crit_array(self):
        return [self.D, self.A, self.E, self.F2, self.L, self.MV, self.G]

    def calc_all(self, n, x, theta, M, D):
        self.calc_d(M)
        self.calc_a(D)
        self.calc_e(M)
        self.calc_f2(n, D)
        self.calc_l(D)
        self.calc_mv(D)
        self.calc_g(n, x, theta, D)

    def calc_d(self, M):
        self.D.value = np.linalg.det(M)
        self.D.rankorder = 1

    def calc_a(self, D):
        self.A.value = np.trace(D)
        self.A.rankorder = -1

    def calc_e(self, M):
        lam_vec = np.linalg.eig(M)[0]
        self.E.value = min(lam_vec)
        self.E.rankorder = 1

    def calc_f2(self, n, D):
        D2 = np.linalg.matrix_power(D, 2)
        self.F2.value = pow(np.trace(D2) / n, 0.5)
        self.F2.rankorder = -1

    def calc_l(self, D):
        lam_vec = np.linalg.eig(D)[0]
        lam_ = np.mean(lam_vec)
        self.L.value = sum((lam - lam_) ** 2 for lam in lam_vec)
        self.L.rankorder = -1

    def calc_mv(self, D):
        self.MV.value = max(np.diag(D))
        self.MV.rankorder = -1

    def calc_g(self, n, x, theta, D):
        d = np.zeros(n)
        for i in range(n):
            fx = f(n, x[i], theta)
            fxT = np.transpose(fx)
            d[i] = (fxT.dot(D)).dot(fx)
        self.G.value = max(d)
        self.G.rankorder = -1


# Вычисление вектор-функции f
def f(n, x, theta):
    return np.array([theta[i] * x**i for i in range(n)]).transpose()


# Вычисление матриц M, D
def findMD(n, x, theta, p):
    M = np.zeros((n, n))
    for i in range(n):
        fx = f(n, x[i], theta)
        fxT = np.transpose(fx)
        M += p[i] * np.outer(fx, fxT)
    D = np.linalg.inv(M)
    return M, D


# Вычисление критериев оптимальности
def calc_crit(n, theta, plans):
    plan_array = []
    for plan in plans:
        x, p = plan
        M, D = findMD(n, x, theta, p)

        new_plan = Plan()
        new_plan.calc_all(n, x, theta, M, D)
        plan_array.append(new_plan)
    return plan_array


#  Ранжирование планов
def rank_plan(plan_array):
    # Элемент массива - массив критериев оптимальности
    plan_crit_array = [plan.get_crit_array() for plan in plan_array]

    # Количество планов
    plan_count = len(plan_crit_array)

    # Количество критериев оптимальности
    crit_cound = len(plan_crit_array[0])

    # Ранжирование критериев
    for i in range(crit_cound):
        # Содержит значения всех планов для одного критерия
        crit_array = [plan[i] for plan in plan_crit_array]

        # Вычисление рангов планов для одного критерия
        ranks = sp.stats.rankdata([crit.value * crit.rankorder for crit in crit_array]).astype(int)

        # Присвоение рангов
        for crit, rank in list(zip(crit_array, ranks)):
            crit.rank = rank


# Определение оптимальных значений параметра и критерия
def check_customplan(x, mult, summ, q_min, q_max, q_delta):
    axisX, axisY = [], []
    plan = Plan()
    q = q_min
    i = 1
    while q < q_max:
        p = [summ[i] + mult[i] * q for i in range(n)]
        M, D = findMD(n, x, theta, p)
        plan.calc_f2(n, D)
        axisX.append(q)
        axisY.append(plan.F2.value)

        q = q_min + q_delta * i
        i += 1

    idx = axisY.index(min(axisY))
    draw(axisX, axisY, str(f'min: q = {axisX[idx]:.4}, F2 = {axisY[idx]:.4}'))


#  #####################################  #
#
#  Работа с входными / выходными данными  #
#
#  #####################################  #


# Чтение файла с планами из таблицы
def read_plan(filename):
    n = 0
    theta = []
    plans = []
    file = open(filename, 'r')

    for i, line in enumerate(file):
        if i == 0:
            n = int(line)
        elif i == 1:
            theta = [float(i) for i in line.split()]
        else:
            plan = line.split()
            x = [float(i) for i in plan[0:n]]
            p = [float(i) for i in plan[n:]]
            plans.append([x, p])
    return n, theta, plans


# Чтение своего плана для пункта 4
def read_customplan(filename):
    file = open(filename, 'r')
    x, mult, summ, q_borders = [line_to_array(line) for line in file]
    return x, mult, summ, q_borders


# Получение массива вещественных чисел из строки
def line_to_array(line):
    return [float(num) for num in line.split()]


# Отрисовка графика
def draw(x, y, option=''):
    plt.plot(x, y)
    plt.xlabel('q')
    plt.ylabel('Ф2')
    plt.title(r'Зависимость $Ф_2$ от q')
    plt.grid(True)
    plt.legend([option])
    plt.savefig('graphic.png')


def make_output(plan_array):
    output = open('output.txt', 'w+')
    for plan in plan_array:
        for crit in plan.get_crit_array():
            output.write('{:.4e} ({})   |   '.format(crit.value, crit.rank))
        output.write('\n')


if __name__ == '__main__':
    n, theta, plans = read_plan('model4/plan.txt')
    plan_array = calc_crit(n, theta, plans)
    rank_plan(plan_array)
    make_output(plan_array)

    x, mult, summ, q_borders = read_customplan('model4/customplan2.txt')
    check_customplan(x, mult, summ, *q_borders)
