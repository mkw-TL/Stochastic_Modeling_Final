# Comp Int Stoch Modeling
# Author: Joe Harrison

# THE INTERGENERATIONAL PROPAGATION OF WEALTH INEQUALITY
# Alan G. Isaac

# xm is a parameter of the pareto. Presumed to be at 100, and just tells us where wealth starts from

# tuna and cprofiler

import math
import random
from bisect import bisect_right
from numpy.random import poisson
from icecream import ic
import matplotlib.pyplot as plt
from statistics import mean, median
import time
import numpy as np
import plotnine
from enum import Enum

# from numba import jit


def alph_from_gini(gini):
    return 1 / (2 * gini) + 1 / 2


def pareto_pdf(alph, k, x):
    return (alph * k ^ alph) / (x ^ (alph + 1))


def pareto_cdf(alph, x, xm=100):
    return 1 - (xm / x) ^ alph


# inverse cdf method to generate pareto distribution
# u = 1 - (xm/y)^a
# ln(1-u) = a * ln(xm/y)
# exp(ln(1-u)/a) = xm/y
# y = xm / exp(ln(1-u)/a)


class MarriageType(Enum):
    ASSORTIVE = 1
    GROUPED = 2
    RANDOM = 3


class DistrType(Enum):
    NOPE = 1
    EQUALLY = 2
    KEEP_IN_SYSTEM = 3


def inv_pareto_cdf(u, alph, xm=100):
    return xm / math.exp(math.log(1 - u) / alph)


# I feel like wikipedia is lying to me
# def calc_gini_coef(incomes):
#     sum = 0
#     denom = 0
#     for i in range(len(incomes)):
#         for j in range(len(incomes)):
#             sum += abs(incomes[i] - incomes[j])
#             denom += incomes[j]
#     return sum / (2 * denom)


# def calc_gini_coef(incomes):
#     sorted_incomes = np.sort(incomes)

#     total_income = np.sum(sorted_incomes)
#     cumulative_income = np.cumsum(sorted_incomes)
#     lorenz_curve = cumulative_income / total_income
#     # equal_spaced = np.linspace(0, 1, len(lorenz_curve))
#     area_under_lorenz_curve = np.trapz(lorenz_curve)  # area under curve
#     return 1 - 2 * area_under_lorenz_curve

# Calculate the Gini coefficient
# return (area_under_lorenz_curve - 0.5) / 0.5


def calc_gini_coef(incomes):
    # ic("our incomes are:", incomes)
    sorted_incomes = np.sort(incomes)
    n = len(incomes)
    i = list(range(1, n + 1))
    weighted = list(map(lambda x, y: x * y, i, sorted_incomes))
    numerator = 2 * sum(weighted)
    denom = n * sum(incomes)
    rat = (n + 1) / n
    return numerator / denom - rat


def calc_hoover_index(incomes):
    mean_inc = mean(incomes)
    sum_x = mean_inc * len(incomes)
    h_numerator = 0
    for i, val in enumerate(incomes):
        h_numerator += abs(incomes[i] - mean_inc)
    h = (h_numerator) / (sum_x)
    return h / 2


def calc_theil_index(incomes):
    mean_inc = mean(incomes)
    T = 0
    for i, val in enumerate(incomes):
        T += (val / mean_inc) * math.log(val / mean_inc)
    return T / len(incomes)


def get_people_wealths(people):
    return [person.getWealth() for person in people]


# def weighted_choice(weights):
#     rnd = random() * sum(weights)
#     for i, w in enumerate(weights):
#         rnd -= w
#         if rnd < 0:
#             return i


# def weighted_shuffle(
#     a, w
# ):  # https://ndvanforeest.github.io/posts/weighted-random-shuffling/
#     r = np.empty_like(a)
#     cumWeights = np.cumsum(w)
#     for i in range(len(a)):
#         rnd = random.random() * cumWeights[-1]
#         j = bisect_right(cumWeights, rnd)
#         # j = np.searchsorted(cumWeights, rnd, side='right')
#         r[i] = a[j]
#         cumWeights[j:] -= w[j]
#     return r


def weighted_shuffle(lst, std):
    weights = [None] * len(lst)
    for i in range(len(lst)):
        weights[i] = i + random.normalvariate(0, std)
    tuples = sorted(zip(weights, lst))
    weights, lst = [t[0] for t in tuples], [t[1] for t in tuples]
    return lst


def marriage(
    people,
    overall_density,
    num,
    assortive=MarriageType.ASSORTIVE,
    pref_towards_males=0.5,
    std=-1,
    random_num_children=True,
    dist_type=DistrType.NOPE,
):
    if std == -1:
        std = num * 0.1  # Note assumption

    pooled_wealth = 0

    females = [person for person in people if person.getSex() == "Female"]
    males = [person for person in people if person.getSex() == "Male"]
    total_children = []
    if assortive == MarriageType.ASSORTIVE:
        females.sort(key=lambda x: x.getWealth())
        males.sort(key=lambda x: x.getWealth())
    elif assortive == MarriageType.RANDOM:
        females = sorted(females, key=lambda x: random.random())
        males = sorted(males, key=lambda x: random.random())
    elif assortive == MarriageType.GROUPED:
        females.sort(key=lambda x: x.getWealth())
        males.sort(key=lambda x: x.getWealth())
        females = weighted_shuffle(females, std)
        males = weighted_shuffle(males, std)
    else:
        Exception("Shouldn't be here...")

    min_length = 0
    if len(females) == len(males):
        min_length = len(females) - 1
    elif len(females) < len(males):
        min_length = len(females) - 1
        if dist_type == DistrType.EQUALLY:
            for j in range(min_length, len(males)):
                pooled_wealth += males[j].getWealth()
        if dist_type == DistrType.KEEP_IN_SYSTEM:
            for j in range(min_length, len(males)):
                total_children.append(males[j])
    elif len(males) < len(females):
        min_length = len(males) - 1
        if dist_type == DistrType.EQUALLY:
            for j in range(min_length, len(females)):
                pooled_wealth += females[j].getWealth()
        if dist_type == DistrType.KEEP_IN_SYSTEM:
            for j in range(min_length, len(females)):
                total_children.append(females[j])

    if min_length == 0:
        Exception("Uh oh, exctinction...")
    for i in range(min_length):
        children = females[i].have_children(
            overall_density,
            num,
            males[i],
            pref_towards_males=pref_towards_males,
            random_num_children=random_num_children,
        )
        if dist_type == DistrType.EQUALLY:
            if len(children) == 0:
                pooled_wealth += females[i].getWealth() + males[i].getWealth()
        total_children.extend(children)
    incr_add = pooled_wealth / len(total_children)
    for child in total_children:
        child.setWealth(child.getWealth() + incr_add)
    return total_children


class Person:
    sex = ""
    wealth = -1
    married = False
    is_poorer = True
    partner = None
    pref_towards_males = 0.5
    random_num_children = True

    def __init__(
        self,
        sex: str,
        wealth: float,
        mx=100,
        pref_towards_males=0.5,
        random_num_children=True,
    ):
        self.sex = sex
        self.wealth = wealth
        if wealth < 3 * mx:
            self.is_poorer = True
        else:
            self.is_poorer = False
        self.pref_towards_males = pref_towards_males
        random_num_children = random_num_children

    def setWealth(self, wealth: float):
        self.wealth = wealth

    def getWealth(self):
        return self.wealth

    def getSex(self):
        return self.sex

    # birth function currently simple poisson with differing amounts of propagating wealth
    # Note wealthy less likely to give birth
    # @jit
    def have_children(
        self,
        overall_density,
        num,
        partner,
        pref_towards_males,
        random_num_children,
        # was .07 and .05
        poor_birth_add=0.07,
        rich_birth_subt=0.05,
    ):
        children = []
        if partner == None:
            Exception("You've goofed (shoulda been implemented in marriage)")
        couple_wealth = self.wealth + partner.getWealth()
        num_girls = 0
        num_boys = 0
        # Here is where we set our birth rates (density dependence)
        lam = 2.0
        x_dist = (overall_density - num) / 100
        if x_dist > 0:
            lam_change = -(x_dist**2) / 50  # this is the strength of the correction
        else:
            lam_change = x_dist**2 / 50
        lam += lam_change  # change direction
        for child in range(
            poisson(lam + poor_birth_add if self.is_poorer else lam - rich_birth_subt)
            if random_num_children
            else 2  # pro coding
        ):
            child_sex = ""
            if random.uniform(0, 1) < 0.5:
                child_sex = "Female"
                num_girls += 1
            else:
                child_sex = "Male"
                num_boys += 1
            children.append(Person(child_sex, -1))
        # avoids floating point issues
        if not (pref_towards_males > 0.999 or pref_towards_males < 0.001):
            perc_to_multiplier = pref_towards_males / (1 - pref_towards_males)
            total_weight = perc_to_multiplier * num_boys + 1 * num_girls
        if num_girls == num_boys == 0:
            if random_num_children == False:
                Exception("goofed")
            pass
        # split evenly
        elif num_girls == 0 or num_boys == 0:
            for child in children:
                child.setWealth(couple_wealth / (num_boys + num_girls))
        # preference given (maybe)
        else:
            for child in children:
                if pref_towards_males == 1:
                    if child.getSex() == "Female":
                        child.setWealth(0)
                    elif child.getSex() == "Male":
                        child.setWealth(couple_wealth / num_boys)
                elif pref_towards_males == 0:
                    if child.getSex() == "Male":
                        child.setWealth(0)
                    elif child.getSex() == "Female":
                        child.setWealth(couple_wealth / num_girls)
                else:
                    if child.getSex() == "Female":
                        child.setWealth(couple_wealth / (total_weight))
                    elif child.getSex() == "Male":
                        child.setWealth(
                            couple_wealth * perc_to_multiplier / (total_weight)
                        )
        return children


def populate(
    N=1000,
    starting_gini_index=0.6,
    pref_towards_males=0.5,
    mx=100,
    random_num_children=True,
):
    people = []
    for i in range(N):
        if random.uniform(0, 1) < 0.5:
            people.append(
                Person(
                    "Female",
                    inv_pareto_cdf(
                        u=random.uniform(0, 1), alph=alph_from_gini(starting_gini_index)
                    ),
                    mx=mx,
                    pref_towards_males=pref_towards_males,
                    random_num_children=random_num_children,
                )
            )
        else:
            people.append(
                Person(
                    "Male",
                    inv_pareto_cdf(
                        u=random.uniform(0, 1), alph=alph_from_gini(starting_gini_index)
                    ),
                    mx=mx,
                    pref_towards_males=pref_towards_males,
                )
            )
    return people


def run_forwards(
    time_start,
    steps=500,
    N=1000,
    starting_gini_index=0.6,
    pref_towards_males=0.5,
    mx=100,
    assortive=MarriageType.ASSORTIVE,
    std=-1,
    random_num_children=True,
    dist_type=DistrType.NOPE,
):
    people = populate(
        N,
        starting_gini_index,
        pref_towards_males=pref_towards_males,
        mx=mx,
        random_num_children=random_num_children,
    )
    gini_coefs = []
    hoovers = []
    theils = []
    wealth_over_time = []
    prev_pop = N
    for i in range(steps):
        if i % 20 == 0:
            ic(i)
            ic(time.time() - time_start)
        people = marriage(
            people,
            assortive=assortive,
            pref_towards_males=pref_towards_males,
            random_num_children=random_num_children,
            overall_density=prev_pop,
            num=N,
            std=std,
            dist_type=dist_type,
        )
        wealth_over_time.append(sum(get_people_wealths(people)))
        new_gini_coef = calc_gini_coef(get_people_wealths(people))
        new_hoover_index = calc_hoover_index(get_people_wealths(people))
        # new_theil_index = calc_theil_index(get_people_wealths(people))
        gini_coefs.append(new_gini_coef)
        hoovers.append(new_hoover_index)
        # theils.append(new_theil_index)
        prev_pop = len(people)
    return gini_coefs, hoovers, wealth_over_time


def debug(
    N,
    starting_gini_index,
    pref_towards_males,
    random_num_children,
    mx,
    assortive,
    std,
    steps,
):
    people = populate(
        N,
        starting_gini_index,
        pref_towards_males=pref_towards_males,
        mx=mx,
        random_num_children=random_num_children,
    )
    gini_coefs = []
    hoovers = []
    theils = []
    wealth_over_time = []
    min_vals = []
    max_vals = []
    prev_pops = []
    prev_pop = N
    for i in range(steps):
        people = marriage(
            people,
            assortive=assortive,
            pref_towards_males=pref_towards_males,
            random_num_children=random_num_children,
            overall_density=prev_pop,
            num=N,
            std=std,
            dist_wealth=DistrType.EQUALLY,
        )
        min_val = min(get_people_wealths(people))
        min_vals.append(min_val)
        max_val = max(get_people_wealths(people))
        max_vals.append(max_val)
        thisList = sorted(get_people_wealths(people), reverse=True)
        # newList = [x / min_val for x in thisList]
        wealth_over_time.append(sum(get_people_wealths(people)))
        new_gini_coef = calc_gini_coef(get_people_wealths(people))
        new_hoover_index = calc_hoover_index(get_people_wealths(people))
        # new_theil_index = calc_theil_index(get_people_wealths(people))
        gini_coefs.append(new_gini_coef)
        hoovers.append(new_hoover_index)
        # theils.append(new_theil_index)
        prev_pop = len(people)
        prev_pops.append(prev_pop)
    plt.plot(list(range(len(gini_coefs))), gini_coefs)
    plt.show()
    # plt.plot(list(range(len(min_vals))), min_vals)
    # plt.plot(list(range(len(max_vals))), max_vals)
    plt.plot(list(range(len(prev_pops))), prev_pops)
    plt.show()


def plotting(
    time_start,
    num_pops,
    start_gini,
    steps,
    assortive,
    N,
    male_pref,
    random_num_children,
    label,
    color,
    alpha,
    dist_type,
    std=-1,
):
    gini_recs = [None] * num_pops
    hoover_idxs = [None] * num_pops
    wealth_recs = [None] * num_pops
    for i in range(num_pops):
        (
            gini_recs[i],
            hoover_idxs[i],
            wealth_recs[i],
        ) = run_forwards(
            time_start=time_start,
            steps=steps,
            assortive=assortive,
            starting_gini_index=start_gini,
            N=N,
            pref_towards_males=male_pref,
            std=std,  # don't specify
            random_num_children=random_num_children,
            dist_type=dist_type,
        )
    x_line = list(range(steps))
    avg_ginis = [mean(x) for x in zip(*gini_recs)]
    rand_avg_hoover = [mean(x) for x in zip(*hoover_idxs)]
    linestyle = "solid"
    if assortive == MarriageType.RANDOM:
        linestyle = (0, (5, 10))
    elif assortive == MarriageType.GROUPED:
        linestyle = (0, (5, 1))
    plt.plot(
        x_line,
        rand_avg_hoover,
        label=label,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
    )


def plot(time_start, num_pops, start_gini, steps, N, dist_type):
    plt.figure(figsize=(8, 6))
    show_rand = True
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.ASSORTIVE,
        N=N,
        male_pref=1,
        random_num_children=True,
        label="M, assort",
        color="C0",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.GROUPED,
        N=N,
        male_pref=1,
        random_num_children=True,
        label="M, grouped",
        color="C0",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.RANDOM,
        N=N,
        male_pref=1,
        random_num_children=True,
        label="M, random",
        color="C0",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.ASSORTIVE,
        N=N,
        male_pref=0.5,
        random_num_children=True,
        label="Eq, assort",
        color="C1",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.GROUPED,
        N=N,
        male_pref=0.5,
        random_num_children=True,
        label="Eq, grouped",
        color="C1",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.RANDOM,
        N=N,
        male_pref=0.5,
        random_num_children=True,
        label="Eq, random",
        color="C1",
        alpha=1 if show_rand else 0.1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.ASSORTIVE,
        N=N,
        male_pref=1,
        random_num_children=False,
        label="Const, M, assort",
        color="C0",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.GROUPED,
        N=N,
        male_pref=1,
        random_num_children=False,
        label="Const, M, grouped",
        color="C0",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.RANDOM,
        N=N,
        male_pref=1,
        random_num_children=False,
        label="Const, M, random",
        color="C0",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.ASSORTIVE,
        N=N,
        male_pref=0.5,
        random_num_children=False,
        label="Const, Eq, assort",
        color="C1",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.GROUPED,
        N=N,
        male_pref=0.5,
        random_num_children=False,
        label="Const, Eq, grouped",
        color="C1",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plotting(
        time_start,
        num_pops,
        start_gini,
        steps,
        assortive=MarriageType.RANDOM,
        N=N,
        male_pref=0.5,
        random_num_children=False,
        label="Const, Eq, random",
        color="C1",
        alpha=0.1 if show_rand else 1,
        dist_type=dist_type,
    )
    plt.xlabel("Number of Generations")
    plt.ylabel("Hover Score")
    plt.title("Hoover Score removing those that did not marry")
    plt.ylim(0, 1)
    plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.show()
    plt.savefig("plot.png", bbox_inches="tight", pad_inches=0.1)


def main(steps, num_pops):
    time_start = time.time()
    start_gini = 0.6
    N = 1000
    plot(time_start, num_pops, start_gini, steps, N, dist_type=DistrType.NOPE)


if __name__ == "__main__":
    # debug(400, 0.6, 1, False, 100, MarriageType.RANDOM, -1, 20)
    main(steps=30, num_pops=10)
