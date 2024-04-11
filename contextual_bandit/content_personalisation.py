import vowpalwabbit
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# VW tries to minimise loss/cost, so pass cost as -reward
USER_LIKED_ARTICLE = -1.0
USER_DISLIKED_ARTICLE = 0.0


# Simulated cost function
def get_cost(context, action):
    if context["user"] == "Tom":
        if context["time_of_day"] == "morning" and action == "politics":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "music":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context["user"] == "Anna":
        if context["time_of_day"] == "morning" and action == "sports":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "politics":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE


def get_cost_new1(context, action):
    if context["user"] == "Tom":
        if context["time_of_day"] == "morning" and action == "politics":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "sports":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context["user"] == "Anna":
        if context["time_of_day"] == "morning" and action == "sports":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "sports":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE


def get_cost_new2(context, action):
    if context["user"] == "Tom":
        if context["time_of_day"] == "morning" and action == "food":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "sports":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE
    elif context["user"] == "Anna":
        if context["time_of_day"] == "morning" and action == "music":
            return USER_LIKED_ARTICLE
        elif context["time_of_day"] == "afternoon" and action == "food":
            return USER_LIKED_ARTICLE
        else:
            return USER_DISLIKED_ARTICLE


# Modify (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label

    example_string = ""
    example_string += (
        f"shared |User user={context['user']} time_of_day={context['time_of_day']}\n"
    )

    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += f"0:{cost}:{prob} "
        example_string += f"|Action article={action} \n"

    # strip last newline
    return example_string[:-1]


def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if sum_prob > draw:
            return index, prob


def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob


def choose_user(users):
    return random.choice(users)


def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)


def get_preference_matrix(cost_fun, users, times_of_day, actions):
    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    df = expand_grid({"users": users, "times_of_day": times_of_day, "actions": actions})

    df["cost"] = df.apply(
        lambda r: cost_fun({"user": r[0], "time_of_day": r[1]}, r[2]), axis=1
    )

    return df.pivot_table(
        index=["users", "times_of_day"], columns="actions", values="cost"
    )


def run_simulation(
    vw, num_iterations, users, times_of_day, actions, cost_function, do_learn=True
):
    cost_sum = 0.0
    ctr = []

    for i in range(1, num_iterations + 1):
        # choose a user
        user = choose_user(users)
        # choose time of day
        time_of_day = choose_time_of_day(times_of_day)
        # pass context to VW to get an action
        context = {"user": user, "time_of_day": time_of_day}
        action, prob = get_action(vw, context, actions)
        # get cost of the action
        cost = cost_function(context, action)
        cost_sum += cost

        if do_learn:
            # inform VW of what happen so it can learn
            vw_format = vw.parse(
                to_vw_example_format(context, actions, (action, cost, prob)),
                vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,
            )
            # learn
            vw.learn(vw_format)

        # negate so on diagram it's maximising reward
        ctr.append(-1 * cost_sum / i)

    return ctr


def run_simulation_multiple_cost_functions(
    vw, num_iterations, users, times_of_day, actions, cost_functions, do_learn=True
):
    cost_sum = 0.0
    ctr = []

    start_counter = 1
    end_counter = start_counter + num_iterations
    for cost_function in cost_functions:
        for i in range(start_counter, end_counter):
            # choose a user
            user = choose_user(users)
            # choose time of day
            time_of_day = choose_time_of_day(times_of_day)
            # pass context to VW to get an action
            context = {"user": user, "time_of_day": time_of_day}
            action, prob = get_action(vw, context, actions)
            # get cost of the action
            cost = cost_function(context, action)
            cost_sum += cost

            if do_learn:
                # inform VW of what happen so it can learn
                vw_format = vw.parse(
                    to_vw_example_format(context, actions, (action, cost, prob)),
                    vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,
                )
                # learn
                vw.learn(vw_format)

            # negate so on diagram it's maximising reward
            ctr.append(-1 * cost_sum / i)

        start_counter = end_counter
        end_counter = start_counter + num_iterations

    return ctr


def plot_ctr(num_iterations, ctr):
    plt.plot(range(1, num_iterations + 1), ctr)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.ylim([0, 1])
