import vowpalwabbit
import pandas as pd

# region create datasets
# train
train_data = [
    {
        "action": 1,
        "cost": 2,
        "probability": 0.4,
        "feature1": "a",
        "feature2": "c",
        "feature3": "",
    },
    {
        "action": 3,
        "cost": 0,
        "probability": 0.2,
        "feature1": "b",
        "feature2": "d",
        "feature3": "",
    },
    {
        "action": 4,
        "cost": 1,
        "probability": 0.5,
        "feature1": "a",
        "feature2": "b",
        "feature3": "",
    },
    {
        "action": 2,
        "cost": 1,
        "probability": 0.3,
        "feature1": "a",
        "feature2": "b",
        "feature3": "c",
    },
    {
        "action": 3,
        "cost": 1,
        "probability": 0.7,
        "feature1": "a",
        "feature2": "d",
        "feature3": "",
    },
]

train_data_2 = [
    {
        "action": 1,
        "cost": 3,
        "probability": 0.4,
        "feature1": "a",
        "feature2": "c",
        "feature3": "",
    },
    {
        "action": 3,
        "cost": 5,
        "probability": 0.2,
        "feature1": "b",
        "feature2": "d",
        "feature3": "",
    },
    {
        "action": 4,
        "cost": 0,
        "probability": 0.5,
        "feature1": "a",
        "feature2": "b",
        "feature3": "",
    },
]

train_df = pd.DataFrame(train_data)
train_df_2 = pd.DataFrame(train_data_2)

# Add index to data frame
train_df["index"] = range(1, len(train_df) + 1)
train_df = train_df.set_index("index")

# test
test_data = [
    {"feature1": "b", "feature2": "c", "feature3": ""},
    {"feature1": "a", "feature2": "", "feature3": "b"},
    {"feature1": "b", "feature2": "b", "feature3": ""},
    {"feature1": "a", "feature2": "", "feature3": "b"},
]

test_df = pd.DataFrame(test_data)

# Add index to data frame
test_df["index"] = range(1, len(test_df) + 1)
test_df = test_df.set_index("index")
# endregion
# region set up bandit
vw = vowpalwabbit.Workspace("--cb 4", quiet=True)
# endregion
# region learn on train examples
for i, r in train_df.iterrows():
    learn_example = f"{r['action']}:{r['cost']}:{r['probability']} | {r['feature1']} {r['feature2']} {r['feature3']}"
    vw.learn(learn_example)
# endregion
# region predict test cases
for i, r in test_df.iterrows():
    test_example = f"| {r['feature1']} {r['feature2']} {r['feature3']}"
    choice = vw.predict(test_example)
    print(i, choice)
# endregion
# region save and reload model to check it can be called upon again
vw.save("cb.model")
del vw

vw = vowpalwabbit.Workspace("--cb 4 -i cb.model", quiet=True)
print(vw.predict("| a b"))
# endregion
# continue training of model with new training instances and check on test set
for i, r in train_df_2.iterrows():
    learn_example = f"{r['action']}:{r['cost']}:{r['probability']} | {r['feature1']} {r['feature2']} {r['feature3']}"
    vw.learn(learn_example)

for i, r in test_df.iterrows():
    test_example = f"| {r['feature1']} {r['feature2']} {r['feature3']}"
    choice = vw.predict(test_example)
    print(i, choice)

# endregion
