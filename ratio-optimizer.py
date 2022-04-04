import torch
from tqdm import tqdm
import json
import math
import sys
import matplotlib.pyplot as plt

LOAD = False

# Optimizes for a ration of recipes which most evenly produces 1 of the given resource.
# To then expand this to an actual set of recipes/machines to use, we can simply multiply all these ratios 1,2,3,... etc.
#  until the requires recipes are integer values.

# Example inputs
# - `python ratio-optimizer.py space-science-pack`
# - `python ratio-optimizer.py iron-gear-wheel`

# Given a recipe, return the balance of resources it uses/produces (positive being resources it produce, negative being resources it requires).
def to_balance_array(recipe, item_index_dict):
    # print("\n\nrecipe:",recipe)
    outputs = torch.zeros(len(items))
    for item, amount in recipe["outputs"]:
        # print(f"item: {item}, amount: {amount}")
        assert item_index_dict[item] is not None
        outputs[item_index_dict[item]] = amount
    inputs = torch.zeros(len(items))
    for item, amount in recipe["inputs"]:
        assert item_index_dict[item] is not None
        inputs[item_index_dict[item]] = amount
    return outputs - inputs

# required: Resources desired out.
# resources: Resource balance in/out of network (negative being resources in, positive being resources out).
# recipes: Number of recipes performed.
# item_depths: How far each resource is from a raw resource (so we weight our model towards raw resources).
def loss_fn(required, resources, recipes, item_depths):
    # Remaining resources after extracting the required resources.
    remaining = resources - required
    # The deeper (more complex) resources are weighted more heavily (meaning we want to optimize to have a lower over/under flow of these variables).
    depth_weight = item_depths * remaining
    # Penalize by resource usage slightly, penalize by negative resources usage heavily (we would rather produce unused resources than require external resources).
    negative_item_loss = 1 * torch.sum(torch.abs((depth_weight < 0) * depth_weight)) # 100*
    positive_item_loss = 1 * torch.sum((depth_weight > 0) * depth_weight) # 10*
    item_loss = negative_item_loss + positive_item_loss
    # Penalize by number of recipes slightly, penalize by negative recipes heavily (as this is impossible).
    negative_recipe_loss = 0 * torch.sum(torch.abs((recipes < 0) * recipes)) # 1000*
    positive_recipe_loss = 0 * torch.sum((recipes > 0) * recipes) # 1*
    recipe_loss = negative_recipe_loss + positive_recipe_loss

    loss = item_loss + recipe_loss
    return (
        loss,
        item_loss,
        recipe_loss,
        negative_item_loss,
        positive_item_loss,
        negative_recipe_loss,
        positive_recipe_loss,
    )

# Loads items
items_file = open("./base/items.json")
items = json.load(items_file)
print(f"number of items: {len(items)}")
item_index_dict = {}
for index, item in enumerate(items):
    item_index_dict[item] = index

# Loads recipes
recipes_file = open("./base/recipes.json")
recipes = json.load(recipes_file)
print(f"number of recipes: {len(recipes)}")
recipe_balances = torch.stack([to_balance_array(r, item_index_dict) for r in recipes])
# print(f"recipe_balances.shape: {recipe_balances.shape}")

# We use these depths to weight our network in favor of requiring lower level inputs.
# We can approximately calculate the depth of an item by counting the maximum number of steps required to reach a raw resource.
def search_depth(item, covered, recipes, depth=0):
    # print(f"item: {item}")
    # Safety check.
    if depth > 15:
        return 0

    max_depth = depth
    for index, recipe in enumerate(recipes):
        # If already covered this recipe, skip.
        if covered[index]:
            continue
        new_covered = covered.copy()
        new_covered[index] = True
        for output_item, _ in recipe["outputs"]:
            if item == output_item:
                # print(f"item: {item}")
                recipe_inputs = [
                    search_depth(input_item, new_covered, recipes, depth + 1)
                    for input_item, _ in recipe["inputs"]
                ]
                for input_depth in recipe_inputs:
                    max_depth = max(max_depth, input_depth)

    return max_depth


if not LOAD:
    # The depths of each item (how far each item is from raw resources).
    item_depths = torch.tensor(
        [search_depth(item, [False for _ in recipes], recipes) for item in items]
    )
    # print(f"item_depths: {item_depths}")
    item_depths = 100 * torch.pow(item_depths, 1)
    print(f"item_depths (adjusted): {item_depths}")

    # The amount of each recipe we produce.
    multiples = torch.randn(len(recipes), requires_grad=True)
    # The optimizer we use.
    optimizer = torch.optim.Adam([multiples])
    # The learning rate scheduler.
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,threshold=1e2)

    # Sets the required amounts
    required = torch.zeros(len(items))
    if len(sys.argv) > 1:
        print(f"item: {sys.argv[1]}")
        required[item_index_dict[sys.argv[1]]] = 1

    print("Optimizing ratio")
    # EPOCHS = 100
    ITERATIONS = 500000
    UPDATE = 1000
    losses = []
    # for epoch in range(EPOCHS):
    # print(f"Epoch {epoch}")
    loop = tqdm(range(ITERATIONS))
    for i in loop:
        optimizer.zero_grad()
        # Forward
        y_pred = torch.matmul(multiples, recipe_balances)
        loss, il, rl, nil, pil, nrl, prl = loss_fn(
            required, y_pred, multiples, item_depths
        )

        # Backward
        loss.backward()
        optimizer.step()

        if i % UPDATE == 0:
            loop.set_postfix(loss=loss.item())
            losses.append(
                [
                    loss.item(),
                    il.item(),
                    rl.item(),
                    nil.item(),
                    pil.item(),
                    nrl.item(),
                    prl.item(),
                ]
            )
        # scheduler.step(loss)
    torch.save(multiples, "./multiples.pt")

    r = range(0, ITERATIONS, UPDATE)
    plt.plot(r, [l[6] for l in losses], label="Positive Recipe Loss")
    plt.plot(r, [l[5] for l in losses], label="Negative Recipe Loss")
    plt.plot(r, [l[4] for l in losses], label="Positive Item Loss")
    plt.plot(r, [l[3] for l in losses], label="Negative Item Loss")
    plt.plot(r, [l[2] for l in losses], label="Recipe Loss")
    plt.plot(r, [l[1] for l in losses], label="Item Loss")
    plt.plot(r, [l[0] for l in losses], label="Loss")

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.title("Loss' Vs Iteration")
    plt.legend()
    plt.show()  # display the graph

multiples = torch.load("./multiples.pt")
# print(f"multiples: {multiples}")
# normalized_multiple = torch.nn.functional.normalize(multiples,dim=0)

# Overview
print()
print(f"Multiple  │ {'Recipe': <32}│ {'Inputs': <110} │ Outputs")
print('─'*200)
for i, m in enumerate(multiples):
    inputs = [f"{f'{x[1]}·' if x[1]!=1 else ''}{x[0]}" for x in recipes[i]['inputs']]
    joined_inputs = ' + '.join(inputs)
    outputs = [f"{f'{x[1]}·' if x[1]!=1 else ''}{x[0]}" for x in recipes[i]['outputs']]
    joined_outputs = ' + '.join(outputs)
    print(f"{m:+.6f} │ {recipes[i]['name']:.<32}│ {joined_inputs:.<110} │ {joined_outputs}")

# We go through all recipes finding best combination (which most evenly fits)
# print("Finding multiple")
# with torch.no_grad():
#     if len(sys.argv) > 1:
#         item = sys.argv[1]
#         for r in recipes:
#             for i,a in r.inputs
#         index = item_index_dict[sys.argv[1]]
#         print(f"prod = 1 / {multiples[index]}")
#         prod = 1 / multiples[index]
#     print(f"prod: {prod}")
#     amounts = multiples * prod
#     print(f"amounts: {amounts}")
