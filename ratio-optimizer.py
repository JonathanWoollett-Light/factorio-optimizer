import torch
from tqdm import tqdm
import json
import math
import sys

# Optimizes for a ration of recipes which most evenly produces 1 of the given resource.
# To then expand this to an actual set of recipes/machines to use, we can simply multiply all these ratios 1,2,3,... etc.
#  until the requires recipes are integer values.

# Example inputs
# - `python3 ratio-optimizer.py space-science-pack`
# - `python3 ratio-optimizer.py iron-gear-wheel`

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
    # Penalize by resource usage slightly, penalize by negative resources usage heavily (we would rather produced unused resources than require external resources).
    item_loss = torch.sum((depth_weight > 0) * depth_weight) + 10 * torch.sum(
        torch.abs((depth_weight < 0) * depth_weight)
    )
    # Penalize by number of recipes slightly, use abs to push negative or positive to same(as this is impossible).
    recipe_loss = torch.sum((recipes > 0) * recipes) + 100 * torch.sum(
        torch.abs((recipes < 0) * recipes)
    )
    return item_loss + recipe_loss


# Loads items
items_file = open("./base/items.json")
items = json.load(items_file)
print(f"len(items): {len(items)}")
item_index_dict = {}
for index, item in enumerate(items):
    item_index_dict[item] = index

# Loads recipes
recipes_file = open("./base/recipes.json")
recipes = json.load(recipes_file)
print(f"len(recipes): {len(recipes)}")
recipe_balances = torch.stack([to_balance_array(r, item_index_dict) for r in recipes])
print(f"recipe_balances.shape: {recipe_balances.shape}")

# We use these depths to weight our network in favor of requiring possibly more but lower level inputs.
# In affect we can calculate depths by simply counting the maximum number of steps required to reach a raw resource.
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


# The depths of each item (how far each item is from raw resources).
item_depths = 100 * torch.tensor(
    [search_depth(item, [False for _ in recipes], recipes) for item in items]
)
print(f"item_depths: {item_depths}")

# The amount of each recipe we produce.
multiples = torch.randn(len(recipes), requires_grad=True)
# The optimizer we use.
optimizer = torch.optim.Adam([multiples])

# Sets the required amounts
required = torch.zeros(len(items))
print(f"item: {sys.argv[1]}")
required[item_index_dict[sys.argv[1]]] = 1

ITERATIONS = 10000
UPDATE = 1000
loop = tqdm(range(ITERATIONS))
for i in loop:
    # Forward
    y_pred = torch.matmul(multiples, recipe_balances)
    loss = loss_fn(required, y_pred, multiples, item_depths)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % UPDATE == 0:
        loop.set_postfix(loss=loss.item())

for i, m in enumerate(multiples):
    if m >= 0.01:
        print(f"{math.floor(100*m)/100}\t{recipes[i]}")
