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
    item_loss = 10 * torch.sum((depth_weight > 0) * depth_weight) + 100 * torch.sum(
        torch.abs((depth_weight < 0) * depth_weight)
    )
    # Penalize by number of recipes slightly, use abs to push negative or positive to same(as this is impossible).
    recipe_loss = torch.sum((recipes > 0) * recipes) + 1000 * torch.sum(
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
    item_depths = torch.tensor([search_depth(item, [False for _ in recipes], recipes) for item in items])
    # print(f"item_depths: {item_depths}")
    item_depths = 100 * torch.pow(item_depths,1)
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
    EPOCHS = 100
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
        loss = loss_fn(required, y_pred, multiples, item_depths)

        # Backward
        loss.backward()
        optimizer.step()

        if i % UPDATE == 0:
            loop.set_postfix(loss=loss.item())
            losses.append(loss.item())
        # scheduler.step(loss)
    torch.save(multiples,"./multiples.pt")

    plt.plot(losses) #plot the data
    # plt.xticks(range(0,len(data)+1, 1))
    plt.ylabel('Loss') #set the label for y axis
    plt.yscale('log')
    plt.xlabel('Iteration') #set the label for x-axis
    plt.title("Loss Vs Iteration") #set the title of the graph
    plt.show() #display the graph

multiples = torch.load("./multiples.pt")
print(f"multiples: {multiples}")

# Overview
print("\n")
for i, m in enumerate(multiples):
    print(f"{m:.6f}\t{recipes[i]}")

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


        
