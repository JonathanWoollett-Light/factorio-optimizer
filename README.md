# factorio-optimizer

An attempt at [Factorio Calculator](http://kirkmcdonald.github.io/calc.html) but generalized.

## WIP

Currently the optimization seems to get stuck, not sure how to fix this, any help would be greatly appreciated.

![Loss Vs Iteration](https://github.com/JonathanWoollett-Light/factorio-optimizer/blob/main/loss-graph.png?raw=true)

```python
def loss_fn(required, resources, recipes, item_depths):
    # Remaining resources after extracting the required resources.
    remaining = resources - required
    # The deeper (more complex) resources are weighted more heavily (meaning we want to optimize to have a lower over/under flow of these variables).
    depth_weight = item_depths * remaining
    # Penalize by resource usage slightly, penalize by negative resources usage heavily (we would rather produced unused resources than require external resources).
    negative_item_loss = 100 * torch.sum(torch.abs((depth_weight < 0) * depth_weight))
    positive_item_loss = 10 * torch.sum((depth_weight > 0) * depth_weight)
    item_loss = negative_item_loss + positive_item_loss
    # Penalize by number of recipes slightly, use abs to push negative or positive to same(as this is impossible).
    negative_recipe_loss = 1000 * torch.sum(torch.abs((recipes < 0) * recipes))
    positive_recipe_loss = torch.sum((recipes > 0) * recipes)
    recipe_loss = negative_recipe_loss + positive_recipe_loss

    loss = item_loss + recipe_loss
    return (loss,item_loss,recipe_loss,negative_item_loss,positive_item_loss,negative_recipe_loss,positive_recipe_loss)
```

## Why not use Factorio Calculator?

- Becuase factorio calculator is not general. 
- The approach it uses only works when everything only has 1 recipe (you can calculate total derivatives, since the change in the number of recipe `a` entirely descrbes the change in the quantity of item `x`).
- When everything has 1 recipe there is only 1 solution  `iron-ore → iron-plate → iron-gear-wheel`, thus when you need `3` `iron-gear-wheel`s you can simply multiply this tree.
- With multiple recipes there are in affect an infinite number of permutations of possible combinations that produce the desired resource (you can only calculate partial derivatives since changes in recipe `a` only partially describes changes in the quantity of item `x`)

## What does this do?

The same thing as factorio calculator, but in a generalized form that can be applied to any number of recipes/mods.

## Why this approach?

1. We can't calculate total derivatives so that shuts off *solving* it, we can only optimize it.
2. Since we can calculate partial derivatives and we have a relatively high dimensionality (the number of recipes, in base game 160) gradient descent allows for more intelligent optimization than an approach like bayesian optimization.
