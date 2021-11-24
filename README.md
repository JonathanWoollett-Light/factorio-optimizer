# factorio-optimizer

Lets do [Factorio Calculator](http://kirkmcdonald.github.io/calc.html) but make it optimize.

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
