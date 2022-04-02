### Define model

```{r}
mesh <- inla.mesh.2d(coo, max.edge = c(1000, 5000), cutoff = 100)
plot(mesh)
points(loc %>% select(-idx), col = "red")

spde <- inla.spde2.pcmatern(mesh = mesh,
  prior.range = c(500, 0.01), # P(practic.range < 0.05) = 0.01
  prior.sigma = c(0.1, 0.01)) # P(sigma > 1) = 0.01

# Create projector matrix for SPDE mesh
A.spat = inla.spde.make.A(mesh = mesh,
                          loc = cbind(data$East, data$North),
                          group = data$Year)
A.yrdiff = inla.spde.make.A(mesh = mesh, 
                            loc = cbind(data$East, data$North))

# Assign index vectors for SPDE model.
yrdiff.idx = inla.spde.make.index(name = "yr.diff", 
                                n.spde = spde$n.spde)

spat.idx = inla.spde.make.index(name = "spat.field",
                                n.spde = spde$n.spde,
                                n.group = max(data$Year))

# Make data stack
dat.stack <- inla.stack(
  data = list(y = data$y),
  A = list(1, 1, A.yrdiff, A.spat),
  effects = list(list(Intercept = rep(1, nrow(data))),
                 tibble(Year=data$Year),
                 yrdiff.idx, 
                 spat.idx),
  tag = 'dat')
```

```{r}
form <- y ~ -1 + Intercept + # to fit mu_beta
  f(yr.diff, Year, model = spde, constr = FALSE) + 
  # Year + f(yr.diff, Year, model = spde) + 
  # f(spat.field, model = spde) + 
  f(spat.field, model=spde, group=spat.field.group, 
    control.group = list(model='ar1'))
  # f(time, model = 'ar1')
```
