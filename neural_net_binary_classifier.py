from matplotlib.pyplot import plot, style, show, contour, subplots, clabel, scatter
from numpy.random import rand
from numpy import array, dot, exp, meshgrid, linspace, dstack, zeros, log, append, ones

style.use('ggplot')

fig, ax = subplots(figsize=(6,6))
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

def plot_data(pairs):
    markers = ['s', '*']

    print(pairs[:num_a,0])
    print(pairs[:num_a,1])
    print(pairs[num_a:,0])
    print(pairs[num_a:,1])
    scatter(pairs[:num_a,0], pairs[:num_a,1], marker='s', c='black')
    scatter(pairs[num_a:,0], pairs[num_a:,1], marker='*', c='black')

def plot_output(weights):
    domain = linspace(0, 10, 10)
    X, Y = meshgrid(domain, domain)
    Z = zeros((10,10))
    for i, x in enumerate(domain):
        for j, y in enumerate(domain):
            Z[i,j] = output([x, y], weights)

    cs = contour(X, Y, Z)
    clabel(cs, inline=1, fontsize=10)

# Output of single neuron
def output(inputs, weights):
    return 1/(1+exp(-dot(inputs, weights[1:]) - weights[0]))

def error_function(weights):
    sum(t*log(output(x, weights)) + (1 - t)*log(1-output(x, weights)) for x, t in zip(inputs, labels))


def update(weights):
    # Compute output for each input
    outputs = array([output(x, weights) for x in inputs])

    # Compute error for each output
    errors = outputs - labels

    g = -[e * x for e, x in zip(errors, inputs)]
    
    # Compute cost function gradient
    Δw = -η * sum(g)

    # Increment weights
    weights += Δw

    return weights


# Number of a and b points
num_a = 10
num_b = 10

# Number of iterations for training
n_iterations = 1

# Randomly generate training pairs for a
inputs = 5*rand(num_a,2)
labels = zeros(num_a)

# Randomly generate training pairs for b
inputs = append(inputs, 5+5*rand(num_b,2), axis=0)
labels = append(labels, ones(num_b))

# Plot data
plot_data(inputs)

weights = array([-0.5, 0.4, -0.2])

plot_output(weights)
#show()

# Train
for i in range(n_iterations):
    weights = update(weights)
    print(error_function(weights))
    

