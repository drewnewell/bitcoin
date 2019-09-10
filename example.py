import random

from pyneat import Environment



def main():
    #fitness_fn = lambda x: random.random()
    fitness_fn = lambda x: x.node_count / 100
    def fitness(organism):
        n_inputs = 1
        outputs = []
        for i in range(n_inputs):
            inputs = [random.random() for i in range(80)]
            outputs.append(sum(organism.get_output_activation(inputs)))
        return max(outputs)
    
    #fitness_fn = fitness
    exp = Environment(population_size=100, fitness_fn=fitness_fn, data_shape=(80, 32), species_desc=(15, 10, 5))

    exp.evolve(generations=100)


if __name__ == '__main__':
    main()
