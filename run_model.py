from model import NonceModel




def main():
    model = NonceModel(restore_model=False, layers=20)
    model.train(epochs=100)


if __name__ == '__main__':
    main()

