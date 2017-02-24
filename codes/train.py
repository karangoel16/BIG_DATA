import os

def main():

    params = {"batchSize": [int, [1, 3]], "learningRate": [float, [1, 3]]}

    for _ in range(10):
        train_args = ""
        for key, values in params:
            value = str(random(values[0], max=values[1]))
            train_args += " --" + key + " " + value
        os.run("main.py" + train_args)

if __name__ == "__main__":
    main()